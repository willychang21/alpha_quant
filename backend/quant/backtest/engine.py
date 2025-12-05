from sqlalchemy.orm import Session
from quant.data.models import Security, MarketDataDaily
from quant.data.data_provider import DataProvider, SQLiteDataProvider, create_data_provider
from quant.selection.ranking import RankingEngine
from quant.portfolio.optimizer import PortfolioOptimizer
from quant.backtest.execution.slippage import SlippageModel, FixedSlippage, VolumeShareSlippage
from datetime import date, timedelta
from typing import Optional, Union
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Production-grade backtesting engine.
    
    Supports both legacy SQLAlchemy sessions and new DataProvider abstraction.
    Uses DataProvider for efficient price lookups with caching.
    
    Usage:
        # New way (recommended)
        provider = create_data_provider(provider_type='parquet', data_lake_path='...')
        engine = BacktestEngine(data_provider=provider)
        
        # Legacy way (backward compatible)
        engine = BacktestEngine(db=session)
    """
    
    def __init__(
        self, 
        db: Session = None,
        data_provider: DataProvider = None,
        initial_capital: float = 100000.0,
        slippage_model: SlippageModel = None,
        commission_rate: float = 0.0  # e.g. 0.0005 for 5bps
    ):
        # Support both old and new interface
        if data_provider is not None:
            self.data_provider = data_provider
            self.db = None
        elif db is not None:
            self.data_provider = SQLiteDataProvider(db)
            self.db = db
        else:
            raise ValueError("Either 'db' or 'data_provider' must be provided")
        
        # For ranking and optimization, we still need db for now
        # TODO: Migrate these to use DataProvider as well
        self._legacy_db = db
        if db is not None:
            self.ranking_engine = RankingEngine(db)
            self.optimizer = PortfolioOptimizer(db)
        else:
            # If using DataProvider without db, need to handle differently
            self.ranking_engine = None
            self.optimizer = None
        
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.holdings = {}  # ticker -> quantity
        
        self.slippage_model = slippage_model if slippage_model else FixedSlippage(spread_bps=10)
        self.commission_rate = commission_rate
        
        self.history = []


    def run_backtest(self, start_date: date, end_date: date, rebalance_freq_days: int = 30):
        """
        Run a historical simulation with realistic execution.
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}...")
        
        current_date = start_date
        next_rebalance = current_date
        
        # Initial State Log
        self._log_state(current_date)
        
        while current_date < end_date:
            # 1. Check for Rebalance
            if current_date >= next_rebalance:
                logger.info(f"Rebalancing on {current_date}")
                self._rebalance(current_date)
                next_rebalance = current_date + timedelta(days=rebalance_freq_days)
            
            # 2. Step Forward (Daily)
            # In a real event-driven engine, we'd process every day.
            # For speed, we can step daily or stick to monthly if we only rebalance monthly.
            # However, to track daily equity curve properly, we should step daily or use a proxy.
            # Let's step daily to be "Hedge Fund Grade" accurate.
            
            current_date += timedelta(days=1)
            if current_date > end_date:
                break
                
            # Log Daily Value (Mark to Market)
            self._log_state(current_date)
            
        return pd.DataFrame(self.history)

    def _rebalance(self, date: date):
        try:
            # 1. Rank
            self.ranking_engine.run_ranking(date)
            
            # 2. Optimize
            allocations = self.optimizer.run_optimization(date)
            
            if not allocations:
                logger.warning(f"No allocations generated on {date}. Holding positions.")
                return

            # 3. Execute Trades
            # Target Portfolio Value = Current Equity
            equity = self._calculate_equity(date)
            
            target_holdings = {} # ticker -> quantity
            
            # Calculate target quantities
            for ticker, weight in allocations:
                price = self._get_price(ticker, date)
                if price and price > 0:
                    target_val = equity * weight
                    qty = target_val / price
                    target_holdings[ticker] = qty
            
            # Generate Orders (Diff)
            # Sell first to raise cash
            for ticker, current_qty in list(self.holdings.items()):
                target_qty = target_holdings.get(ticker, 0)
                if target_qty < current_qty:
                    sell_qty = current_qty - target_qty
                    self._execute_order(ticker, -sell_qty, date)
            
            # Buy next
            for ticker, target_qty in target_holdings.items():
                current_qty = self.holdings.get(ticker, 0)
                if target_qty > current_qty:
                    buy_qty = target_qty - current_qty
                    self._execute_order(ticker, buy_qty, date)
                    
        except Exception as e:
            logger.error(f"Rebalance failed on {date}: {e}")

    def _execute_order(self, ticker: str, quantity: float, date: date):
        if quantity == 0:
            return

        price = self._get_price(ticker, date)
        if not price or price <= 0:
            logger.warning(f"Cannot execute {ticker}: No price on {date}")
            return

        # Apply Slippage
        # We need volume/volatility for advanced models, passing None for now (FixedSlippage ignores them)
        exec_price = self.slippage_model.calculate_price(price, quantity)
        
        trade_value = abs(quantity * exec_price)
        commission = trade_value * self.commission_rate
        
        cost = (quantity * exec_price) + commission # Positive for buy, Negative (but add commission) for sell?
        # Wait:
        # Buy: Cash -= (Qty * Price) + Comm
        # Sell: Cash += (Qty * Price) - Comm
        
        if quantity > 0: # Buy
            total_cost = (quantity * exec_price) + commission
            if self.cash >= total_cost:
                self.cash -= total_cost
                self.holdings[ticker] = self.holdings.get(ticker, 0) + quantity
            else:
                # Partial fill or skip
                # logger.warning(f"Insufficient cash for {ticker}")
                pass 
        else: # Sell
            proceeds = (abs(quantity) * exec_price) - commission
            self.cash += proceeds
            self.holdings[ticker] = self.holdings.get(ticker, 0) + quantity
            
            # Cleanup small positions
            if abs(self.holdings[ticker]) < 1e-6:
                del self.holdings[ticker]

    def _calculate_equity(self, date: date) -> float:
        equity = self.cash
        for ticker, qty in self.holdings.items():
            price = self._get_price(ticker, date)
            if price:
                equity += qty * price
        return equity

    def _log_state(self, date: date):
        equity = self._calculate_equity(date)
        self.history.append({
            "date": date,
            "equity": equity,
            "cash": self.cash,
            "holdings_count": len(self.holdings)
        })

    def _get_price(self, ticker: str, target_date: date):
        """
        Get price using DataProvider (with caching).
        
        Uses the new DataProvider abstraction which handles:
        - Efficient caching for sequential backtest queries
        - Point-in-Time correct data access
        """
        return self.data_provider.get_price(ticker, target_date)

