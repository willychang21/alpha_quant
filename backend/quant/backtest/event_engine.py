"""Event-Driven Backtester.

Complete implementation of event-driven backtest with:
- Chronological event processing
- Fill model integration
- Benchmark comparison in results
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional
import logging

from quant.backtest.interface import BacktestEngine, Strategy, BacktestResult
from quant.backtest.execution.fill_model import FillModel, LiquidityConstrainedFill
from quant.backtest.execution.slippage import SlippageModel, FixedSlippage
from quant.backtest.benchmark import calculate_benchmark_metrics

logger = logging.getLogger(__name__)


class EventDrivenBacktester(BacktestEngine):
    """Event-driven backtester with fill model and benchmark metrics.
    
    Iterates through time chronologically, allowing strategy to react
    to each day's data and generating orders that are executed through
    the fill model.
    """
    
    def __init__(
        self,
        data_provider=None,
        fill_model: Optional[FillModel] = None,
        slippage_model: Optional[SlippageModel] = None,
        slippage_bps: float = 10.0,
        commission_rate: float = 0.0005
    ):
        self.data_provider = data_provider
        self.fill_model = fill_model or LiquidityConstrainedFill(max_participation=0.1)
        self.slippage_model = slippage_model or FixedSlippage(spread_bps=slippage_bps)
        self.commission_rate = commission_rate
        
        # Portfolio state
        self.cash = 0.0
        self.holdings: Dict[str, float] = {}
        self.history: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        self.unfilled: Dict[str, float] = {}
    
    def run(
        self, 
        strategy: Strategy, 
        start_date: datetime, 
        end_date: datetime, 
        initial_capital: float
    ) -> BacktestResult:
        """Run event-driven backtest.
        
        Args:
            strategy: Trading strategy instance
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
        
        Returns:
            BacktestResult with equity_curve, metrics, and trades
        """
        # Initialize portfolio
        self.cash = initial_capital
        self.holdings = {}
        self.history = []
        self.trades = []
        self.unfilled = {}
        
        # Get trading dates
        dates = self._get_trading_dates(start_date, end_date)
        
        if not dates:
            logger.warning("No trading dates in range")
            return BacktestResult()
        
        # Event loop - process dates chronologically
        for current_date in dates:
            # 1. Get market snapshot for this date
            current_data = self._get_market_snapshot(current_date)
            
            # 2. Get current portfolio state
            portfolio_state = self._get_portfolio_state(current_date)
            
            # 3. Strategy generates orders based on data
            orders = strategy.on_data(current_data, portfolio_state)
            
            # 4. Execute orders through fill model
            self._execute_orders(orders, current_date)
            
            # 5. Log daily state
            self._log_state(current_date)
        
        # Generate result with metrics
        return self._generate_result()
    
    def _get_trading_dates(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[date]:
        """Get list of trading dates in range.
        
        Returns dates in strictly ascending chronological order.
        """
        # Simple implementation: generate all weekdays
        dates = []
        current = start_date
        
        if isinstance(current, datetime):
            current = current.date() if hasattr(current, 'date') else current
        if isinstance(end_date, datetime):
            end_date = end_date.date() if hasattr(end_date, 'date') else end_date
        
        while current <= end_date:
            # Skip weekends (0=Monday, 6=Sunday)
            if current.weekday() < 5:
                dates.append(current)
            current += timedelta(days=1)
        
        return dates
    
    def _get_market_snapshot(self, current_date: date) -> Dict[str, Any]:
        """Get market data snapshot for date.
        
        In production, this would fetch prices/volumes from data provider.
        """
        snapshot = {
            'date': current_date,
            'prices': {},
            'volumes': {}
        }
        
        if self.data_provider:
            # Fetch data for held tickers
            for ticker in self.holdings.keys():
                price = self.data_provider.get_price(ticker, current_date)
                if price:
                    snapshot['prices'][ticker] = price
        
        return snapshot
    
    def _get_portfolio_state(self, current_date: date) -> Dict[str, Any]:
        """Get current portfolio state."""
        equity = self._calculate_equity(current_date)
        
        return {
            'cash': self.cash,
            'holdings': self.holdings.copy(),
            'equity': equity,
            'date': current_date
        }
    
    def _execute_orders(self, orders: List[Dict], current_date: date):
        """Execute orders through fill model."""
        if not orders:
            return
        
        for order in orders:
            ticker = order.get('ticker')
            quantity = order.get('quantity', 0)
            
            if not ticker or quantity == 0:
                continue
            
            # Get price
            price = self._get_price(ticker, current_date)
            if not price or price <= 0:
                logger.debug(f"No price for {ticker} on {current_date}")
                continue
            
            # Get volume (if available)
            volume = order.get('volume')
            
            # Apply fill model
            filled_qty = self.fill_model.get_fill_quantity(quantity, volume)
            remaining = quantity - filled_qty
            
            # Track unfilled
            if remaining != 0:
                self.unfilled[ticker] = self.unfilled.get(ticker, 0) + remaining
            
            if filled_qty == 0:
                continue
            
            # Apply slippage
            exec_price = self.slippage_model.calculate_price(price, filled_qty)
            
            # Execute fill
            self._execute_fill(ticker, filled_qty, exec_price, current_date)
    
    def _execute_fill(
        self, 
        ticker: str, 
        quantity: float, 
        price: float,
        current_date: date
    ):
        """Execute a single fill."""
        trade_value = abs(quantity * price)
        commission = trade_value * self.commission_rate
        
        if quantity > 0:  # Buy
            total_cost = (quantity * price) + commission
            if self.cash >= total_cost:
                self.cash -= total_cost
                self.holdings[ticker] = self.holdings.get(ticker, 0) + quantity
                self._record_trade(ticker, quantity, price, commission, current_date, 'BUY')
        else:  # Sell
            proceeds = (abs(quantity) * price) - commission
            self.cash += proceeds
            self.holdings[ticker] = self.holdings.get(ticker, 0) + quantity
            
            # Cleanup zero positions
            if abs(self.holdings.get(ticker, 0)) < 1e-6:
                del self.holdings[ticker]
            
            self._record_trade(ticker, quantity, price, commission, current_date, 'SELL')
    
    def _record_trade(
        self, 
        ticker: str, 
        quantity: float, 
        price: float, 
        commission: float,
        trade_date: date,
        side: str
    ):
        """Record a trade."""
        self.trades.append({
            'date': trade_date,
            'ticker': ticker,
            'side': side,
            'quantity': abs(quantity),
            'price': price,
            'commission': commission,
            'value': abs(quantity * price)
        })
    
    def _get_price(self, ticker: str, target_date: date) -> Optional[float]:
        """Get price for ticker on date."""
        if self.data_provider:
            return self.data_provider.get_price(ticker, target_date)
        return None
    
    def _calculate_equity(self, current_date: date) -> float:
        """Calculate total portfolio value."""
        equity = self.cash
        for ticker, qty in self.holdings.items():
            price = self._get_price(ticker, current_date)
            if price:
                equity += qty * price
        return equity
    
    def _log_state(self, current_date: date):
        """Log daily portfolio state."""
        equity = self._calculate_equity(current_date)
        self.history.append({
            'date': current_date,
            'equity': equity,
            'cash': self.cash,
            'holdings_count': len(self.holdings)
        })
    
    def _generate_result(self) -> BacktestResult:
        """Generate BacktestResult with all metrics."""
        if not self.history:
            return BacktestResult()
        
        df = pd.DataFrame(self.history)
        df['returns'] = df['equity'].pct_change().fillna(0)
        
        # Base metrics
        metrics = self._calculate_base_metrics(df)
        
        # Benchmark metrics
        if len(df) > 20:
            try:
                returns_series = df.set_index('date')['returns']
                if not isinstance(returns_series.index, pd.DatetimeIndex):
                    returns_series.index = pd.to_datetime(returns_series.index)
                
                benchmark_metrics = calculate_benchmark_metrics(
                    returns_series,
                    benchmark='SPY'
                )
                metrics.update(benchmark_metrics)
            except Exception as e:
                logger.debug(f"Could not calculate benchmark metrics: {e}")
        
        return BacktestResult(
            equity_curve=df['equity'],
            metrics=metrics,
            trades=self.trades
        )
    
    def _calculate_base_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate base performance metrics."""
        if df.empty or 'equity' not in df.columns:
            return {}
        
        total_return = (df['equity'].iloc[-1] / df['equity'].iloc[0]) - 1
        
        # CAGR
        days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
        if days > 0:
            cagr = (1 + total_return) ** (365 / days) - 1
        else:
            cagr = 0
        
        # Volatility
        volatility = df['returns'].std() * np.sqrt(252)
        
        # Sharpe
        rf = 0.04
        sharpe = (cagr - rf) / volatility if volatility > 0 else 0
        
        # Max Drawdown
        df['cummax'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['cummax']) / df['cummax']
        max_drawdown = df['drawdown'].min()
        
        return {
            'total_return': float(total_return),
            'cagr': float(cagr),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'num_trades': len(self.trades)
        }
