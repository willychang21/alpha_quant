import pandas as pd
import logging
from typing import List, Dict, Any
from datetime import date
from sqlalchemy.orm import Session
from quant.data.models import PortfolioTargets, Security, ModelSignals
import json

logger = logging.getLogger(__name__)

class TradeListGenerator:
    """
    Generates a weekly trade list for manual execution.
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        
    def generate_trade_list(self, 
                            target_date: date, 
                            model_name: str = 'kelly_v1', 
                            capital: float = 100000.0) -> pd.DataFrame:
        """
        Generate a trade list based on portfolio targets.
        
        Args:
            target_date: Date of the optimization.
            model_name: Name of the model (e.g. 'kelly_v1').
            capital: Total capital to allocate (default $100k).
            
        Returns:
            pd.DataFrame: Trade list with columns [Ticker, Name, Sector, Weight, Shares, Value, Action]
        """
        logger.info(f"Generating trade list for {target_date} using {model_name}...")
        
        # 1. Fetch Targets
        targets = self.db.query(PortfolioTargets)\
            .filter(PortfolioTargets.date == target_date, PortfolioTargets.model_name == model_name)\
            .all()
            
        if not targets:
            logger.warning(f"No targets found for {target_date} and model {model_name}")
            return pd.DataFrame()
            
        # 2. Fetch Security Info & Metadata (for Sector)
        # We need to join with ModelSignals to get Sector if possible, or just use what we have
        # ModelSignals might be on the same date
        
        trade_list = []
        
        for target in targets:
            sec = target.security
            weight = target.weight
            
            if weight <= 0.001: # Skip negligible weights
                continue
                
            # Try to find sector from recent signals
            signal = self.db.query(ModelSignals)\
                .filter(ModelSignals.sid == sec.sid, ModelSignals.date == target_date)\
                .first()
                
            sector = "Unknown"
            if signal and signal.metadata_json:
                try:
                    meta = json.loads(signal.metadata_json)
                    sector = meta.get('sector', 'Unknown')
                except:
                    pass
            
            # Calculate Shares
            # We need current price. 
            # We can use the price from MarketDataDaily if available, or fetch it.
            # For this report, we'll assume price is roughly what was used in optimization.
            # Or better, fetch latest price.
            
            # Fetch latest price from DB or assume we have it
            # For simplicity in this generator, we'll skip exact share calculation if price is missing
            # or use a placeholder.
            # Ideally, we should have price in the DB from the daily job run.
            
            from quant.data.models import MarketDataDaily
            price_rec = self.db.query(MarketDataDaily)\
                .filter(MarketDataDaily.sid == sec.sid, MarketDataDaily.date == target_date)\
                .first()
                
            price = price_rec.close if price_rec else 0.0
            
            # Fallback to YFinance if price is missing
            if price == 0.0:
                try:
                    import yfinance as yf
                    ticker_obj = yf.Ticker(sec.ticker)
                    # Try fast info first
                    price = ticker_obj.fast_info.last_price
                    if not price:
                        hist = ticker_obj.history(period="1d")
                        if not hist.empty:
                            price = hist['Close'].iloc[-1]
                except Exception as e:
                    logger.warning(f"Failed to fetch price for {sec.ticker}: {e}")
            
            value = weight * capital
            shares = int(value / price) if price > 0 else 0
            
            trade_list.append({
                'Ticker': sec.ticker,
                'Name': sec.name,
                'Sector': sector,
                'Weight': weight,
                'Value': value,
                'Price': price,
                'Shares': shares,
                'Action': 'BUY' # Assuming we are building from scratch
            })
            
        df = pd.DataFrame(trade_list)
        if not df.empty:
            df = df.sort_values('Weight', ascending=False)
            
        return df
