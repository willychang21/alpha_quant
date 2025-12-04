from sqlalchemy.orm import Session
from quant.data.models import Security, MarketDataDaily, Fundamentals
from sqlalchemy import func
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class FactorEngine:
    def __init__(self, db: Session):
        self.db = db

    def calculate_momentum(self, sid: int, date):
        """
        Calculate 12M-1M Momentum.
        Return (today_price / price_12m_ago) - 1, excluding last month?
        Standard: (Price(t-1M) / Price(t-12M)) - 1
        """
        # Fetch prices
        # Optimization: In production, load time-series into pandas/numpy for vectorized calc.
        # Here, we do simple SQL queries for MVP.
        
        # Get price 1 month ago
        date_1m = date - pd.Timedelta(days=30)
        price_1m = self._get_price(sid, date_1m)
        
        # Get price 12 months ago
        date_12m = date - pd.Timedelta(days=365)
        price_12m = self._get_price(sid, date_12m)
        
        if price_1m and price_12m and price_12m > 0:
            return (price_1m / price_12m) - 1
        return None

    def calculate_value_factors(self, sid: int, date):
        """
        Calculate P/E, P/B, EV/EBITDA.
        Needs latest price and latest fundamentals relative to 'date'.
        """
        price = self._get_price(sid, date)
        if not price:
            return {}
            
        # Get latest fundamentals before date
        # We need 'Basic EPS', 'Book Value', 'EBITDA', 'Total Debt', 'Cash And Cash Equivalents', 'Ordinary Shares Number'
        
        metrics = ['Basic EPS', 'Book Value Per Share', 'EBITDA', 'Total Debt', 'Cash And Cash Equivalents', 'Ordinary Shares Number']
        fund_data = {}
        
        for m in metrics:
            val = self._get_latest_fundamental(sid, date, m)
            if val is not None:
                fund_data[m] = val
                
        factors = {}
        
        # P/E
        if fund_data.get('Basic EPS'):
            factors['pe_ratio'] = price / fund_data['Basic EPS']
            
        # P/B
        if fund_data.get('Book Value Per Share'):
            factors['pb_ratio'] = price / fund_data['Book Value Per Share']
            
        return factors

    def _get_price(self, sid: int, date):
        # Find closest price on or before date
        record = self.db.query(MarketDataDaily)\
            .filter(MarketDataDaily.sid == sid, MarketDataDaily.date <= date)\
            .order_by(MarketDataDaily.date.desc())\
            .first()
        return record.adj_close if record else None

    def _get_latest_fundamental(self, sid: int, date, metric: str):
        record = self.db.query(Fundamentals)\
            .filter(Fundamentals.sid == sid, Fundamentals.metric == metric, Fundamentals.date <= date)\
            .order_by(Fundamentals.date.desc())\
            .first()
        return record.value if record else None
