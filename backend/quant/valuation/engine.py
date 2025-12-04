from sqlalchemy.orm import Session
from quant.data.models import Security, MarketDataDaily, Fundamentals
from datetime import date
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ValuationEngine:
    def __init__(self, db: Session):
        self.db = db

    def calculate_dcf(self, sid: int, valuation_date: date):
        """
        Calculate DCF Valuation for a security.
        """
        # 1. Fetch Inputs
        price = self._get_price(sid, valuation_date)
        if not price:
            return None
            
        # Fundamentals
        revenue = self._get_fundamental(sid, valuation_date, 'Total Revenue')
        ebitda = self._get_fundamental(sid, valuation_date, 'EBITDA') or self._get_fundamental(sid, valuation_date, 'Normalized EBITDA')
        net_income = self._get_fundamental(sid, valuation_date, 'Net Income')
        fcf = self._get_fundamental(sid, valuation_date, 'Free Cash Flow')
        
        total_debt = self._get_fundamental(sid, valuation_date, 'Total Debt') or 0
        cash = self._get_fundamental(sid, valuation_date, 'Cash And Cash Equivalents') or 0
        net_debt = total_debt - cash
        
        shares = self._get_fundamental(sid, valuation_date, 'Ordinary Shares Number')
        if not shares and price > 0:
            # Fallback: Market Cap / Price if Market Cap exists
            market_cap = self._get_fundamental(sid, valuation_date, 'Market Cap')
            if market_cap:
                shares = market_cap / price
                
        if not shares or not fcf or not ebitda:
            logger.warning(f"Missing key inputs for DCF (SID: {sid})")
            return None
            
        # 2. Assumptions (Simplified for MVP)
        # In production, these would come from a 'Assumptions' table or model
        wacc = 0.09 # 9% WACC
        growth_rate = 0.05 # 5% Growth
        terminal_growth = 0.02 # 2% Terminal
        projection_years = 5
        
        # 3. Projection
        projected_fcf = []
        current_fcf = fcf
        
        pv_fcf_sum = 0
        
        for i in range(1, projection_years + 1):
            current_fcf *= (1 + growth_rate)
            projected_fcf.append(current_fcf)
            pv_fcf_sum += current_fcf / ((1 + wacc) ** i)
            
        # 4. Terminal Value (Gordon Growth)
        last_fcf = projected_fcf[-1]
        tv = last_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
        pv_tv = tv / ((1 + wacc) ** projection_years)
        
        # 5. Equity Value
        enterprise_value = pv_fcf_sum + pv_tv
        equity_value = enterprise_value - net_debt
        fair_value = equity_value / shares
        
        return {
            'fair_value': fair_value,
            'upside': (fair_value - price) / price,
            'wacc': wacc,
            'growth_rate': growth_rate,
            'price': price
        }

    def _get_price(self, sid: int, date):
        record = self.db.query(MarketDataDaily)\
            .filter(MarketDataDaily.sid == sid, MarketDataDaily.date <= date)\
            .order_by(MarketDataDaily.date.desc())\
            .first()
        return record.adj_close if record else None

    def _get_fundamental(self, sid: int, date, metric: str):
        record = self.db.query(Fundamentals)\
            .filter(Fundamentals.sid == sid, Fundamentals.metric == metric, Fundamentals.date <= date)\
            .order_by(Fundamentals.date.desc())\
            .first()
        return record.value if record else None
