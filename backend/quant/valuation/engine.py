"""Valuation Engine Module.

Implements DCF (Discounted Cash Flow) valuation with configurable assumptions.
"""

from sqlalchemy.orm import Session
from quant.data.models import Security, MarketDataDaily, Fundamentals
from datetime import date
from typing import Optional, Dict, Any
import pandas as pd

# Import infrastructure
from core.structured_logger import get_structured_logger
from config.quant_config import get_valuation_config, ValuationConfig

logger = get_structured_logger("ValuationEngine")


class ValuationEngine:
    """DCF Valuation Engine with configurable assumptions.
    
    Uses ValuationConfig for WACC, growth rates, and projection parameters.
    All assumptions can be overridden via environment variables.
    """
    
    def __init__(self, db: Session, config: Optional[ValuationConfig] = None):
        """Initialize ValuationEngine.
        
        Args:
            db: SQLAlchemy database session
            config: Optional ValuationConfig, uses singleton if not provided
        """
        self.db = db
        self.config = config or get_valuation_config()

    def calculate_dcf(self, sid: int, valuation_date: date) -> Optional[Dict[str, Any]]:
        """Calculate DCF Valuation for a security.
        
        Args:
            sid: Security ID
            valuation_date: Date for valuation
            
        Returns:
            Dict with fair_value, upside, wacc, growth_rate, price
            or None if insufficient data
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
            
        # 2. Get assumptions from config (supports env variable overrides)
        wacc = self.config.wacc
        growth_rate = self.config.growth_rate
        terminal_growth = self.config.terminal_growth
        projection_years = self.config.projection_years
        
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

    def _get_price(self, sid: int, target_date: date) -> Optional[float]:
        """Get the most recent price for a security.
        
        Args:
            sid: Security ID
            target_date: Target date for price lookup
            
        Returns:
            Adjusted close price or None if not found
        """
        record = self.db.query(MarketDataDaily)\
            .filter(MarketDataDaily.sid == sid, MarketDataDaily.date <= target_date)\
            .order_by(MarketDataDaily.date.desc())\
            .first()
        return record.adj_close if record else None

    def _get_fundamental(self, sid: int, target_date: date, metric: str) -> Optional[float]:
        """Get the most recent fundamental metric for a security.
        
        Args:
            sid: Security ID
            target_date: Target date for lookup
            metric: Metric name (e.g., 'Total Revenue', 'EBITDA')
            
        Returns:
            Metric value or None if not found
        """
        record = self.db.query(Fundamentals)\
            .filter(Fundamentals.sid == sid, Fundamentals.metric == metric, Fundamentals.date <= target_date)\
            .order_by(Fundamentals.date.desc())\
            .first()
        return record.value if record else None
