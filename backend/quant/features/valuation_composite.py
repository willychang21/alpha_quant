import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ValuationComposite:
    """
    Valuation Composite Factor (Tier-3 / Weekly System).
    
    Logic:
    Combine multiple valuation metrics for robustness.
    
    Components:
    1. DCF Upside: (Intrinsic Value - Price) / Price
    2. Earnings Yield (EY): EPS / Price
    3. Yield Spread: EY - 10Y Treasury Yield
    
    Formula:
    Score = 0.5 * DCF_Upside + 0.5 * Earnings_Yield
    (Spread is implicitly covered by EY in cross-section, but we calculate it for metadata)
    """
    
    def __init__(self):
        pass
        
    def compute(self, ticker_data: Dict[str, Any], dcf_upside: float, risk_free_rate: float = 0.04) -> Dict[str, float]:
        """
        Compute the valuation composite score.
        
        Args:
            ticker_data: Dictionary containing 'info' (YFinance info dict).
            dcf_upside: The upside calculated by the Valuation Engine (e.g. 0.20 for 20%).
            risk_free_rate: Current 10Y Treasury Yield (e.g. 0.04).
        
        Returns:
            Dict containing 'score', 'earnings_yield', 'yield_spread'.
        """
        try:
            info = ticker_data.get('info', {})
            if not info:
                return {'score': 0.0, 'earnings_yield': 0.0, 'yield_spread': 0.0}
                
            price = info.get('currentPrice') or info.get('previousClose')
            eps = info.get('trailingEps')
            
            earnings_yield = 0.0
            if price and eps:
                earnings_yield = eps / price
                
            yield_spread = earnings_yield - risk_free_rate
            
            # Composite Score
            # We treat DCF Upside and Earnings Yield as two views on value.
            # Both are roughly in the same "return" unit space.
            # We average them.
            
            # Handle cases where DCF might be missing (0.0)
            if dcf_upside == 0.0 and earnings_yield != 0.0:
                score = earnings_yield
            elif dcf_upside != 0.0 and earnings_yield == 0.0:
                score = dcf_upside
            else:
                score = 0.5 * dcf_upside + 0.5 * earnings_yield
            
            return {
                'score': float(score),
                'earnings_yield': float(earnings_yield),
                'yield_spread': float(yield_spread)
            }
            
        except Exception as e:
            logger.debug(f"ValuationComposite calculation failed: {e}")
            return {'score': 0.0, 'earnings_yield': 0.0, 'yield_spread': 0.0}
