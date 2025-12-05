import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AnalystRevisions:
    """
    Analyst Revisions Factor (Tier-3 / Weekly System).
    
    Logic:
    Analysts tend to revise estimates slowly (anchoring). 
    Upward revisions predict future outperformance.
    
    Formula:
    Rev = (EPS_est_t - EPS_est_t-1m) / |EPS_est_t-1m|
    
    We use '0y' (Current Year) estimates.
    """
    
    def __init__(self):
        pass
        
    def compute(self, ticker_data: Dict[str, Any]) -> float:
        """
        Compute the revision score.
        
        Args:
            ticker_data: Dictionary containing 'info', 'financials', and 'estimates' (if available).
                         We expect 'estimates' to be the result of yf.Ticker.eps_trend
        
        Returns:
            float: Revision score (percentage change).
        """
        try:
            # We need to fetch this specifically if not present
            # For now, assume the caller (RankingEngine) puts it in ticker_data['estimates']
            # or we fetch it here? 
            # Ideally RankingEngine fetches it via YFinanceProvider.get_estimates()
            
            estimates = ticker_data.get('estimates')
            
            if estimates is None or estimates.empty:
                return 0.0
                
            # estimates is the eps_trend DataFrame
            # Index: 0q, +1q, 0y, +1y
            # Columns: current, 7daysAgo, 30daysAgo, etc.
            
            if '0y' not in estimates.index:
                return 0.0
                
            row = estimates.loc['0y']
            
            current = row.get('current')
            prev_30d = row.get('30daysAgo')
            
            if pd.isna(current) or pd.isna(prev_30d) or prev_30d == 0:
                return 0.0
                
            revision = (current - prev_30d) / abs(prev_30d)
            
            return float(revision)
            
        except Exception as e:
            logger.debug(f"Revisions calculation failed: {e}")
            return 0.0
