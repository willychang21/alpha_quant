"""
Post-Earnings Announcement Drift (PEAD) Factor

Based on Ball & Brown (1968), Bernard & Thomas (1989):
Stocks with positive earnings surprises tend to drift higher for 60-90 days.
Stocks with negative surprises tend to drift lower.

Key Metrics:
- SUE (Standardized Unexpected Earnings) = (Actual - Expected) / Std(Historical Surprises)
- Earnings Surprise % = (Actual - Expected) / |Expected|
- Recency Weight: More recent announcements have stronger signal

Signal: LONG positive surprises, SHORT negative surprises
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from quant.features.base import FeatureGenerator
import yfinance as yf
import logging

logger = logging.getLogger(__name__)


class PostEarningsAnnouncementDrift(FeatureGenerator):
    """
    Calculates PEAD score based on earnings surprise and recency.
    
    Higher score = more positive surprise = expected outperformance
    Lower score = more negative surprise = expected underperformance
    """
    
    def __init__(self, lookback_days: int = 60, decay_halflife: int = 30):
        """
        Args:
            lookback_days: How far back to look for earnings announcements
            decay_halflife: Days for signal decay (recency weighting)
        """
        self.lookback_days = lookback_days
        self.decay_halflife = decay_halflife
    
    @property
    def name(self) -> str:
        return "PostEarningsAnnouncementDrift"
    
    @property
    def description(self) -> str:
        return "Earnings surprise with recency weighting. Positive surprises expected to outperform."
    
    def compute(
        self, 
        history: pd.DataFrame, 
        ticker_info: Optional[dict] = None,
        ticker: Optional[str] = None
    ) -> pd.Series:
        """
        Compute PEAD score from earnings data.
        
        Args:
            history: OHLCV DataFrame (used for date reference)
            ticker_info: Dict with yfinance info (optional)
            ticker: Ticker symbol to fetch earnings data
            
        Returns:
            pd.Series with PEAD score
        """
        if ticker is None:
            return pd.Series([0.0], index=[pd.Timestamp.now()])
        
        try:
            # Fetch earnings data
            stock = yf.Ticker(ticker)
            
            # Get earnings history (actual vs estimate)
            earnings_history = stock.earnings_history
            
            if earnings_history is None or earnings_history.empty:
                logger.debug(f"{ticker}: No earnings history available")
                return pd.Series([0.0], index=[pd.Timestamp.now()])
            
            # Get most recent earnings
            earnings_history = earnings_history.reset_index()
            
            # Find columns for actual and estimate
            actual_col = None
            estimate_col = None
            date_col = None
            
            for col in earnings_history.columns:
                col_lower = str(col).lower()
                if 'actual' in col_lower or 'epsactual' in col_lower:
                    actual_col = col
                elif 'estimate' in col_lower or 'epsestimate' in col_lower:
                    estimate_col = col
                elif 'date' in col_lower or 'earnings' in col_lower:
                    date_col = col
            
            # Alternative column names
            if actual_col is None and 'epsActual' in earnings_history.columns:
                actual_col = 'epsActual'
            if estimate_col is None and 'epsEstimate' in earnings_history.columns:
                estimate_col = 'epsEstimate'
            if date_col is None and 'Earnings Date' in earnings_history.columns:
                date_col = 'Earnings Date'
            
            if actual_col is None or estimate_col is None:
                logger.debug(f"{ticker}: Missing actual/estimate columns. Columns: {earnings_history.columns.tolist()}")
                return pd.Series([0.0], index=[pd.Timestamp.now()])
            
            # Calculate surprise for each earnings
            earnings_history['surprise'] = (
                earnings_history[actual_col] - earnings_history[estimate_col]
            )
            earnings_history['surprise_pct'] = (
                earnings_history['surprise'] / np.abs(earnings_history[estimate_col].replace(0, np.nan))
            ).fillna(0)
            
            # Cap extreme surprises
            earnings_history['surprise_pct'] = earnings_history['surprise_pct'].clip(-2, 2)
            
            # Get most recent announcement
            if date_col is not None:
                try:
                    earnings_history[date_col] = pd.to_datetime(earnings_history[date_col])
                    earnings_history = earnings_history.sort_values(date_col, ascending=False)
                    latest = earnings_history.iloc[0]
                    days_since = (datetime.now() - latest[date_col].to_pydatetime().replace(tzinfo=None)).days
                except:
                    latest = earnings_history.iloc[0]
                    days_since = 30  # Default
            else:
                latest = earnings_history.iloc[0]
                days_since = 30
            
            # Only consider if within lookback window
            if days_since > self.lookback_days:
                logger.debug(f"{ticker}: Last earnings too old ({days_since} days)")
                return pd.Series([0.0], index=[pd.Timestamp.now()])
            
            # Calculate recency-weighted score
            # Decay factor: more recent = stronger signal
            decay = np.exp(-np.log(2) * days_since / self.decay_halflife)
            
            surprise_pct = latest['surprise_pct']
            
            # SUE-like standardization using historical surprises
            historical_surprises = earnings_history['surprise_pct'].dropna()
            if len(historical_surprises) > 2:
                std_surprise = historical_surprises.std()
                if std_surprise > 0:
                    sue = surprise_pct / std_surprise
                else:
                    sue = surprise_pct
            else:
                sue = surprise_pct
            
            # Final PEAD score = SUE * decay factor
            pead_score = sue * decay
            
            logger.debug(
                f"{ticker}: Surprise={surprise_pct:.2%}, SUE={sue:.2f}, "
                f"Days Since={days_since}, Decay={decay:.2f}, Score={pead_score:.2f}"
            )
            
            return pd.Series([pead_score], index=[pd.Timestamp.now()])
            
        except Exception as e:
            logger.warning(f"PEAD calculation failed for {ticker}: {e}")
            return pd.Series([0.0], index=[pd.Timestamp.now()])
    
    def get_earnings_calendar(self, ticker: str) -> Optional[Dict]:
        """
        Get upcoming earnings date for a ticker.
        
        Returns:
            Dict with 'earnings_date', 'days_until' if available
        """
        try:
            stock = yf.Ticker(ticker)
            calendar = stock.calendar
            
            if calendar is None:
                return None
            
            if isinstance(calendar, dict):
                earnings_date = calendar.get('Earnings Date', [None])[0]
            else:
                earnings_date = None
            
            if earnings_date:
                days_until = (earnings_date - datetime.now()).days
                return {
                    'earnings_date': earnings_date,
                    'days_until': days_until
                }
            
            return None
            
        except:
            return None


class EarningsMomentum(FeatureGenerator):
    """
    Alternative PEAD implementation using earnings trend.
    
    Looks at direction of earnings revisions and YoY growth.
    """
    
    def __init__(self):
        pass
    
    @property
    def name(self) -> str:
        return "EarningsMomentum"
    
    @property
    def description(self) -> str:
        return "Earnings revision and growth trend. Positive revisions expected to outperform."
    
    def compute(
        self, 
        history: pd.DataFrame, 
        ticker_info: Optional[dict] = None
    ) -> pd.Series:
        """
        Compute earnings momentum from analyst revisions.
        
        Uses yfinance info fields:
        - earningsQuarterlyGrowth
        - revenueGrowth
        - recommendationMean changes
        """
        if ticker_info is None:
            return pd.Series([0.0], index=[pd.Timestamp.now()])
        
        try:
            score = 0.0
            
            # 1. Quarterly Earnings Growth
            eps_growth = ticker_info.get('earningsQuarterlyGrowth', 0) or 0
            # Cap at reasonable bounds
            eps_growth = max(-1, min(1, eps_growth))
            score += eps_growth * 0.5
            
            # 2. Revenue Growth
            rev_growth = ticker_info.get('revenueGrowth', 0) or 0
            rev_growth = max(-1, min(1, rev_growth))
            score += rev_growth * 0.3
            
            # 3. Forward EPS > Trailing EPS (growth expected)
            forward_eps = ticker_info.get('forwardEps', 0) or 0
            trailing_eps = ticker_info.get('trailingEps', 0) or 0
            
            if trailing_eps and trailing_eps > 0:
                eps_trajectory = (forward_eps - trailing_eps) / abs(trailing_eps)
                eps_trajectory = max(-1, min(1, eps_trajectory))
                score += eps_trajectory * 0.2
            
            logger.debug(
                f"EarningsMomentum: EPS Growth={eps_growth:.2f}, Rev Growth={rev_growth:.2f}, Score={score:.2f}"
            )
            
            return pd.Series([score], index=[pd.Timestamp.now()])
            
        except Exception as e:
            logger.warning(f"EarningsMomentum calculation failed: {e}")
            return pd.Series([0.0], index=[pd.Timestamp.now()])
