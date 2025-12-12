"""Levy Relative Strength Calculator.

Calculates Levy Relative Strength (RSL) for momentum screening.
RSL = Close / SMA(Close, 26 weeks)

RSL > 1.0 indicates price above average (momentum leader)
RSL < 1.0 indicates price below average (momentum laggard)

References:
- Robert A. Levy (1967): Relative Strength as a Criterion for Investment Selection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

from quant.rotation.models import LevyRSResult

logger = logging.getLogger(__name__)


class LevyRSCalculator:
    """Calculates Levy Relative Strength (RSL) for momentum screening.
    
    Levy RS measures price momentum by comparing current price to its
    moving average. Stocks with high RSL values are momentum leaders.
    
    Attributes:
        period: Lookback period for SMA calculation (default: 130 days = 26 weeks)
    """
    
    def __init__(self, period: int = 130):
        """Initialize Levy RS Calculator.
        
        Args:
            period: Lookback period for SMA calculation (default: 130 days = 26 weeks)
        """
        self.period = period
    
    def calculate_rsl(self, close: pd.Series) -> pd.Series:
        """Calculate Levy Relative Strength.
        
        RSL = Close / SMA(Close, period)
        
        Args:
            close: Series of closing prices with DatetimeIndex
            
        Returns:
            Series of RSL values (typically 0.8 - 1.2)
            
        Note:
            Returns RSL of 1.0 for periods with insufficient history.
        """
        if len(close) < self.period:
            logger.warning(
                f"Insufficient price history ({len(close)} < {self.period}). "
                "Returning RSL = 1.0"
            )
            return pd.Series(1.0, index=close.index)
        
        # Calculate 26-week (130-day) Simple Moving Average
        sma = close.rolling(window=self.period).mean()
        
        # Calculate RSL = Close / SMA
        # Handle division by zero by replacing 0 with NaN, then filling with 1.0
        rsl = close / sma.replace(0, np.nan)
        rsl = rsl.fillna(1.0)
        
        return rsl
    
    def get_sma(self, close: pd.Series) -> pd.Series:
        """Get the 26-week SMA used in RSL calculation.
        
        Args:
            close: Series of closing prices
            
        Returns:
            Series of SMA values
        """
        return close.rolling(window=self.period).mean()
    
    def get_momentum_signal(self, rsl: float) -> str:
        """Get momentum signal based on RSL value.
        
        Args:
            rsl: Levy Relative Strength value
            
        Returns:
            Signal classification:
            - 'strong': RSL > 1.05 (strong momentum)
            - 'positive': RSL > 1.0 (positive momentum)
            - 'breakdown': RSL < 1.0 (momentum breakdown)
            - 'weak': RSL < 0.95 (weak momentum)
        """
        if rsl > 1.05:
            return 'strong'
        elif rsl > 1.0:
            return 'positive'
        elif rsl >= 0.95:
            return 'breakdown'
        else:
            return 'weak'
    
    def rank_by_rsl(self, rsl_dict: Dict[str, float]) -> List[Tuple[str, float, int]]:
        """Rank stocks by RSL in descending order.
        
        Args:
            rsl_dict: Dictionary of ticker -> RSL value
            
        Returns:
            List of (ticker, rsl, rank) tuples sorted by RSL descending.
            Rank is 1-indexed (rank 1 = highest RSL).
        """
        if not rsl_dict:
            return []
        
        # Sort by RSL descending
        sorted_items = sorted(rsl_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Add rank (1-indexed)
        ranked = [(ticker, rsl, rank + 1) for rank, (ticker, rsl) in enumerate(sorted_items)]
        
        return ranked
    
    def detect_breakdown(self, rsl: pd.Series) -> pd.Series:
        """Detect when RSL crosses below 1.0 from above.
        
        This is a momentum breakdown signal indicating the stock
        is losing relative strength.
        
        Args:
            rsl: Series of RSL values
            
        Returns:
            Boolean Series where True indicates a breakdown signal
        """
        # Shift to get previous value
        rsl_prev = rsl.shift(1)
        
        # Breakdown: previous RSL >= 1.0 and current RSL < 1.0
        breakdown = (rsl_prev >= 1.0) & (rsl < 1.0)
        
        return breakdown
    
    def compute_result(
        self,
        ticker: str,
        close: pd.Series,
        universe_rsl: Optional[Dict[str, float]] = None
    ) -> LevyRSResult:
        """Compute complete Levy RS result for a stock.
        
        Args:
            ticker: Stock ticker symbol
            close: Series of closing prices
            universe_rsl: Optional dict of all stocks' RSL for percentile calculation
            
        Returns:
            LevyRSResult with all computed values
        """
        rsl_series = self.calculate_rsl(close)
        sma_series = self.get_sma(close)
        
        # Get latest values
        current_rsl = float(rsl_series.iloc[-1])
        current_sma = float(sma_series.iloc[-1]) if not pd.isna(sma_series.iloc[-1]) else 0.0
        
        # Calculate percentile rank if universe provided
        percentile_rank = None
        if universe_rsl and ticker in universe_rsl:
            all_rsl = list(universe_rsl.values())
            rank = sum(1 for r in all_rsl if r <= current_rsl)
            percentile_rank = (rank / len(all_rsl)) * 100
        
        return LevyRSResult(
            ticker=ticker,
            rsl=current_rsl,
            sma_26w=current_sma,
            signal=self.get_momentum_signal(current_rsl),
            percentile_rank=percentile_rank,
            timestamp=datetime.now()
        )
