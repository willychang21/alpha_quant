"""Mansfield Relative Strength Calculator.

Calculates Mansfield Relative Strength (MRS) for breakout detection.
MRS = ((RS / SMA(RS, 52 weeks)) - 1) × 100
where RS = Stock Price / Benchmark Price

MRS centered around 0:
- MRS > 0: Outperforming benchmark
- MRS < 0: Underperforming benchmark
- Zero crossover from below: Breakout signal

References:
- Stan Weinstein (1988): Secrets For Profiting in Bull and Bear Markets
- Mansfield Charts relative strength methodology
"""

import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime
import logging

from quant.rotation.models import MansfieldRSResult

logger = logging.getLogger(__name__)


class MansfieldRSCalculator:
    """Calculates Mansfield Relative Strength (MRS) for breakout detection.
    
    MRS identifies stocks breaking out relative to the market by comparing
    their relative strength to its own moving average.
    
    Attributes:
        benchmark: Benchmark index symbol (default: 'SPY')
        period: Lookback period for RS SMA (default: 252 days = 52 weeks)
        slope_period: Period for calculating MRS slope (default: 5 days)
    """
    
    def __init__(
        self,
        benchmark: str = 'SPY',
        period: int = 252,
        slope_period: int = 5
    ):
        """Initialize Mansfield RS Calculator.
        
        Args:
            benchmark: Benchmark index symbol (default: 'SPY')
            period: Lookback period for SMA calculation (default: 252 days = 52 weeks)
            slope_period: Period for calculating MRS slope (default: 5 days)
        """
        self.benchmark = benchmark
        self.period = period
        self.slope_period = slope_period
    
    def calculate_raw_rs(
        self,
        stock_price: pd.Series,
        benchmark_price: pd.Series
    ) -> pd.Series:
        """Calculate raw relative strength.
        
        RS = Stock Price / Benchmark Price
        
        Args:
            stock_price: Series of stock closing prices
            benchmark_price: Series of benchmark closing prices
            
        Returns:
            Series of raw RS values
        """
        # Align indices
        stock_aligned, benchmark_aligned = stock_price.align(benchmark_price, join='inner')
        
        # Handle division by zero
        rs = stock_aligned / benchmark_aligned.replace(0, np.nan)
        rs = rs.ffill().bfill()
        
        return rs
    
    def calculate_mrs(
        self,
        stock_price: pd.Series,
        benchmark_price: pd.Series
    ) -> pd.Series:
        """Calculate Mansfield Relative Strength.
        
        MRS = ((RS / SMA(RS, period)) - 1) × 100
        
        Args:
            stock_price: Series of stock closing prices
            benchmark_price: Series of benchmark closing prices
            
        Returns:
            Series of MRS values centered around 0
        """
        # Check for sufficient data
        if len(stock_price) < self.period or len(benchmark_price) < self.period:
            logger.warning(
                f"Insufficient price history for MRS calculation. "
                f"Need {self.period}, got stock:{len(stock_price)}, benchmark:{len(benchmark_price)}. "
                "Returning MRS = 0.0"
            )
            common_idx = stock_price.index.intersection(benchmark_price.index)
            return pd.Series(0.0, index=common_idx)
        
        # Calculate raw RS
        rs = self.calculate_raw_rs(stock_price, benchmark_price)
        
        if rs.empty:
            logger.warning("Empty RS series. Returning MRS = 0.0")
            return pd.Series(0.0, index=stock_price.index)
        
        # Calculate SMA of RS
        rs_sma = rs.rolling(window=self.period).mean()
        
        # Calculate MRS = ((RS / SMA(RS)) - 1) × 100
        mrs = ((rs / rs_sma.replace(0, np.nan)) - 1) * 100
        mrs = mrs.fillna(0.0)
        
        return mrs
    
    def calculate_mrs_slope(self, mrs: pd.Series) -> pd.Series:
        """Calculate the slope (rate of change) of MRS.
        
        Args:
            mrs: Series of MRS values
            
        Returns:
            Series of MRS slope values
        """
        # Simple difference over slope_period
        slope = mrs.diff(periods=self.slope_period) / self.slope_period
        return slope.fillna(0.0)
    
    def detect_zero_crossover(self, mrs: pd.Series) -> pd.Series:
        """Detect when MRS crosses above zero from below.
        
        This is a relative breakout signal indicating the stock
        is starting to outperform the benchmark.
        
        Args:
            mrs: Series of MRS values
            
        Returns:
            Boolean Series where True indicates a bullish crossover
        """
        mrs_prev = mrs.shift(1)
        
        # Bullish crossover: previous < 0 and current >= 0
        crossover = (mrs_prev < 0) & (mrs >= 0)
        
        return crossover
    
    def get_improvement_signal(self, mrs: float, mrs_slope: float) -> str:
        """Get improvement signal based on MRS and its slope.
        
        Args:
            mrs: Current MRS value
            mrs_slope: Current MRS slope (rate of change)
            
        Returns:
            Signal classification:
            - 'breakout': MRS > 0 and slope > 0 (outperforming and improving)
            - 'improving': MRS < 0 and slope > 0 (underperforming but improving)
            - 'weakening': MRS > 0 and slope < 0 (outperforming but weakening)
            - 'lagging': MRS < 0 and slope < 0 (underperforming and worsening)
        """
        if mrs >= 0 and mrs_slope >= 0:
            return 'breakout'
        elif mrs < 0 and mrs_slope > 0:
            return 'improving'
        elif mrs >= 0 and mrs_slope < 0:
            return 'weakening'
        else:  # mrs < 0 and mrs_slope <= 0
            return 'lagging'
    
    def compute_result(
        self,
        ticker: str,
        stock_price: pd.Series,
        benchmark_price: pd.Series
    ) -> MansfieldRSResult:
        """Compute complete Mansfield RS result for a stock.
        
        Args:
            ticker: Stock ticker symbol
            stock_price: Series of stock closing prices
            benchmark_price: Series of benchmark closing prices
            
        Returns:
            MansfieldRSResult with all computed values
        """
        # Calculate MRS
        mrs_series = self.calculate_mrs(stock_price, benchmark_price)
        
        if mrs_series.empty:
            return MansfieldRSResult(
                ticker=ticker,
                mrs=0.0,
                mrs_slope=0.0,
                raw_rs=1.0,
                signal='lagging',
                zero_crossover=False,
                timestamp=datetime.now()
            )
        
        # Calculate raw RS
        rs_series = self.calculate_raw_rs(stock_price, benchmark_price)
        
        # Calculate slope
        slope_series = self.calculate_mrs_slope(mrs_series)
        
        # Detect zero crossover
        crossover_series = self.detect_zero_crossover(mrs_series)
        
        # Get latest values
        current_mrs = float(mrs_series.iloc[-1])
        current_slope = float(slope_series.iloc[-1])
        current_rs = float(rs_series.iloc[-1]) if not rs_series.empty else 1.0
        current_crossover = bool(crossover_series.iloc[-1]) if not crossover_series.empty else False
        
        return MansfieldRSResult(
            ticker=ticker,
            mrs=current_mrs,
            mrs_slope=current_slope,
            raw_rs=current_rs,
            signal=self.get_improvement_signal(current_mrs, current_slope),
            zero_crossover=current_crossover,
            timestamp=datetime.now()
        )
