"""Volume Structure Analyzer.

Analyzes volume structure to detect institutional activity through:
- Volume Z-score: Statistical anomaly detection
- Absorption patterns: High volume with minimal price change
- Smart money rejection: New low with close near high on high volume

Volume is often the first indicator of institutional accumulation/distribution
before price confirms the trend.
"""

import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime
import logging

from quant.rotation.models import VolumeAnalysisResult

logger = logging.getLogger(__name__)


class VolumeStructureAnalyzer:
    """Analyzes volume structure to detect institutional activity.
    
    Uses statistical methods (Z-score) to identify abnormal volume
    and specific patterns that indicate smart money behavior.
    
    Attributes:
        zscore_period: Rolling window for Z-score calculation (default: 20)
        absorption_threshold: Z-score threshold for absorption detection (default: 2.0)
        price_threshold: Maximum price change for absorption (default: 0.5%)
    """
    
    def __init__(
        self,
        zscore_period: int = 20,
        absorption_threshold: float = 2.0,
        price_threshold: float = 0.005
    ):
        """Initialize Volume Structure Analyzer.
        
        Args:
            zscore_period: Rolling window for Z-score calculation (default: 20 days)
            absorption_threshold: Z-score threshold for absorption detection (default: 2.0)
            price_threshold: Maximum price change for absorption (default: 0.5%)
        """
        self.zscore_period = zscore_period
        self.absorption_threshold = absorption_threshold
        self.price_threshold = price_threshold
    
    def calculate_volume_zscore(self, volume: pd.Series) -> pd.Series:
        """Calculate volume Z-score using rolling statistics.
        
        Z = (V - mean(V, period)) / std(V, period)
        
        Args:
            volume: Series of volume data
            
        Returns:
            Series of volume Z-scores
            
        Note:
            Excludes zero and missing values from calculation.
            Returns 0.0 for periods with zero standard deviation.
        """
        # Handle zero/missing values
        volume_clean = volume.replace(0, np.nan)
        
        # Calculate rolling statistics
        rolling_mean = volume_clean.rolling(window=self.zscore_period, min_periods=5).mean()
        rolling_std = volume_clean.rolling(window=self.zscore_period, min_periods=5).std()
        
        # Calculate Z-score, handling zero std
        zscore = (volume - rolling_mean) / rolling_std.replace(0, np.nan)
        zscore = zscore.fillna(0.0)
        
        return zscore
    
    def classify_volume(self, zscore: float) -> str:
        """Classify volume based on Z-score.
        
        Args:
            zscore: Volume Z-score value
            
        Returns:
            Classification:
            - 'normal': |Z| < 1.5
            - 'elevated': 1.5 <= |Z| < 2.5
            - 'extreme': |Z| >= 2.5
        """
        abs_z = abs(zscore)
        
        if abs_z < 1.5:
            return 'normal'
        elif abs_z < 2.5:
            return 'elevated'
        else:
            return 'extreme'
    
    def detect_absorption(
        self,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """Detect absorption pattern: high volume with minimal price change.
        
        Absorption indicates institutional accumulation when there's
        significant trading activity but price doesn't move much,
        suggesting buying is absorbing selling pressure (or vice versa).
        
        Conditions:
        - Volume Z-score > absorption_threshold
        - |Price change| < price_threshold
        
        Args:
            close: Series of closing prices
            volume: Series of volume data
            
        Returns:
            Boolean Series where True indicates absorption pattern
        """
        # Calculate price change
        price_change = close.pct_change().abs()
        
        # Calculate volume Z-score
        vol_zscore = self.calculate_volume_zscore(volume)
        
        # Absorption: high volume + low price change
        absorption = (vol_zscore > self.absorption_threshold) & (price_change < self.price_threshold)
        
        return absorption
    
    def detect_smart_money_rejection(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        lookback: int = 20
    ) -> pd.Series:
        """Detect smart money rejection pattern.
        
        Smart money rejection occurs when price makes a new low,
        but closes near the high on elevated volume - indicating
        institutional buying at new lows.
        
        Conditions:
        - Price makes new `lookback`-day low
        - Close is in upper 25% of day's range
        - Volume Z-score > absorption_threshold
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            volume: Series of volume data
            lookback: Period for new low detection (default: 20)
            
        Returns:
            Boolean Series where True indicates rejection pattern
        """
        # Calculate new low
        rolling_low = low.rolling(window=lookback).min()
        is_new_low = low <= rolling_low
        
        # Calculate where close is in day's range
        day_range = high - low
        close_position = (close - low) / day_range.replace(0, np.nan)
        close_position = close_position.fillna(0.5)
        
        # Close in upper 25% of range
        close_near_high = close_position >= 0.75
        
        # Calculate volume Z-score
        vol_zscore = self.calculate_volume_zscore(volume)
        high_volume = vol_zscore > self.absorption_threshold
        
        # Rejection: new low + close near high + high volume
        rejection = is_new_low & close_near_high & high_volume
        
        return rejection
    
    def detect_distribution(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        lookback: int = 20
    ) -> pd.Series:
        """Detect distribution pattern (opposite of rejection).
        
        Distribution occurs when price makes a new high, but closes
        near the low on elevated volume - indicating institutional selling.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            volume: Series of volume data
            lookback: Period for new high detection
            
        Returns:
            Boolean Series where True indicates distribution pattern
        """
        # Calculate new high
        rolling_high = high.rolling(window=lookback).max()
        is_new_high = high >= rolling_high
        
        # Calculate where close is in day's range
        day_range = high - low
        close_position = (close - low) / day_range.replace(0, np.nan)
        close_position = close_position.fillna(0.5)
        
        # Close in lower 25% of range
        close_near_low = close_position <= 0.25
        
        # Calculate volume Z-score
        vol_zscore = self.calculate_volume_zscore(volume)
        high_volume = vol_zscore > self.absorption_threshold
        
        # Distribution: new high + close near low + high volume
        distribution = is_new_high & close_near_low & high_volume
        
        return distribution
    
    def get_pattern(
        self,
        absorption: bool,
        rejection: bool,
        distribution: bool = False
    ) -> str:
        """Determine the overall volume pattern.
        
        Args:
            absorption: True if absorption detected
            rejection: True if smart money rejection detected
            distribution: True if distribution detected
            
        Returns:
            Pattern classification:
            - 'rejection': Smart money rejection (bullish)
            - 'absorption': Accumulation pattern (bullish)
            - 'distribution': Institutional selling (bearish)
            - 'neutral': No significant pattern
        """
        if rejection:
            return 'rejection'
        elif absorption:
            return 'absorption'
        elif distribution:
            return 'distribution'
        else:
            return 'neutral'
    
    def compute_result(
        self,
        ticker: str,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> VolumeAnalysisResult:
        """Compute complete volume analysis result for a stock.
        
        Args:
            ticker: Stock ticker symbol
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            volume: Series of volume data
            
        Returns:
            VolumeAnalysisResult with all computed values
        """
        # Calculate Z-score
        zscore_series = self.calculate_volume_zscore(volume)
        current_zscore = float(zscore_series.iloc[-1]) if not zscore_series.empty else 0.0
        
        # Detect patterns
        absorption_series = self.detect_absorption(close, volume)
        rejection_series = self.detect_smart_money_rejection(high, low, close, volume)
        distribution_series = self.detect_distribution(high, low, close, volume)
        
        current_absorption = bool(absorption_series.iloc[-1]) if not absorption_series.empty else False
        current_rejection = bool(rejection_series.iloc[-1]) if not rejection_series.empty else False
        current_distribution = bool(distribution_series.iloc[-1]) if not distribution_series.empty else False
        
        return VolumeAnalysisResult(
            ticker=ticker,
            volume_zscore=current_zscore,
            volume_classification=self.classify_volume(current_zscore),
            absorption_detected=current_absorption,
            rejection_detected=current_rejection,
            pattern=self.get_pattern(current_absorption, current_rejection, current_distribution),
            timestamp=datetime.now()
        )
