"""
Money Flow Calculator Module.

Implements Money Flow Index (MFI) and On-Balance Volume (OBV) calculations
for detecting institutional accumulation/distribution patterns.

Reference:
- MFI: Volume-weighted RSI indicator (0-100 range)
- OBV: Cumulative volume based on price direction
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MoneyFlowCalculator:
    """Calculates Money Flow Index and On-Balance Volume indicators.
    
    MFI is a volume-weighted momentum indicator that measures buying and
    selling pressure. OBV tracks cumulative volume flow to detect
    accumulation/distribution.
    """
    
    def __init__(self, mfi_period: int = 14):
        """Initialize MoneyFlowCalculator.
        
        Args:
            mfi_period: Lookback period for MFI calculation (default: 14)
        """
        self.mfi_period = mfi_period
    
    def calculate_mfi(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: Optional[int] = None
    ) -> pd.Series:
        """Calculate Money Flow Index.
        
        MFI = 100 - (100 / (1 + Money Flow Ratio))
        Money Flow Ratio = Positive Money Flow / Negative Money Flow
        
        Args:
            high: High prices series
            low: Low prices series
            close: Close prices series
            volume: Volume series
            period: Optional override for lookback period
            
        Returns:
            Series with MFI values in range [0, 100]
        """
        period = period or self.mfi_period
        
        # Validate inputs
        if len(high) < period + 1:
            logger.warning(f"Insufficient data for MFI: {len(high)} < {period + 1}")
            return pd.Series([50.0] * len(high), index=high.index)
        
        try:
            # Calculate Typical Price: (High + Low + Close) / 3
            typical_price = (high + low + close) / 3
            
            # Calculate Raw Money Flow: Typical Price * Volume
            raw_money_flow = typical_price * volume
            
            # Determine positive/negative flow based on typical price direction
            tp_diff = typical_price.diff()
            
            positive_flow = pd.Series(0.0, index=raw_money_flow.index)
            negative_flow = pd.Series(0.0, index=raw_money_flow.index)
            
            positive_mask = tp_diff > 0
            negative_mask = tp_diff < 0
            
            positive_flow[positive_mask] = raw_money_flow[positive_mask]
            negative_flow[negative_mask] = raw_money_flow[negative_mask]
            
            # Calculate rolling sums
            positive_sum = positive_flow.rolling(window=period).sum()
            negative_sum = negative_flow.rolling(window=period).sum()
            
            # Calculate Money Flow Ratio (avoid division by zero)
            epsilon = 1e-10
            money_flow_ratio = positive_sum / (negative_sum + epsilon)
            
            # Calculate MFI: 100 - (100 / (1 + MFR))
            mfi = 100 - (100 / (1 + money_flow_ratio))
            
            # Clamp to valid range [0, 100]
            mfi = mfi.clip(lower=0.0, upper=100.0)
            
            # Handle NaN values with neutral default
            mfi = mfi.fillna(50.0)
            
            return mfi
            
        except Exception as e:
            logger.error(f"MFI calculation failed: {e}")
            return pd.Series([50.0] * len(high), index=high.index)
    
    def calculate_obv(
        self,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """Calculate On-Balance Volume.
        
        OBV = Previous OBV + Volume (if close > previous close)
        OBV = Previous OBV - Volume (if close < previous close)
        OBV = Previous OBV (if close == previous close)
        
        Args:
            close: Close prices series
            volume: Volume series
            
        Returns:
            Series with OBV values
        """
        if len(close) < 2:
            logger.warning("Insufficient data for OBV calculation")
            return pd.Series([0.0] * len(close), index=close.index)
        
        try:
            # Calculate price direction
            price_diff = close.diff()
            
            # Determine volume direction: +vol, -vol, or 0
            signed_volume = pd.Series(0.0, index=volume.index)
            signed_volume[price_diff > 0] = volume[price_diff > 0]
            signed_volume[price_diff < 0] = -volume[price_diff < 0]
            # No change in volume when price unchanged (stays 0)
            
            # Cumulative sum for OBV
            obv = signed_volume.cumsum()
            
            # Handle NaN/Inf
            obv = obv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            return obv
            
        except Exception as e:
            logger.error(f"OBV calculation failed: {e}")
            return pd.Series([0.0] * len(close), index=close.index)
    
    def normalize_obv(
        self,
        obv: pd.Series,
        lookback: int = 252
    ) -> pd.Series:
        """Normalize OBV to Z-score for cross-stock comparison.
        
        Z-score = (OBV - mean) / std
        
        Args:
            obv: OBV series
            lookback: Lookback period for Z-score calculation (default: 252)
            
        Returns:
            Series with Z-score normalized OBV
        """
        if len(obv) < lookback:
            # Use available data if less than lookback
            effective_lookback = max(20, len(obv))
        else:
            effective_lookback = lookback
        
        try:
            rolling_mean = obv.rolling(window=effective_lookback, min_periods=20).mean()
            rolling_std = obv.rolling(window=effective_lookback, min_periods=20).std()
            
            # Avoid division by zero
            epsilon = 1e-10
            z_score = (obv - rolling_mean) / (rolling_std + epsilon)
            
            # Handle NaN/Inf values
            z_score = z_score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            # Clip extreme values for stability
            z_score = z_score.clip(lower=-5.0, upper=5.0)
            
            return z_score
            
        except Exception as e:
            logger.error(f"OBV normalization failed: {e}")
            return pd.Series([0.0] * len(obv), index=obv.index)
    
    def classify_mfi(self, mfi_value: float) -> str:
        """Classify MFI value into signal category.
        
        Args:
            mfi_value: MFI value (0-100)
            
        Returns:
            'oversold' if < 20, 'overbought' if > 80, else 'neutral'
        """
        if mfi_value < 20:
            return 'oversold'
        elif mfi_value > 80:
            return 'overbought'
        else:
            return 'neutral'
    
    def classify_obv_trend(
        self,
        obv: pd.Series,
        lookback: int = 20
    ) -> str:
        """Classify OBV trend direction.
        
        Args:
            obv: OBV series
            lookback: Period to analyze trend
            
        Returns:
            'accumulation' if OBV trending up, 'distribution' if down, else 'neutral'
        """
        if len(obv) < lookback:
            return 'neutral'
        
        try:
            recent_obv = obv.iloc[-lookback:]
            
            # Simple linear regression slope
            x = np.arange(len(recent_obv))
            y = recent_obv.values
            
            # Remove NaN
            valid_mask = ~np.isnan(y)
            if valid_mask.sum() < 5:
                return 'neutral'
            
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            
            slope = np.polyfit(x_valid, y_valid, 1)[0]
            
            # Normalize slope by average OBV magnitude
            avg_obv = np.abs(y_valid).mean() + 1e-10
            normalized_slope = slope / avg_obv
            
            # Threshold for trend classification
            if normalized_slope > 0.01:
                return 'accumulation'
            elif normalized_slope < -0.01:
                return 'distribution'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.debug(f"OBV trend classification failed: {e}")
            return 'neutral'
