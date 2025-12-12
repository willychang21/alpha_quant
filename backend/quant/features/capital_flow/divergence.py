"""
Divergence Detector Module.

Detects bullish/bearish divergences between price and volume indicators
(MFI, OBV) to identify potential accumulation zones and smart money activity.

Key concepts:
- Bullish Divergence: Price makes lower low, indicator makes higher low
- Bearish Divergence: Price makes higher high, indicator makes lower high
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import logging

from quant.features.capital_flow.models import DivergenceSignal

logger = logging.getLogger(__name__)


class DivergenceDetector:
    """Detects bullish/bearish divergences between price and volume indicators."""
    
    def __init__(self, lookback: int = 20, min_swing_pct: float = 0.02):
        """Initialize DivergenceDetector.
        
        Args:
            lookback: Period to look for swing highs/lows (default: 20)
            min_swing_pct: Minimum percentage move to qualify as swing (default: 2%)
        """
        self.lookback = lookback
        self.min_swing_pct = min_swing_pct
    
    def find_swing_lows(
        self,
        series: pd.Series,
        window: int = 5
    ) -> List[Tuple[int, float]]:
        """Find local minima (swing lows) in a series.
        
        Args:
            series: Price or indicator series
            window: Window size for local minimum detection
            
        Returns:
            List of (index, value) tuples for each swing low
        """
        swing_lows = []
        
        if len(series) < window * 2 + 1:
            return swing_lows
        
        try:
            values = series.values
            
            for i in range(window, len(values) - window):
                # Check if this is a local minimum
                is_low = True
                center_val = values[i]
                
                for j in range(i - window, i + window + 1):
                    if j != i and values[j] < center_val:
                        is_low = False
                        break
                
                if is_low:
                    # Check if it's a significant swing (% from recent peak)
                    recent_high = values[max(0, i-self.lookback):i+1].max()
                    if recent_high > 0:
                        swing_pct = (recent_high - center_val) / recent_high
                        if swing_pct >= self.min_swing_pct:
                            swing_lows.append((i, center_val))
            
            return swing_lows
            
        except Exception as e:
            logger.debug(f"Swing low detection failed: {e}")
            return []
    
    def find_swing_highs(
        self,
        series: pd.Series,
        window: int = 5
    ) -> List[Tuple[int, float]]:
        """Find local maxima (swing highs) in a series.
        
        Args:
            series: Price or indicator series
            window: Window size for local maximum detection
            
        Returns:
            List of (index, value) tuples for each swing high
        """
        swing_highs = []
        
        if len(series) < window * 2 + 1:
            return swing_highs
        
        try:
            values = series.values
            
            for i in range(window, len(values) - window):
                # Check if this is a local maximum
                is_high = True
                center_val = values[i]
                
                for j in range(i - window, i + window + 1):
                    if j != i and values[j] > center_val:
                        is_high = False
                        break
                
                if is_high:
                    # Check if it's a significant swing
                    recent_low = values[max(0, i-self.lookback):i+1].min()
                    if recent_low > 0:
                        swing_pct = (center_val - recent_low) / recent_low
                        if swing_pct >= self.min_swing_pct:
                            swing_highs.append((i, center_val))
            
            return swing_highs
            
        except Exception as e:
            logger.debug(f"Swing high detection failed: {e}")
            return []
    
    def detect_bullish_divergence(
        self,
        price: pd.Series,
        indicator: pd.Series
    ) -> DivergenceSignal:
        """Detect bullish divergence: price makes lower low, indicator makes higher low.
        
        Args:
            price: Price series (typically Close prices)
            indicator: Indicator series (MFI or OBV)
            
        Returns:
            DivergenceSignal with detection result
        """
        try:
            # Find swing lows in both series
            price_lows = self.find_swing_lows(price)
            indicator_lows = self.find_swing_lows(indicator)
            
            if len(price_lows) < 2 or len(indicator_lows) < 2:
                return DivergenceSignal(
                    divergence_type='none',
                    confidence=0.0,
                    lookback_bars=self.lookback
                )
            
            # Get the two most recent lows
            recent_price_lows = sorted(price_lows, key=lambda x: x[0])[-2:]
            recent_indicator_lows = sorted(indicator_lows, key=lambda x: x[0])[-2:]
            
            prev_price_low = recent_price_lows[0][1]
            curr_price_low = recent_price_lows[1][1]
            
            prev_ind_low = recent_indicator_lows[0][1]
            curr_ind_low = recent_indicator_lows[1][1]
            
            # Bullish divergence: price lower low, indicator higher low
            if curr_price_low < prev_price_low and curr_ind_low > prev_ind_low:
                # Calculate confidence based on divergence magnitude
                price_change = (prev_price_low - curr_price_low) / prev_price_low
                ind_change = (curr_ind_low - prev_ind_low) / (abs(prev_ind_low) + 1e-10)
                
                # Confidence is proportional to the divergence strength
                confidence = min(1.0, (abs(price_change) + abs(ind_change)) / 0.1)
                
                return DivergenceSignal(
                    divergence_type='bullish',
                    confidence=confidence,
                    price_swing=(prev_price_low, curr_price_low),
                    indicator_swing=(prev_ind_low, curr_ind_low),
                    lookback_bars=self.lookback
                )
            
            return DivergenceSignal(
                divergence_type='none',
                confidence=0.0,
                price_swing=(prev_price_low, curr_price_low),
                indicator_swing=(prev_ind_low, curr_ind_low),
                lookback_bars=self.lookback
            )
            
        except Exception as e:
            logger.debug(f"Bullish divergence detection failed: {e}")
            return DivergenceSignal(
                divergence_type='none',
                confidence=0.0,
                lookback_bars=self.lookback
            )
    
    def detect_bearish_divergence(
        self,
        price: pd.Series,
        indicator: pd.Series
    ) -> DivergenceSignal:
        """Detect bearish divergence: price makes higher high, indicator makes lower high.
        
        Args:
            price: Price series
            indicator: Indicator series
            
        Returns:
            DivergenceSignal with detection result
        """
        try:
            # Find swing highs in both series
            price_highs = self.find_swing_highs(price)
            indicator_highs = self.find_swing_highs(indicator)
            
            if len(price_highs) < 2 or len(indicator_highs) < 2:
                return DivergenceSignal(
                    divergence_type='none',
                    confidence=0.0,
                    lookback_bars=self.lookback
                )
            
            # Get the two most recent highs
            recent_price_highs = sorted(price_highs, key=lambda x: x[0])[-2:]
            recent_indicator_highs = sorted(indicator_highs, key=lambda x: x[0])[-2:]
            
            prev_price_high = recent_price_highs[0][1]
            curr_price_high = recent_price_highs[1][1]
            
            prev_ind_high = recent_indicator_highs[0][1]
            curr_ind_high = recent_indicator_highs[1][1]
            
            # Bearish divergence: price higher high, indicator lower high
            if curr_price_high > prev_price_high and curr_ind_high < prev_ind_high:
                # Calculate confidence
                price_change = (curr_price_high - prev_price_high) / prev_price_high
                ind_change = (prev_ind_high - curr_ind_high) / (abs(prev_ind_high) + 1e-10)
                
                confidence = min(1.0, (abs(price_change) + abs(ind_change)) / 0.1)
                
                return DivergenceSignal(
                    divergence_type='bearish',
                    confidence=confidence,
                    price_swing=(prev_price_high, curr_price_high),
                    indicator_swing=(prev_ind_high, curr_ind_high),
                    lookback_bars=self.lookback
                )
            
            return DivergenceSignal(
                divergence_type='none',
                confidence=0.0,
                price_swing=(prev_price_high, curr_price_high),
                indicator_swing=(prev_ind_high, curr_ind_high),
                lookback_bars=self.lookback
            )
            
        except Exception as e:
            logger.debug(f"Bearish divergence detection failed: {e}")
            return DivergenceSignal(
                divergence_type='none',
                confidence=0.0,
                lookback_bars=self.lookback
            )
    
    def calculate_divergence_score(
        self,
        price: pd.Series,
        mfi: pd.Series,
        obv: pd.Series
    ) -> float:
        """Calculate composite divergence score from -1 (bearish) to 1 (bullish).
        
        Combines MFI and OBV divergence signals into a single score.
        
        Args:
            price: Close price series
            mfi: MFI series
            obv: OBV series
            
        Returns:
            Composite divergence score in range [-1, 1]
        """
        try:
            # Detect divergences for both indicators
            mfi_bullish = self.detect_bullish_divergence(price, mfi)
            mfi_bearish = self.detect_bearish_divergence(price, mfi)
            
            obv_bullish = self.detect_bullish_divergence(price, obv)
            obv_bearish = self.detect_bearish_divergence(price, obv)
            
            # Calculate individual scores
            mfi_score = 0.0
            if mfi_bullish.divergence_type == 'bullish':
                mfi_score = mfi_bullish.confidence
            elif mfi_bearish.divergence_type == 'bearish':
                mfi_score = -mfi_bearish.confidence
            
            obv_score = 0.0
            if obv_bullish.divergence_type == 'bullish':
                obv_score = obv_bullish.confidence
            elif obv_bearish.divergence_type == 'bearish':
                obv_score = -obv_bearish.confidence
            
            # Combine with equal weights (can be adjusted)
            composite_score = (mfi_score * 0.5 + obv_score * 0.5)
            
            # Clamp to [-1, 1]
            return max(-1.0, min(1.0, composite_score))
            
        except Exception as e:
            logger.debug(f"Divergence score calculation failed: {e}")
            return 0.0
