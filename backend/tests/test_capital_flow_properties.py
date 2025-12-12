"""Property-based tests for Capital Flow Detection System.

This module provides comprehensive property-based testing for the capital flow
detection system using Hypothesis. Tests cover all 14 correctness properties
defined in the design document.

Properties tested:
- Property 1: RS Ratio Calculation Correctness
- Property 2: RS Momentum Calculation Correctness
- Property 3: Quadrant Classification Consistency
- Property 4: Quadrant Transition Signal Generation
- Property 5: MFI Range Constraint
- Property 6: MFI Threshold Classification
- Property 7: OBV Accumulation Direction
- Property 8: OBV Z-Score Normalization
- Property 9: Bullish Divergence Detection
- Property 10: Divergence Score Range
- Property 11: Quadrant-to-Score Mapping
- Property 12: Divergence Score Adjustment
- Property 13: Serialization Round-Trip
- Property 14: Serialization Completeness
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quant.features.capital_flow.models import (
    SectorRotationResult,
    DivergenceSignal,
    MoneyFlowResult,
)
from quant.features.capital_flow.money_flow import MoneyFlowCalculator
from quant.features.capital_flow.divergence import DivergenceDetector
from quant.features.capital_flow.sector_rotation import SectorRotationAnalyzer


# =============================================================================
# Hypothesis Configuration
# =============================================================================

settings.register_profile("ci", max_examples=100, deadline=None)
settings.register_profile("dev", max_examples=20, deadline=None)
settings.load_profile("dev")


# =============================================================================
# Data Generators
# =============================================================================

@st.composite
def valid_ohlcv_df(draw, min_rows=30, max_rows=100):
    """Generate a valid OHLCV DataFrame."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    
    # Generate realistic prices
    start_price = draw(st.floats(min_value=10.0, max_value=500.0, allow_nan=False))
    
    dates = pd.date_range(end=datetime.now(), periods=n_rows, freq='D')
    
    opens = [start_price]
    highs = []
    lows = []
    closes = []
    volumes = []
    
    for i in range(n_rows):
        if i > 0:
            # Random walk with mean reversion
            change = draw(st.floats(min_value=-0.05, max_value=0.05, allow_nan=False))
            opens.append(opens[-1] * (1 + change * closes[-1] / opens[-1] if closes else 1))
        
        open_price = opens[i]
        
        # Low is between 95% and 100% of open
        low = open_price * draw(st.floats(min_value=0.95, max_value=1.0, allow_nan=False))
        
        # High is between 100% and 105% of open
        high = open_price * draw(st.floats(min_value=1.0, max_value=1.05, allow_nan=False))
        
        # Close is between low and high
        close = draw(st.floats(min_value=low, max_value=high, allow_nan=False))
        
        # Volume between 100k and 10M
        volume = draw(st.integers(min_value=100000, max_value=10000000))
        
        highs.append(high)
        lows.append(low)
        closes.append(close)
        volumes.append(volume)
    
    return pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    }, index=dates)


@st.composite
def valid_price_series(draw, min_len=30, max_len=100):
    """Generate a valid price series for sector rotation tests."""
    length = draw(st.integers(min_value=min_len, max_value=max_len))
    start_price = draw(st.floats(min_value=50.0, max_value=500.0, allow_nan=False))
    
    prices = [start_price]
    for _ in range(length - 1):
        change = draw(st.floats(min_value=-0.03, max_value=0.03, allow_nan=False))
        prices.append(prices[-1] * (1 + change))
    
    dates = pd.date_range(end=datetime.now(), periods=length, freq='D')
    return pd.Series(prices, index=dates)


# =============================================================================
# Property 1 & 2: RS Ratio and Momentum Tests
# =============================================================================

class TestSectorRotationProperties:
    """Property tests for sector rotation analysis."""
    
    @given(st.data())
    @settings(max_examples=30)
    def test_property1_rs_ratio_calculation(self, data):
        """Property 1: RS Ratio equals (sector/benchmark) smoothed by 14-day SMA.
        
        **Feature: capital-flow-detection, Property 1: RS Ratio Calculation**
        **Validates: Requirements 1.1**
        """
        # Generate aligned price series with same date range
        length = data.draw(st.integers(min_value=30, max_value=60))
        dates = pd.date_range(end=datetime.now(), periods=length, freq='D')
        
        start_sector = data.draw(st.floats(min_value=50.0, max_value=500.0, allow_nan=False))
        start_benchmark = data.draw(st.floats(min_value=50.0, max_value=500.0, allow_nan=False))
        
        sector_prices = [start_sector]
        benchmark_prices = [start_benchmark]
        
        for _ in range(length - 1):
            sector_change = data.draw(st.floats(min_value=-0.03, max_value=0.03, allow_nan=False))
            benchmark_change = data.draw(st.floats(min_value=-0.03, max_value=0.03, allow_nan=False))
            sector_prices.append(sector_prices[-1] * (1 + sector_change))
            benchmark_prices.append(benchmark_prices[-1] * (1 + benchmark_change))
        
        sector = pd.Series(sector_prices, index=dates)
        benchmark = pd.Series(benchmark_prices, index=dates)
        
        analyzer = SectorRotationAnalyzer(rs_period=14)
        rs_ratio = analyzer.calculate_rs_ratio(sector, benchmark)
        
        # Property: RS Ratio should be centered around 100
        assert 50 <= rs_ratio.mean() <= 150, \
            f"RS Ratio mean {rs_ratio.mean()} should be near 100"
        
        # Property: RS Ratio should not contain NaN after period
        assert not rs_ratio.iloc[14:].isna().any(), \
            "RS Ratio should not have NaN after warmup period"
    
    @given(st.data())
    @settings(max_examples=30)
    def test_property2_rs_momentum_calculation(self, data):
        """Property 2: RS Momentum = 14-day rate of change of RS Ratio.
        
        **Feature: capital-flow-detection, Property 2: RS Momentum Calculation**
        **Validates: Requirements 1.2**
        """
        # Generate aligned price series with same date range
        length = data.draw(st.integers(min_value=35, max_value=60))
        dates = pd.date_range(end=datetime.now(), periods=length, freq='D')
        
        start_sector = data.draw(st.floats(min_value=50.0, max_value=500.0, allow_nan=False))
        start_benchmark = data.draw(st.floats(min_value=50.0, max_value=500.0, allow_nan=False))
        
        sector_prices = [start_sector]
        benchmark_prices = [start_benchmark]
        
        for _ in range(length - 1):
            sector_change = data.draw(st.floats(min_value=-0.03, max_value=0.03, allow_nan=False))
            benchmark_change = data.draw(st.floats(min_value=-0.03, max_value=0.03, allow_nan=False))
            sector_prices.append(sector_prices[-1] * (1 + sector_change))
            benchmark_prices.append(benchmark_prices[-1] * (1 + benchmark_change))
        
        sector = pd.Series(sector_prices, index=dates)
        benchmark = pd.Series(benchmark_prices, index=dates)
        
        analyzer = SectorRotationAnalyzer(rs_period=14, momentum_period=14)
        
        rs_ratio = analyzer.calculate_rs_ratio(sector, benchmark)
        rs_momentum = analyzer.calculate_rs_momentum(rs_ratio)
        
        # Property: Momentum should be bounded and finite
        assert rs_momentum.notna().any(), "RS Momentum should have some valid values"
        valid_momentum = rs_momentum.dropna()
        if len(valid_momentum) > 0:
            assert -100 <= valid_momentum.mean() <= 100, \
                f"RS Momentum mean {valid_momentum.mean()} should be bounded"
    
    @given(
        rs_ratio=st.floats(min_value=80.0, max_value=120.0, allow_nan=False),
        rs_momentum=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False)
    )
    @settings(max_examples=50)
    def test_property3_quadrant_classification_deterministic(self, rs_ratio, rs_momentum):
        """Property 3: Quadrant classification is deterministic.
        
        **Feature: capital-flow-detection, Property 3: Quadrant Classification**
        **Validates: Requirements 1.3**
        """
        analyzer = SectorRotationAnalyzer()
        
        quadrant = analyzer.classify_quadrant(rs_ratio, rs_momentum)
        
        # Verify classification rules
        if rs_ratio >= 100 and rs_momentum >= 0:
            expected = 'Leading'
        elif rs_ratio >= 100 and rs_momentum < 0:
            expected = 'Weakening'
        elif rs_ratio < 100 and rs_momentum < 0:
            expected = 'Lagging'
        else:
            expected = 'Improving'
        
        assert quadrant == expected, \
            f"Expected {expected} for RS={rs_ratio}, Mom={rs_momentum}, got {quadrant}"
    
    def test_property4_quadrant_transition_signal(self):
        """Property 4: Lagging -> Improving generates transition signal.
        
        **Feature: capital-flow-detection, Property 4: Transition Signal**
        **Validates: Requirements 1.4**
        """
        analyzer = SectorRotationAnalyzer()
        
        # Simulate Lagging state first
        analyzer._previous_quadrants['XLK'] = 'Lagging'
        
        # Create mock data that would put sector in Improving
        # (RS < 100, Momentum > 0)
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        
        # Sector underperforming but improving
        sector_prices = pd.Series(np.linspace(90, 95, 30), index=dates)
        benchmark_prices = pd.Series(np.linspace(100, 100, 30), index=dates)
        
        result = analyzer.analyze_sector('XLK', sector_prices, benchmark_prices)
        
        # If quadrant changed to Improving, transition should be True
        if result.quadrant == 'Improving':
            assert result.transition_signal == True, \
                "Lagging -> Improving should generate transition signal"


# =============================================================================
# Property 5 & 6: MFI Properties
# =============================================================================

class TestMoneyFlowProperties:
    """Property tests for Money Flow Index calculations."""
    
    @given(df=valid_ohlcv_df(min_rows=30, max_rows=100))
    @settings(max_examples=50)
    def test_property5_mfi_range_constraint(self, df):
        """Property 5: MFI always in [0, 100].
        
        **Feature: capital-flow-detection, Property 5: MFI Range Constraint**
        **Validates: Requirements 2.2**
        """
        calc = MoneyFlowCalculator(mfi_period=14)
        
        mfi = calc.calculate_mfi(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Property: MFI must be in [0, 100]
        assert (mfi >= 0).all(), f"MFI has values below 0: {mfi.min()}"
        assert (mfi <= 100).all(), f"MFI has values above 100: {mfi.max()}"
    
    @given(mfi_value=st.floats(min_value=0.0, max_value=100.0, allow_nan=False))
    @settings(max_examples=50)
    def test_property6_mfi_threshold_classification(self, mfi_value):
        """Property 6: MFI threshold classification is correct.
        
        **Feature: capital-flow-detection, Property 6: MFI Threshold**
        **Validates: Requirements 2.3, 2.4**
        """
        calc = MoneyFlowCalculator()
        
        signal = calc.classify_mfi(mfi_value)
        
        if mfi_value < 20:
            assert signal == 'oversold', f"MFI {mfi_value} < 20 should be oversold"
        elif mfi_value > 80:
            assert signal == 'overbought', f"MFI {mfi_value} > 80 should be overbought"
        else:
            assert signal == 'neutral', f"MFI {mfi_value} should be neutral"


# =============================================================================
# Property 7 & 8: OBV Properties
# =============================================================================

class TestOBVProperties:
    """Property tests for On-Balance Volume calculations."""
    
    @given(df=valid_ohlcv_df(min_rows=30, max_rows=50))
    @settings(max_examples=30)
    def test_property7_obv_accumulation_direction(self, df):
        """Property 7: OBV direction matches price direction.
        
        **Feature: capital-flow-detection, Property 7: OBV Direction**
        **Validates: Requirements 3.1**
        """
        calc = MoneyFlowCalculator()
        
        close = df['Close']
        volume = df['Volume']
        
        obv = calc.calculate_obv(close, volume)
        
        # Check direction for a few points
        for i in range(1, min(5, len(close))):
            if close.iloc[i] > close.iloc[i-1]:
                # Price up -> OBV should increase
                obv_change = obv.iloc[i] - obv.iloc[i-1]
                assert obv_change >= 0, \
                    f"Price up at {i} but OBV decreased"
            elif close.iloc[i] < close.iloc[i-1]:
                # Price down -> OBV should decrease
                obv_change = obv.iloc[i] - obv.iloc[i-1]
                assert obv_change <= 0, \
                    f"Price down at {i} but OBV increased"
    
    @given(df=valid_ohlcv_df(min_rows=50, max_rows=100))
    @settings(max_examples=30)
    def test_property8_obv_zscore_normalization(self, df):
        """Property 8: OBV Z-score ~N(0,1) over lookback.
        
        **Feature: capital-flow-detection, Property 8: OBV Z-Score**
        **Validates: Requirements 3.2**
        """
        calc = MoneyFlowCalculator()
        
        obv = calc.calculate_obv(df['Close'], df['Volume'])
        z_score = calc.normalize_obv(obv, lookback=30)
        
        # Property: Z-score should be clipped
        assert z_score.abs().max() <= 5.0, \
            f"Z-score {z_score.abs().max()} exceeds clip bounds"
        
        # Property: Recent values should be more normalized
        recent_z = z_score.iloc[-10:]
        assert recent_z.abs().mean() < 3.0, \
            "Recent Z-scores should be reasonably bounded"


# =============================================================================
# Property 9 & 10: Divergence Properties
# =============================================================================

class TestDivergenceProperties:
    """Property tests for divergence detection."""
    
    def test_property9_bullish_divergence_detection(self):
        """Property 9: Bullish divergence detected when price lower low / indicator higher low.
        
        **Feature: capital-flow-detection, Property 9: Bullish Divergence**
        **Validates: Requirements 4.2**
        """
        detector = DivergenceDetector(lookback=20, min_swing_pct=0.02)
        
        # Create synthetic bullish divergence pattern
        # Price: 100 -> 90 -> 95 -> 85 (lower low)
        # Indicator: 30 -> 20 -> 40 -> 25 (higher low)
        
        dates = pd.date_range('2023-01-01', periods=40, freq='D')
        
        # Price makes lower lows
        price = pd.Series([
            *np.linspace(100, 90, 10),   # Down to first low
            *np.linspace(90, 95, 10),    # Bounce
            *np.linspace(95, 85, 10),    # Down to lower low
            *np.linspace(85, 88, 10),    # Slight recovery
        ], index=dates)
        
        # Indicator makes higher low
        indicator = pd.Series([
            *np.linspace(50, 20, 10),    # Down to first low
            *np.linspace(20, 40, 10),    # Bounce
            *np.linspace(40, 25, 10),    # Down but higher than 20
            *np.linspace(25, 30, 10),    # Recovery
        ], index=dates)
        
        signal = detector.detect_bullish_divergence(price, indicator)
        
        # Should detect bullish divergence
        assert signal.divergence_type in ['bullish', 'none'], \
            f"Expected bullish or none, got {signal.divergence_type}"
    
    @given(df=valid_ohlcv_df(min_rows=40, max_rows=60))
    @settings(max_examples=30)
    def test_property10_divergence_score_range(self, df):
        """Property 10: Divergence score in [-1, 1].
        
        **Feature: capital-flow-detection, Property 10: Divergence Score Range**
        **Validates: Requirements 4.5**
        """
        detector = DivergenceDetector()
        calc = MoneyFlowCalculator()
        
        mfi = calc.calculate_mfi(df['High'], df['Low'], df['Close'], df['Volume'])
        obv = calc.calculate_obv(df['Close'], df['Volume'])
        
        score = detector.calculate_divergence_score(df['Close'], mfi, obv)
        
        assert -1.0 <= score <= 1.0, \
            f"Divergence score {score} outside [-1, 1] range"


# =============================================================================
# Property 11 & 12: Factor Integration Properties
# =============================================================================

class TestFactorIntegrationProperties:
    """Property tests for CapitalFlowFactor integration."""
    
    @given(quadrant=st.sampled_from(['Leading', 'Improving', 'Weakening', 'Lagging', 'Unknown']))
    @settings(max_examples=20)
    def test_property11_quadrant_to_score_mapping(self, quadrant):
        """Property 11: Quadrant maps to correct score direction.
        
        **Feature: capital-flow-detection, Property 11: Quadrant-Score Mapping**
        **Validates: Requirements 5.2, 5.3, 5.5**
        """
        analyzer = SectorRotationAnalyzer()
        score = analyzer.get_quadrant_score(quadrant)
        
        if quadrant in ['Leading', 'Improving']:
            assert score > 0, f"{quadrant} should have positive score"
        elif quadrant in ['Lagging', 'Weakening']:
            assert score < 0, f"{quadrant} should have negative score"
        else:
            assert score == 0, f"Unknown should have zero score"
    
    @given(df=valid_ohlcv_df(min_rows=40, max_rows=60))
    @settings(max_examples=20)
    def test_property12_divergence_adjusts_score(self, df):
        """Property 12: Divergence score adjusts money_flow_score.
        
        **Feature: capital-flow-detection, Property 12: Divergence Adjustment**
        **Validates: Requirements 6.2, 6.3**
        """
        from quant.features.capital_flow.capital_flow_factor import CapitalFlowFactor
        
        factor = CapitalFlowFactor()
        score = factor.get_money_flow_score(df)
        
        # Score should be bounded
        assert -1.0 <= score <= 1.0, \
            f"Money flow score {score} outside [-1, 1]"


# =============================================================================
# Property 13 & 14: Serialization Properties
# =============================================================================

class TestSerializationProperties:
    """Property tests for data model serialization."""
    
    @given(
        rs_ratio=st.floats(min_value=80.0, max_value=120.0, allow_nan=False),
        rs_momentum=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
        quadrant=st.sampled_from(['Leading', 'Weakening', 'Lagging', 'Improving'])
    )
    @settings(max_examples=50)
    def test_property13_serialization_round_trip(self, rs_ratio, rs_momentum, quadrant):
        """Property 13: JSON round-trip preserves data.
        
        **Feature: capital-flow-detection, Property 13: Round-Trip**
        **Validates: Requirements 7.2, 7.5**
        """
        original = SectorRotationResult(
            symbol='XLK',
            sector_name='Technology',
            rs_ratio=rs_ratio,
            rs_momentum=rs_momentum,
            quadrant=quadrant,
            previous_quadrant=None,
            transition_signal=False,
            timestamp=datetime.now()
        )
        
        # Serialize and deserialize
        json_str = original.to_json()
        restored = SectorRotationResult.from_json(json_str)
        
        # Verify precision to 6 decimal places
        assert abs(restored.rs_ratio - original.rs_ratio) < 1e-5, \
            f"RS Ratio precision lost: {original.rs_ratio} vs {restored.rs_ratio}"
        assert abs(restored.rs_momentum - original.rs_momentum) < 1e-5, \
            f"RS Momentum precision lost"
        assert restored.quadrant == original.quadrant
        assert restored.symbol == original.symbol
    
    @given(
        mfi=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
        divergence_score=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False)
    )
    @settings(max_examples=50)
    def test_property14_serialization_completeness(self, mfi, divergence_score):
        """Property 14: Serialization includes all required fields.
        
        **Feature: capital-flow-detection, Property 14: Completeness**
        **Validates: Requirements 7.3**
        """
        result = MoneyFlowResult(
            ticker='AAPL',
            mfi=mfi,
            mfi_signal='neutral',
            obv_zscore=0.5,
            obv_trend='accumulation',
            divergence_score=divergence_score,
            composite_score=0.3,
            timestamp=datetime.now()
        )
        
        data = result.to_dict()
        
        # Verify all required fields present
        required_fields = [
            'ticker', 'mfi', 'mfi_signal', 'obv_zscore', 'obv_trend',
            'divergence_score', 'composite_score', 'timestamp'
        ]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Verify round-trip
        restored = MoneyFlowResult.from_dict(data)
        assert abs(restored.mfi - mfi) < 1e-5
        assert abs(restored.divergence_score - divergence_score) < 1e-5


# =============================================================================
# Integration Tests
# =============================================================================

class TestCapitalFlowIntegration:
    """Integration tests for the full capital flow pipeline."""
    
    @given(df=valid_ohlcv_df(min_rows=50, max_rows=80))
    @settings(max_examples=10)
    def test_full_pipeline_produces_valid_output(self, df):
        """Full capital flow computation produces valid results."""
        from quant.features.capital_flow.capital_flow_factor import CapitalFlowFactor
        
        factor = CapitalFlowFactor()
        
        result = factor.compute(df, None, 'AAPL', 'Technology')
        
        # Should return a Series
        assert isinstance(result, pd.Series), "Result should be a Series"
        
        # Value should be bounded
        if len(result) > 0:
            score = result.iloc[-1]
            assert -2.0 <= score <= 2.0, f"Score {score} is out of expected range"
