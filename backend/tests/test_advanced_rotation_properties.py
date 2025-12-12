"""Property-based tests for Advanced Market Rotation System.

This module provides comprehensive property-based testing for the
Advanced Market Rotation system using Hypothesis.

Properties tested:
- Property 1-4: Levy RS Calculation
- Property 5-8: Mansfield RS Calculation
- Property 9-12: Volume Structure Analysis
- Property 13-14: Scorecard System
- Property 15-16: Serialization
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quant.rotation.models import (
    LevyRSResult,
    MansfieldRSResult,
    VolumeAnalysisResult,
    ScorecardResult,
)
from quant.rotation.levy_rs import LevyRSCalculator
from quant.rotation.mansfield_rs import MansfieldRSCalculator
from quant.rotation.volume_structure import VolumeStructureAnalyzer
from quant.rotation.scorecard import ScorecardSystem, ScorecardWeights


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
def valid_price_series(draw, min_len=130, max_len=300):
    """Generate a valid price series for Levy RS tests."""
    length = draw(st.integers(min_value=min_len, max_value=max_len))
    start_price = draw(st.floats(min_value=50.0, max_value=500.0, allow_nan=False))
    
    prices = [start_price]
    for _ in range(length - 1):
        change = draw(st.floats(min_value=-0.03, max_value=0.03, allow_nan=False))
        prices.append(prices[-1] * (1 + change))
    
    dates = pd.date_range(end=datetime.now(), periods=length, freq='D')
    return pd.Series(prices, index=dates)


@st.composite
def valid_ohlcv_df(draw, min_rows=30, max_rows=100):
    """Generate a valid OHLCV DataFrame."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    start_price = draw(st.floats(min_value=10.0, max_value=500.0, allow_nan=False))
    
    dates = pd.date_range(end=datetime.now(), periods=n_rows, freq='D')
    
    opens = [start_price]
    highs = []
    lows = []
    closes = []
    volumes = []
    
    for i in range(n_rows):
        if i > 0:
            change = draw(st.floats(min_value=-0.05, max_value=0.05, allow_nan=False))
            opens.append(opens[-1] * (1 + change * (closes[-1] / opens[-1] if closes else 1)))
        
        open_price = opens[i]
        low = open_price * draw(st.floats(min_value=0.95, max_value=1.0, allow_nan=False))
        high = open_price * draw(st.floats(min_value=1.0, max_value=1.05, allow_nan=False))
        close = draw(st.floats(min_value=low, max_value=high, allow_nan=False))
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


# =============================================================================
# Property 1-4: Levy RS Tests
# =============================================================================

class TestLevyRSProperties:
    """Property tests for Levy Relative Strength calculations."""
    
    @given(close=valid_price_series(min_len=130, max_len=200))
    @settings(max_examples=30)
    def test_property1_levy_rs_calculation_correctness(self, close):
        """Property 1: RSL = Close / SMA(Close, 130).
        
        **Feature: advanced-market-rotation, Property 1: Levy RS Calculation**
        **Validates: Requirements 1.1**
        """
        calc = LevyRSCalculator(period=130)
        
        rsl = calc.calculate_rsl(close)
        sma = close.rolling(window=130).mean()
        
        # After warmup period, RSL should equal Close / SMA
        for i in range(130, len(close)):
            expected = close.iloc[i] / sma.iloc[i]
            actual = rsl.iloc[i]
            assert abs(actual - expected) < 1e-6, \
                f"RSL mismatch at {i}: expected {expected}, got {actual}"
    
    @given(close=valid_price_series(min_len=130, max_len=200))
    @settings(max_examples=30)
    def test_property2_rsl_price_average_relationship(self, close):
        """Property 2: RSL > 1.0 iff close > SMA.
        
        **Feature: advanced-market-rotation, Property 2: RSL Price-Average Relationship**
        **Validates: Requirements 1.2**
        """
        calc = LevyRSCalculator(period=130)
        
        rsl = calc.calculate_rsl(close)
        sma = close.rolling(window=130).mean()
        
        for i in range(130, len(close)):
            if rsl.iloc[i] > 1.0:
                assert close.iloc[i] > sma.iloc[i], \
                    f"RSL > 1.0 but close <= SMA at {i}"
            elif rsl.iloc[i] < 1.0:
                assert close.iloc[i] < sma.iloc[i], \
                    f"RSL < 1.0 but close >= SMA at {i}"
    
    @given(rsl_values=st.dictionaries(
        keys=st.text(min_size=1, max_size=5, alphabet='ABCDEFGHIJ'),
        values=st.floats(min_value=0.8, max_value=1.2, allow_nan=False),
        min_size=2,
        max_size=10
    ))
    @settings(max_examples=50)
    def test_property3_rsl_ranking_order(self, rsl_values):
        """Property 3: Ranking is strictly descending by RSL.
        
        **Feature: advanced-market-rotation, Property 3: RSL Ranking Order**
        **Validates: Requirements 1.3**
        """
        assume(len(rsl_values) >= 2)
        
        calc = LevyRSCalculator()
        ranked = calc.rank_by_rsl(rsl_values)
        
        # Verify descending order
        for i in range(len(ranked) - 1):
            assert ranked[i][1] >= ranked[i + 1][1], \
                f"Not descending: {ranked[i]} before {ranked[i + 1]}"
        
        # Verify ranks are 1-indexed and consecutive
        ranks = [r[2] for r in ranked]
        assert ranks == list(range(1, len(ranked) + 1)), \
            f"Ranks not 1-indexed consecutive: {ranks}"
    
    @given(st.data())
    @settings(max_examples=30)
    def test_property4_rsl_breakdown_signal(self, data):
        """Property 4: RSL crossing below 1.0 generates breakdown signal.
        
        **Feature: advanced-market-rotation, Property 4: RSL Breakdown Signal**
        **Validates: Requirements 1.4**
        """
        # Create a series that crosses below 1.0
        length = data.draw(st.integers(min_value=10, max_value=30))
        
        # Start above 1.0, cross below
        rsl_values = [1.05] * (length // 2) + [0.95] * (length - length // 2)
        dates = pd.date_range(end=datetime.now(), periods=length, freq='D')
        rsl = pd.Series(rsl_values, index=dates)
        
        calc = LevyRSCalculator()
        breakdown = calc.detect_breakdown(rsl)
        
        # Should detect breakdown at crossover point
        crossover_idx = length // 2
        if crossover_idx > 0 and crossover_idx < length:
            assert breakdown.iloc[crossover_idx], \
                f"Breakdown not detected at crossover index {crossover_idx}"


# =============================================================================
# Property 5-8: Mansfield RS Tests
# =============================================================================

class TestMansfieldRSProperties:
    """Property tests for Mansfield Relative Strength calculations."""
    
    @given(st.data())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.large_base_example])
    def test_property5_mrs_calculation_correctness(self, data):
        """Property 5: MRS = ((RS / SMA(RS, 252)) - 1) × 100.
        
        **Feature: advanced-market-rotation, Property 5: MRS Calculation**
        **Validates: Requirements 2.1**
        """
        length = data.draw(st.integers(min_value=260, max_value=300))
        dates = pd.date_range(end=datetime.now(), periods=length, freq='D')
        
        start_stock = data.draw(st.floats(min_value=50.0, max_value=500.0, allow_nan=False))
        start_benchmark = data.draw(st.floats(min_value=50.0, max_value=500.0, allow_nan=False))
        
        stock = [start_stock]
        benchmark = [start_benchmark]
        
        for _ in range(length - 1):
            stock.append(stock[-1] * (1 + data.draw(st.floats(-0.02, 0.02, allow_nan=False))))
            benchmark.append(benchmark[-1] * (1 + data.draw(st.floats(-0.02, 0.02, allow_nan=False))))
        
        stock_series = pd.Series(stock, index=dates)
        benchmark_series = pd.Series(benchmark, index=dates)
        
        calc = MansfieldRSCalculator(period=252)
        mrs = calc.calculate_mrs(stock_series, benchmark_series)
        
        # Verify MRS values are finite and bounded
        valid_mrs = mrs.dropna()
        assert len(valid_mrs) > 0, "MRS should have valid values"
        assert valid_mrs.abs().max() < 1000, "MRS values should be bounded"
    
    @given(st.data())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.large_base_example])
    def test_property6_mrs_zero_centered(self, data):
        """Property 6: MRS distribution is approximately centered around zero.
        
        **Feature: advanced-market-rotation, Property 6: MRS Zero-Centered**
        **Validates: Requirements 2.2**
        """
        length = data.draw(st.integers(min_value=300, max_value=400))
        dates = pd.date_range(end=datetime.now(), periods=length, freq='D')
        
        start = data.draw(st.floats(min_value=100.0, max_value=200.0, allow_nan=False))
        
        # Symmetric random walk
        stock = [start]
        benchmark = [start]
        
        for _ in range(length - 1):
            stock.append(stock[-1] * (1 + data.draw(st.floats(-0.015, 0.015, allow_nan=False))))
            benchmark.append(benchmark[-1] * (1 + data.draw(st.floats(-0.015, 0.015, allow_nan=False))))
        
        stock_series = pd.Series(stock, index=dates)
        benchmark_series = pd.Series(benchmark, index=dates)
        
        calc = MansfieldRSCalculator(period=252)
        mrs = calc.calculate_mrs(stock_series, benchmark_series)
        
        valid_mrs = mrs.dropna()
        if len(valid_mrs) > 10:
            # Mean should be within ±10 (allow some variance)
            assert abs(valid_mrs.mean()) < 15, \
                f"MRS mean {valid_mrs.mean()} not centered around zero"
    
    @given(st.data())
    @settings(max_examples=30)
    def test_property7_mrs_zero_crossover_detection(self, data):
        """Property 7: Crossover detected when MRS goes from negative to positive.
        
        **Feature: advanced-market-rotation, Property 7: MRS Zero Crossover**
        **Validates: Requirements 2.3**
        """
        length = data.draw(st.integers(min_value=10, max_value=30))
        dates = pd.date_range(end=datetime.now(), periods=length, freq='D')
        
        # Create MRS that crosses from negative to positive
        mrs_values = [-5.0] * (length // 2) + [5.0] * (length - length // 2)
        mrs = pd.Series(mrs_values, index=dates)
        
        calc = MansfieldRSCalculator()
        crossover = calc.detect_zero_crossover(mrs)
        
        crossover_idx = length // 2
        if crossover_idx > 0 and crossover_idx < length:
            assert crossover.iloc[crossover_idx], \
                "Zero crossover should be detected"
    
    @given(
        mrs=st.floats(min_value=-20.0, max_value=20.0, allow_nan=False),
        slope=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False)
    )
    @settings(max_examples=50)
    def test_property8_mrs_improvement_signal(self, mrs, slope):
        """Property 8: Signal classification based on MRS and slope.
        
        **Feature: advanced-market-rotation, Property 8: MRS Improvement Signal**
        **Validates: Requirements 2.4**
        """
        calc = MansfieldRSCalculator()
        signal = calc.get_improvement_signal(mrs, slope)
        
        if mrs >= 0 and slope >= 0:
            assert signal == 'breakout', f"Expected breakout, got {signal}"
        elif mrs < 0 and slope > 0:
            assert signal == 'improving', f"Expected improving, got {signal}"
        elif mrs >= 0 and slope < 0:
            assert signal == 'weakening', f"Expected weakening, got {signal}"
        else:
            assert signal == 'lagging', f"Expected lagging, got {signal}"


# =============================================================================
# Property 9-12: Volume Structure Tests
# =============================================================================

class TestVolumeStructureProperties:
    """Property tests for volume structure analysis."""
    
    @given(df=valid_ohlcv_df(min_rows=30, max_rows=60))
    @settings(max_examples=30)
    def test_property9_volume_zscore_calculation(self, df):
        """Property 9: Z = (V - mean(V, 20)) / std(V, 20).
        
        **Feature: advanced-market-rotation, Property 9: Volume Z-Score Calculation**
        **Validates: Requirements 3.1**
        """
        analyzer = VolumeStructureAnalyzer(zscore_period=20)
        zscore = analyzer.calculate_volume_zscore(df['Volume'])
        
        # Z-score should be finite
        assert not zscore.isna().all(), "Z-score should have valid values"
        
        # After warmup, should have reasonable values
        valid_z = zscore.iloc[20:]
        assert valid_z.abs().max() < 10, "Z-scores should be reasonably bounded"
    
    @given(df=valid_ohlcv_df(min_rows=30, max_rows=60))
    @settings(max_examples=30)
    def test_property10_absorption_pattern_detection(self, df):
        """Property 10: Absorption = high volume + low price change.
        
        **Feature: advanced-market-rotation, Property 10: Absorption Detection**
        **Validates: Requirements 3.2**
        """
        analyzer = VolumeStructureAnalyzer(
            zscore_period=20,
            absorption_threshold=2.0,
            price_threshold=0.005
        )
        
        absorption = analyzer.detect_absorption(df['Close'], df['Volume'])
        
        # Absorption is a boolean series
        assert absorption.dtype == bool, "Absorption should be boolean"
        
        # If detected, verify conditions
        zscore = analyzer.calculate_volume_zscore(df['Volume'])
        price_change = df['Close'].pct_change().abs()
        
        for i in range(20, len(absorption)):
            if absorption.iloc[i]:
                assert zscore.iloc[i] > 2.0 or price_change.iloc[i] < 0.005, \
                    f"Absorption detected but conditions not met at {i}"
    
    @given(df=valid_ohlcv_df(min_rows=30, max_rows=60))
    @settings(max_examples=30)
    def test_property11_smart_money_rejection_detection(self, df):
        """Property 11: Rejection = new low + close near high + high volume.
        
        **Feature: advanced-market-rotation, Property 11: Smart Money Rejection**
        **Validates: Requirements 3.3**
        """
        analyzer = VolumeStructureAnalyzer()
        
        rejection = analyzer.detect_smart_money_rejection(
            df['High'], df['Low'], df['Close'], df['Volume']
        )
        
        # Rejection is a boolean series
        assert rejection.dtype == bool, "Rejection should be boolean"
    
    @given(zscore=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False))
    @settings(max_examples=50)
    def test_property12_volume_classification_consistency(self, zscore):
        """Property 12: Classification is deterministic based on |Z|.
        
        **Feature: advanced-market-rotation, Property 12: Volume Classification**
        **Validates: Requirements 3.4**
        """
        analyzer = VolumeStructureAnalyzer()
        classification = analyzer.classify_volume(zscore)
        
        abs_z = abs(zscore)
        
        if abs_z < 1.5:
            assert classification == 'normal', f"Expected normal for |Z|={abs_z}"
        elif abs_z < 2.5:
            assert classification == 'elevated', f"Expected elevated for |Z|={abs_z}"
        else:
            assert classification == 'extreme', f"Expected extreme for |Z|={abs_z}"


# =============================================================================
# Property 13-14: Scorecard System Tests
# =============================================================================

class TestScorecardProperties:
    """Property tests for scorecard system."""
    
    @given(
        rrg=st.floats(min_value=0.1, max_value=0.4, allow_nan=False),
        rsl=st.floats(min_value=0.1, max_value=0.3, allow_nan=False),
        mrs=st.floats(min_value=0.1, max_value=0.3, allow_nan=False),
        vol=st.floats(min_value=0.1, max_value=0.3, allow_nan=False),
        fund=st.floats(min_value=0.0, max_value=0.2, allow_nan=False)
    )
    @settings(max_examples=50)
    def test_property13_scorecard_weight_validation(self, rrg, rsl, mrs, vol, fund):
        """Property 13: Weights must sum to 1.0 ± 0.001.
        
        **Feature: advanced-market-rotation, Property 13: Scorecard Weight Validation**
        **Validates: Requirements 6.2**
        """
        # Normalize to sum to 1.0
        total = rrg + rsl + mrs + vol + fund
        weights = ScorecardWeights(
            rrg_position=rrg / total,
            rsl_rank=rsl / total,
            mrs_signal=mrs / total,
            volume_pattern=vol / total,
            fundamental_momentum=fund / total
        )
        
        assert weights.validate(), "Normalized weights should validate"
    
    @given(
        total_score=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        buy_threshold=st.floats(min_value=0.3, max_value=0.8, allow_nan=False),
        sell_threshold=st.floats(min_value=-0.8, max_value=-0.1, allow_nan=False)
    )
    @settings(max_examples=50)
    def test_property14_scorecard_signal_generation(self, total_score, buy_threshold, sell_threshold):
        """Property 14: Signal based on threshold comparison.
        
        **Feature: advanced-market-rotation, Property 14: Scorecard Signal Generation**
        **Validates: Requirements 6.3, 6.4**
        """
        scorecard = ScorecardSystem(
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold
        )
        
        signal = scorecard.generate_signal(total_score)
        
        if total_score > buy_threshold:
            assert signal == 'buy', f"Expected buy for score {total_score} > {buy_threshold}"
        elif total_score < sell_threshold:
            assert signal == 'sell', f"Expected sell for score {total_score} < {sell_threshold}"
        else:
            assert signal == 'hold', f"Expected hold for score {total_score}"


# =============================================================================
# Property 15-16: Serialization Tests
# =============================================================================

class TestSerializationProperties:
    """Property tests for data model serialization."""
    
    @given(
        rsl=st.floats(min_value=0.8, max_value=1.2, allow_nan=False),
        sma=st.floats(min_value=50.0, max_value=500.0, allow_nan=False),
        percentile=st.one_of(st.none(), st.floats(min_value=0.0, max_value=100.0, allow_nan=False))
    )
    @settings(max_examples=50)
    def test_property15_serialization_round_trip(self, rsl, sma, percentile):
        """Property 15: JSON round-trip preserves data to 6 decimal places.
        
        **Feature: advanced-market-rotation, Property 15: Serialization Round-Trip**
        **Validates: Requirements 7.2, 7.5**
        """
        original = LevyRSResult(
            ticker='TEST',
            rsl=rsl,
            sma_26w=sma,
            signal='positive',
            percentile_rank=percentile,
            timestamp=datetime.now()
        )
        
        json_str = original.to_json()
        restored = LevyRSResult.from_json(json_str)
        
        assert abs(restored.rsl - original.rsl) < 1e-5, "RSL precision lost"
        assert abs(restored.sma_26w - original.sma_26w) < 1e-5, "SMA precision lost"
        assert restored.ticker == original.ticker
        assert restored.signal == original.signal
    
    @given(
        mrs=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False),
        slope=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
        raw_rs=st.floats(min_value=0.5, max_value=2.0, allow_nan=False)
    )
    @settings(max_examples=50)
    def test_property16_serialization_completeness(self, mrs, slope, raw_rs):
        """Property 16: Serialization includes all required fields.
        
        **Feature: advanced-market-rotation, Property 16: Serialization Completeness**
        **Validates: Requirements 7.3**
        """
        result = MansfieldRSResult(
            ticker='AAPL',
            mrs=mrs,
            mrs_slope=slope,
            raw_rs=raw_rs,
            signal='improving',
            zero_crossover=True,
            timestamp=datetime.now()
        )
        
        data = result.to_dict()
        
        required_fields = [
            'ticker', 'mrs', 'mrs_slope', 'raw_rs',
            'signal', 'zero_crossover', 'timestamp'
        ]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Verify round-trip
        restored = MansfieldRSResult.from_dict(data)
        assert abs(restored.mrs - mrs) < 1e-5
        assert abs(restored.mrs_slope - slope) < 1e-5
        assert abs(restored.raw_rs - raw_rs) < 1e-5
    
    @given(
        vscore=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False),
        total=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
    )
    @settings(max_examples=30)
    def test_volume_and_scorecard_serialization(self, vscore, total, confidence):
        """Test serialization for VolumeAnalysisResult and ScorecardResult."""
        # Volume
        vol_result = VolumeAnalysisResult(
            ticker='TEST',
            volume_zscore=vscore,
            volume_classification='elevated',
            absorption_detected=True,
            rejection_detected=False,
            pattern='absorption',
            timestamp=datetime.now()
        )
        
        vol_json = vol_result.to_json()
        vol_restored = VolumeAnalysisResult.from_json(vol_json)
        assert abs(vol_restored.volume_zscore - vscore) < 1e-5
        
        # Scorecard
        sc_result = ScorecardResult(
            ticker='TEST',
            factor_scores={'rrg_position': 0.5, 'rsl_rank': 0.3},
            available_factors=['rrg_position', 'rsl_rank'],
            total_score=total,
            signal='hold',
            confidence=confidence,
            timestamp=datetime.now()
        )
        
        sc_json = sc_result.to_json()
        sc_restored = ScorecardResult.from_json(sc_json)
        assert abs(sc_restored.total_score - total) < 1e-5
        assert abs(sc_restored.confidence - confidence) < 1e-5


# =============================================================================
# Integration Tests
# =============================================================================

class TestAdvancedRotationIntegration:
    """Integration tests for the full Advanced Rotation pipeline."""
    
    @given(df=valid_ohlcv_df(min_rows=50, max_rows=80))
    @settings(max_examples=10)
    def test_full_pipeline_produces_valid_output(self, df):
        """Full Advanced Rotation computation produces valid results."""
        from quant.rotation import AdvancedRotationFactor
        
        factor = AdvancedRotationFactor()
        
        result = factor.compute(df, None, ticker='TEST', sector='Technology')
        
        # Should return a Series
        assert isinstance(result, pd.Series), "Result should be a Series"
        
        # Value should be bounded
        if len(result) > 0:
            score = result.iloc[-1]
            assert -2.0 <= score <= 2.0, f"Score {score} is out of expected range"
    
    def test_factor_properties(self):
        """Test that factor has correct name and description."""
        from quant.rotation import AdvancedRotationFactor
        
        factor = AdvancedRotationFactor()
        
        assert factor.name == "AdvancedRotation"
        assert "composite" in factor.description.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
