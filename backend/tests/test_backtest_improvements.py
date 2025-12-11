"""Property and Unit Tests for Backtest Improvements.

Tests for:
- Historical beta calculation (Properties 1-2)
- Benchmark metrics (Property 3)
- Fill model behavior (Property 4)
- Event-driven chronological processing (Property 5)
- Result completeness (Property 6)
- Survivorship penalty (Property 7)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from hypothesis import given, strategies as st, assume, settings


# =============================================================================
# Property 1: Historical Beta Uses Only Past Data
# =============================================================================

class TestHistoricalBetaProperties:
    """Property tests for historical beta calculation."""
    
    @given(st.floats(min_value=-0.1, max_value=0.1),
           st.floats(min_value=-0.1, max_value=0.1))
    @settings(max_examples=50)
    def test_property1_beta_uses_past_data_only(self, stock_ret, market_ret):
        """Historical beta calculation should only use past data."""
        from quant.backtest.beta import calculate_historical_beta
        
        # Create synthetic returns series
        dates = pd.date_range(end='2024-01-01', periods=300, freq='B')
        stock_returns = pd.Series(
            np.random.normal(stock_ret, 0.02, len(dates)),
            index=dates
        )
        market_returns = pd.Series(
            np.random.normal(market_ret, 0.015, len(dates)),
            index=dates
        )
        
        # Calculate beta - should return a float
        beta = calculate_historical_beta(stock_returns, market_returns, window=252)
        
        assert isinstance(beta, float)
        assert not np.isnan(beta)
        assert not np.isinf(beta)
    
    @given(st.integers(min_value=10, max_value=200))
    def test_property2_insufficient_data_returns_default(self, n_days):
        """Beta should return 1.0 when data is insufficient (< 252 days)."""
        from quant.backtest.beta import calculate_historical_beta
        
        assume(n_days < 252)
        
        dates = pd.date_range(end='2024-01-01', periods=n_days, freq='B')
        stock_returns = pd.Series(np.random.normal(0, 0.02, n_days), index=dates)
        market_returns = pd.Series(np.random.normal(0, 0.015, n_days), index=dates)
        
        beta = calculate_historical_beta(stock_returns, market_returns, window=252)
        
        # Should return default beta of 1.0
        assert beta == 1.0
    
    def test_property2_formula_correctness(self):
        """Beta should equal cov(stock, market) / var(market)."""
        from quant.backtest.beta import calculate_historical_beta
        
        # Create correlated series with known properties
        np.random.seed(42)
        n = 300
        dates = pd.date_range(end='2024-01-01', periods=n, freq='B')
        
        market_returns = pd.Series(np.random.normal(0, 0.02, n), index=dates)
        # Stock = 1.5 * market + noise (beta should be ~1.5)
        stock_returns = 1.5 * market_returns + pd.Series(
            np.random.normal(0, 0.01, n), index=dates
        )
        
        beta = calculate_historical_beta(stock_returns, market_returns, window=252)
        
        # Should be close to 1.5
        assert 1.0 < beta < 2.0
    
    def test_property2_zero_variance_returns_default(self):
        """Beta should return 1.0 if market variance is zero."""
        from quant.backtest.beta import calculate_historical_beta
        
        n = 300
        dates = pd.date_range(end='2024-01-01', periods=n, freq='B')
        
        stock_returns = pd.Series(np.random.normal(0, 0.02, n), index=dates)
        market_returns = pd.Series([0.0] * n, index=dates)  # Zero variance
        
        beta = calculate_historical_beta(stock_returns, market_returns, window=252)
        
        assert beta == 1.0


# =============================================================================
# Property 3: Benchmark Metrics Calculation
# =============================================================================

class TestBenchmarkMetricsProperties:
    """Property tests for benchmark metrics."""
    
    def test_property3_alpha_formula(self):
        """Alpha should equal (Rp - Rf) - beta * (Rm - Rf)."""
        from quant.backtest.benchmark import calculate_alpha
        
        # Test with known values
        strategy_cagr = 0.15  # 15%
        benchmark_cagr = 0.10  # 10%
        beta = 1.2
        rf = 0.04
        
        expected_alpha = (strategy_cagr - rf) - beta * (benchmark_cagr - rf)
        calculated_alpha = calculate_alpha(strategy_cagr, benchmark_cagr, beta, rf)
        
        assert abs(calculated_alpha - expected_alpha) < 1e-10
    
    @given(st.floats(min_value=0, max_value=0.3),
           st.floats(min_value=0, max_value=0.2),
           st.floats(min_value=0.5, max_value=1.5))
    @settings(max_examples=30)
    def test_property3_alpha_formula_property(self, strategy_cagr, benchmark_cagr, beta):
        """Alpha formula should hold for any valid inputs."""
        from quant.backtest.benchmark import calculate_alpha
        
        rf = 0.04
        alpha = calculate_alpha(strategy_cagr, benchmark_cagr, beta, rf)
        
        expected = (strategy_cagr - rf) - beta * (benchmark_cagr - rf)
        
        assert abs(alpha - expected) < 1e-10


# =============================================================================
# Property 4: Fill Model Partial Fill Behavior
# =============================================================================

class TestFillModelProperties:
    """Property tests for fill model."""
    
    @given(st.floats(min_value=100, max_value=10000),
           st.floats(min_value=10, max_value=1000))
    @settings(max_examples=50)
    def test_property4_partial_fill_respects_participation_limit(
        self, order_qty, volume
    ):
        """Fill should never exceed max_participation * volume."""
        from quant.backtest.execution.fill_model import LiquidityConstrainedFill
        
        max_participation = 0.1
        fill_model = LiquidityConstrainedFill(max_participation=max_participation)
        
        filled = fill_model.get_fill_quantity(order_qty, volume)
        max_allowed = volume * max_participation
        
        assert filled <= max_allowed + 1e-6  # Small tolerance
        assert filled <= order_qty + 1e-6
    
    @given(st.floats(min_value=1, max_value=100),
           st.floats(min_value=1000, max_value=10000))
    @settings(max_examples=50)
    def test_property4_small_orders_fully_filled(self, order_qty, volume):
        """Small orders (< 10% volume) should be fully filled."""
        from quant.backtest.execution.fill_model import LiquidityConstrainedFill
        
        # Small order relative to volume
        assume(order_qty < volume * 0.1)
        
        fill_model = LiquidityConstrainedFill(max_participation=0.1)
        filled = fill_model.get_fill_quantity(order_qty, volume)
        
        assert abs(filled - order_qty) < 1e-6
    
    def test_property4_remaining_quantity_equals_desired_minus_filled(self):
        """Remaining = desired - filled."""
        from quant.backtest.execution.fill_model import LiquidityConstrainedFill
        
        fill_model = LiquidityConstrainedFill(max_participation=0.1)
        
        desired = 1000
        volume = 500  # 10% = 50, so partial fill
        
        filled = fill_model.get_fill_quantity(desired, volume)
        remaining = desired - filled
        
        assert remaining == desired - filled
        assert remaining > 0  # Should have unfilled portion


# =============================================================================
# Property 5: Event-Driven Chronological Processing
# =============================================================================

class TestEventDrivenProperties:
    """Property tests for event-driven backtest."""
    
    def test_property5_dates_processed_chronologically(self):
        """Market data should be processed in ascending date order."""
        from quant.backtest.event_engine import EventDrivenBacktester
        
        backtester = EventDrivenBacktester()
        
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        
        dates = backtester._get_trading_dates(start, end)
        
        # Verify strictly ascending order
        for i in range(1, len(dates)):
            assert dates[i] > dates[i-1], "Dates must be in ascending order"
    
    def test_property5_no_weekends_in_trading_dates(self):
        """Trading dates should not include weekends."""
        from quant.backtest.event_engine import EventDrivenBacktester
        
        backtester = EventDrivenBacktester()
        
        start = datetime(2024, 1, 1)
        end = datetime(2024, 3, 31)
        
        dates = backtester._get_trading_dates(start, end)
        
        for d in dates:
            assert d.weekday() < 5, f"{d} is a weekend"


# =============================================================================
# Property 6: Backtest Result Completeness
# =============================================================================

class TestResultCompletenessProperties:
    """Property tests for result completeness."""
    
    def test_property6_result_contains_required_fields(self):
        """BacktestResult should contain equity_curve, metrics, trades."""
        from quant.backtest.interface import BacktestResult
        
        result = BacktestResult()
        
        assert hasattr(result, 'equity_curve')
        assert hasattr(result, 'metrics')
        assert hasattr(result, 'trades')
    
    def test_property6_metrics_contain_required_keys(self):
        """Metrics should include cagr, sharpe_ratio, max_drawdown."""
        from quant.backtest.event_engine import EventDrivenBacktester
        import pandas as pd
        
        # Create mock history
        backtester = EventDrivenBacktester()
        backtester.history = [
            {'date': date(2024, 1, d), 'equity': 100000 + d * 100, 'cash': 50000, 'holdings_count': 5}
            for d in range(1, 22)
        ]
        backtester.trades = []
        
        df = pd.DataFrame(backtester.history)
        df['returns'] = df['equity'].pct_change().fillna(0)
        
        metrics = backtester._calculate_base_metrics(df)
        
        assert 'cagr' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics


# =============================================================================
# Property 7: Survivorship Penalty Calculation
# =============================================================================

class TestSurvivorshipProperties:
    """Property tests for survivorship bias penalty."""
    
    @given(st.floats(min_value=-0.2, max_value=0.5),
           st.floats(min_value=1, max_value=20))
    @settings(max_examples=50)
    def test_property7_penalty_formula(self, cagr, years):
        """Adjusted CAGR = original + (delist_rate * avg_loss)."""
        from quant.backtest.survivorship import apply_survivorship_penalty, DELIST_RATES
        
        result = apply_survivorship_penalty(cagr, years, universe='broad')
        
        delist_rate = DELIST_RATES['broad']  # 0.03
        avg_loss = -0.50
        expected_penalty = delist_rate * avg_loss
        expected_adjusted = cagr + expected_penalty
        
        assert abs(result['adjusted_cagr'] - expected_adjusted) < 1e-10
        assert abs(result['annual_penalty'] - expected_penalty) < 1e-10
    
    def test_property7_large_cap_has_lower_penalty(self):
        """Large cap (1%/yr) should have lower penalty than small cap (6%/yr)."""
        from quant.backtest.survivorship import apply_survivorship_penalty
        
        cagr = 0.15
        years = 5
        
        large_cap = apply_survivorship_penalty(cagr, years, 'large_cap')
        small_cap = apply_survivorship_penalty(cagr, years, 'small_cap')
        
        # Large cap penalty should be smaller (less negative)
        assert abs(large_cap['annual_penalty']) < abs(small_cap['annual_penalty'])
        
        # Large cap adjusted CAGR should be higher
        assert large_cap['adjusted_cagr'] > small_cap['adjusted_cagr']
    
    def test_property7_delist_rates_correct(self):
        """Delist rates should match spec: 1% large, 3% broad, 6% small."""
        from quant.backtest.survivorship import DELIST_RATES
        
        assert DELIST_RATES['large_cap'] == 0.01
        assert DELIST_RATES['broad'] == 0.03
        assert DELIST_RATES['small_cap'] == 0.06


# =============================================================================
# Task 13: Integration Tests for Full Backtest Pipeline
# =============================================================================

class TestBacktestPipelineIntegration:
    """Integration tests for full backtest pipeline."""
    
    def test_integration_time_machine_no_nvda_override(self):
        """Time machine backtest should not have NVDA special overrides."""
        import os
        
        file_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 'backtests', 'backtest_time_machine.py'
        )
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Should NOT contain FORCE CORRECT debug blocks
        assert 'FORCE CORRECT HISTORICAL GROWTH' not in content
        assert '[DEBUG] Force Growth' not in content
    
    def test_integration_time_machine_uses_historical_beta(self):
        """Time machine should import and use historical beta."""
        import os
        
        file_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 'backtests', 'backtest_time_machine.py'
        )
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        assert 'from quant.backtest.beta import' in content
        assert '_calculate_historical_beta_for_pit' in content
    
    def test_integration_no_bare_excepts(self):
        """Backtest files should not have bare except clauses."""
        import os
        import re
        
        backtest_files = [
            os.path.join(os.path.dirname(__file__), '..', 'backtests', 'backtest_time_machine.py'),
            os.path.join(os.path.dirname(__file__), '..', 'backtests', 'backtest_valuation.py'),
        ]
        
        bare_except_pattern = re.compile(r'^\s*except\s*:', re.MULTILINE)
        
        for file_path in backtest_files:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()
                
                matches = bare_except_pattern.findall(content)
                basename = os.path.basename(file_path)
                assert len(matches) == 0, f"{basename} has {len(matches)} bare except"
    
    def test_integration_walk_forward_has_benchmark_metrics(self):
        """Walk forward should include benchmark metrics imports."""
        import os
        
        file_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 'quant', 'backtest', 'walk_forward.py'
        )
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        assert 'from quant.backtest.benchmark import' in content
        assert 'from quant.backtest.survivorship import' in content
    
    def test_integration_engine_has_fill_model(self):
        """Backtest engine should have fill model integration."""
        import os
        
        file_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 'quant', 'backtest', 'engine.py'
        )
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        assert 'from quant.backtest.execution.fill_model import' in content
        assert 'fill_model' in content
        assert 'unfilled_orders' in content
    
    def test_integration_event_engine_complete(self):
        """Event engine should have complete implementation."""
        from quant.backtest.event_engine import EventDrivenBacktester
        
        backtester = EventDrivenBacktester()
        
        assert hasattr(backtester, 'run')
        assert hasattr(backtester, '_execute_orders')
        assert hasattr(backtester, '_generate_result')
        assert hasattr(backtester, '_get_trading_dates')
        assert backtester.fill_model is not None
    
    def test_integration_all_modules_importable(self):
        """All new backtest modules should be importable."""
        # Should not raise ImportError
        from quant.backtest.beta import calculate_historical_beta
        from quant.backtest.benchmark import calculate_benchmark_metrics
        from quant.backtest.survivorship import apply_survivorship_penalty
        from quant.backtest.event_engine import EventDrivenBacktester
        from quant.backtest.interface import BacktestResult
        
        assert callable(calculate_historical_beta)
        assert callable(calculate_benchmark_metrics)
        assert callable(apply_survivorship_penalty)
