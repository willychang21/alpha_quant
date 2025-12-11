"""Property-Based Tests for Quant Configuration.

Tests using Hypothesis to verify configuration properties:
- Property 3: Environment Variable Override

**Feature: code-quality-improvements**
"""

import pytest
import os
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import patch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.quant_config import (
    ValuationConfig,
    OptimizationConfig,
    FactorConfig,
    RateLimitConfig,
    LoggingConfig,
    reset_configs,
    get_valuation_config,
    get_optimization_config,
)


@pytest.fixture(autouse=True)
def reset_config_singletons():
    """Reset config singletons before and after each test."""
    reset_configs()
    yield
    reset_configs()


# =============================================================================
# Property 3: Environment Variable Override
# =============================================================================
# **Validates: Requirements 3.4**
# For any configuration parameter and for any valid environment variable value,
# the configuration SHALL use the environment variable value over the default.


class TestEnvironmentVariableOverride:
    """Property tests for environment variable override behavior."""
    
    @given(st.floats(min_value=0.01, max_value=0.50, allow_nan=False))
    @settings(max_examples=20)
    def test_valuation_wacc_override(self, wacc_value: float):
        """WACC should be overridden by VALUATION_WACC env var.
        
        **Property 3: Environment Variable Override**
        **Validates: Requirements 3.4**
        """
        reset_configs()
        
        with patch.dict(os.environ, {'VALUATION_WACC': str(wacc_value)}):
            config = ValuationConfig()
            assert abs(config.wacc - wacc_value) < 1e-6, \
                f"Expected wacc={wacc_value}, got {config.wacc}"
    
    @given(st.floats(min_value=0.01, max_value=0.30, allow_nan=False))
    @settings(max_examples=20)
    def test_valuation_growth_rate_override(self, growth_value: float):
        """Growth rate should be overridden by VALUATION_GROWTH_RATE env var.
        
        **Property 3: Environment Variable Override**
        **Validates: Requirements 3.4**
        """
        reset_configs()
        
        with patch.dict(os.environ, {'VALUATION_GROWTH_RATE': str(growth_value)}):
            config = ValuationConfig()
            assert abs(config.growth_rate - growth_value) < 1e-6
    
    @given(st.integers(min_value=1, max_value=20))
    @settings(max_examples=20)
    def test_valuation_projection_years_override(self, years: int):
        """Projection years should be overridden by VALUATION_PROJECTION_YEARS env var.
        
        **Property 3: Environment Variable Override**
        **Validates: Requirements 3.4**
        """
        reset_configs()
        
        with patch.dict(os.environ, {'VALUATION_PROJECTION_YEARS': str(years)}):
            config = ValuationConfig()
            assert config.projection_years == years
    
    @given(st.floats(min_value=0.01, max_value=0.50, allow_nan=False))
    @settings(max_examples=20)
    def test_optimization_max_weight_override(self, max_weight: float):
        """Max weight should be overridden by OPTIMIZATION_MAX_WEIGHT env var.
        
        **Property 3: Environment Variable Override**
        **Validates: Requirements 3.4**
        """
        reset_configs()
        
        with patch.dict(os.environ, {'OPTIMIZATION_MAX_WEIGHT': str(max_weight)}):
            config = OptimizationConfig()
            assert abs(config.max_weight - max_weight) < 1e-6
    
    @given(st.integers(min_value=10, max_value=200))
    @settings(max_examples=20)
    def test_optimization_top_n_override(self, top_n: int):
        """Top N should be overridden by OPTIMIZATION_TOP_N env var.
        
        **Property 3: Environment Variable Override**
        **Validates: Requirements 3.4**
        """
        reset_configs()
        
        with patch.dict(os.environ, {'OPTIMIZATION_TOP_N': str(top_n)}):
            config = OptimizationConfig()
            assert config.top_n == top_n
    
    @given(st.floats(min_value=0.5, max_value=10.0, allow_nan=False))
    @settings(max_examples=20)
    def test_rate_limit_rps_override(self, rps: float):
        """Requests per second should be overridden by env var.
        
        **Property 3: Environment Variable Override**
        **Validates: Requirements 3.4**
        """
        reset_configs()
        
        with patch.dict(os.environ, {'RATE_LIMIT_REQUESTS_PER_SECOND': str(rps)}):
            config = RateLimitConfig()
            assert abs(config.requests_per_second - rps) < 1e-6
    
    @given(st.integers(min_value=1, max_value=20))
    @settings(max_examples=20)
    def test_rate_limit_burst_override(self, burst: int):
        """Burst size should be overridden by RATE_LIMIT_BURST_SIZE env var.
        
        **Property 3: Environment Variable Override**
        **Validates: Requirements 3.4**
        """
        reset_configs()
        
        with patch.dict(os.environ, {'RATE_LIMIT_BURST_SIZE': str(burst)}):
            config = RateLimitConfig()
            assert config.burst_size == burst
    
    def test_logging_dev_mode_override_true(self):
        """Dev mode should be True when LOG_DEV_MODE=true.
        
        **Property 3: Environment Variable Override**
        **Validates: Requirements 3.4**
        """
        reset_configs()
        
        for true_value in ['true', 'True', 'TRUE', '1', 'yes']:
            with patch.dict(os.environ, {'LOG_DEV_MODE': true_value}):
                config = LoggingConfig()
                assert config.dev_mode is True, f"Failed for value: {true_value}"
    
    def test_logging_dev_mode_override_false(self):
        """Dev mode should be False when LOG_DEV_MODE=false.
        
        **Property 3: Environment Variable Override**
        **Validates: Requirements 3.4**
        """
        reset_configs()
        
        for false_value in ['false', 'False', 'FALSE', '0', 'no']:
            with patch.dict(os.environ, {'LOG_DEV_MODE': false_value}):
                config = LoggingConfig()
                assert config.dev_mode is False, f"Failed for value: {false_value}"


class TestDefaultValues:
    """Tests for default configuration values."""
    
    def test_valuation_config_defaults(self):
        """ValuationConfig should have sensible defaults."""
        config = ValuationConfig()
        
        assert config.wacc == 0.09
        assert config.growth_rate == 0.05
        assert config.terminal_growth == 0.02
        assert config.projection_years == 5
        assert config.risk_free_rate == 0.04
        assert config.market_risk_premium == 0.05
    
    def test_optimization_config_defaults(self):
        """OptimizationConfig should have sensible defaults."""
        config = OptimizationConfig()
        
        assert config.max_weight == 0.10
        assert config.risk_aversion == 1.0
        assert config.top_n == 50
        assert config.target_vol is None
        assert config.sector_max_weight == 0.30
    
    def test_rate_limit_config_defaults(self):
        """RateLimitConfig should have sensible defaults."""
        config = RateLimitConfig()
        
        assert config.requests_per_second == 2.0
        assert config.burst_size == 5
    
    def test_logging_config_defaults(self):
        """LoggingConfig should have sensible defaults."""
        config = LoggingConfig()
        
        assert config.dev_mode is False


class TestInvalidEnvironmentValues:
    """Tests for handling invalid environment variable values."""
    
    def test_invalid_float_uses_default(self):
        """Invalid float env var should use default value."""
        reset_configs()
        
        with patch.dict(os.environ, {'VALUATION_WACC': 'not_a_number'}):
            config = ValuationConfig()
            assert config.wacc == 0.09  # Default
    
    def test_invalid_int_uses_default(self):
        """Invalid int env var should use default value."""
        reset_configs()
        
        with patch.dict(os.environ, {'OPTIMIZATION_TOP_N': 'fifty'}):
            config = OptimizationConfig()
            assert config.top_n == 50  # Default


class TestSingletonBehavior:
    """Tests for config singleton behavior."""
    
    def test_get_valuation_config_returns_same_instance(self):
        """get_valuation_config should return same instance."""
        config1 = get_valuation_config()
        config2 = get_valuation_config()
        
        assert config1 is config2
    
    def test_get_optimization_config_returns_same_instance(self):
        """get_optimization_config should return same instance."""
        config1 = get_optimization_config()
        config2 = get_optimization_config()
        
        assert config1 is config2
    
    def test_reset_configs_clears_singletons(self):
        """reset_configs should clear all singleton instances."""
        config1 = get_valuation_config()
        reset_configs()
        config2 = get_valuation_config()
        
        assert config1 is not config2
