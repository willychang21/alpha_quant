"""Property-Based Tests for Configuration.

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
    get_factor_config,
    get_rate_limit_config,
    get_logging_config,
)


@pytest.fixture(autouse=True)
def reset_config_singletons():
    """Reset config singletons before and after each test."""
    reset_configs()
    yield
    reset_configs()


@pytest.fixture
def clean_env():
    """Fixture to save and restore environment variables."""
    env_vars = [
        'VALUATION_WACC',
        'VALUATION_GROWTH_RATE',
        'VALUATION_TERMINAL_GROWTH',
        'VALUATION_PROJECTION_YEARS',
        'VALUATION_RISK_FREE_RATE',
        'VALUATION_MARKET_RISK_PREMIUM',
        'OPTIMIZATION_MAX_WEIGHT',
        'OPTIMIZATION_RISK_AVERSION',
        'OPTIMIZATION_TOP_N',
        'OPTIMIZATION_TARGET_VOL',
        'OPTIMIZATION_SECTOR_MAX_WEIGHT',
        'RATE_LIMIT_REQUESTS_PER_SECOND',
        'RATE_LIMIT_BURST_SIZE',
        'LOG_DEV_MODE',
    ]
    
    # Save original values
    original = {var: os.environ.get(var) for var in env_vars}
    
    # Clear all env vars
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]
    
    yield
    
    # Restore original values
    for var, value in original.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]


# =============================================================================
# Unit Tests for Default Values
# =============================================================================

class TestDefaultValues:
    """Unit tests for configuration default values."""
    
    def test_valuation_config_defaults(self, clean_env):
        """ValuationConfig should have correct defaults."""
        config = ValuationConfig()
        
        assert config.wacc == 0.09
        assert config.growth_rate == 0.05
        assert config.terminal_growth == 0.02
        assert config.projection_years == 5
        assert config.risk_free_rate == 0.04
        assert config.market_risk_premium == 0.05
    
    def test_optimization_config_defaults(self, clean_env):
        """OptimizationConfig should have correct defaults."""
        config = OptimizationConfig()
        
        assert config.max_weight == 0.10
        assert config.risk_aversion == 1.0
        assert config.top_n == 50
        assert config.target_vol is None
        assert config.sector_max_weight == 0.30
    
    def test_factor_config_defaults(self, clean_env):
        """FactorConfig should have correct defaults."""
        config = FactorConfig()
        
        assert "vsm" in config.bull_weights
        assert "qmj" in config.bear_weights
        assert config.bull_weights["vsm"] == 0.25
        assert config.bear_weights["qmj"] == 0.30
    
    def test_rate_limit_config_defaults(self, clean_env):
        """RateLimitConfig should have correct defaults."""
        config = RateLimitConfig()
        
        assert config.requests_per_second == 2.0
        assert config.burst_size == 5
    
    def test_logging_config_defaults(self, clean_env):
        """LoggingConfig should have correct defaults."""
        config = LoggingConfig()
        
        assert config.dev_mode is False


# =============================================================================
# Property 3: Environment Variable Override
# =============================================================================
# **Validates: Requirements 3.4**
# For any configuration key in Settings and for any corresponding environment
# variable, setting the environment variable SHALL override the default value.

class TestEnvironmentVariableOverride:
    """Property tests for environment variable override."""
    
    def _clear_env(self):
        """Helper to clear relevant env vars."""
        for var in ['VALUATION_WACC', 'VALUATION_GROWTH_RATE', 'VALUATION_PROJECTION_YEARS',
                    'OPTIMIZATION_MAX_WEIGHT', 'OPTIMIZATION_TOP_N', 
                    'RATE_LIMIT_REQUESTS_PER_SECOND', 'RATE_LIMIT_BURST_SIZE', 'LOG_DEV_MODE']:
            if var in os.environ:
                del os.environ[var]
    
    @given(st.floats(min_value=0.01, max_value=0.50, allow_nan=False))
    @settings(max_examples=20)
    def test_valuation_wacc_env_override(self, wacc_value):
        """VALUATION_WACC should override default wacc.
        
        **Property 3: Environment Variable Override**
        **Validates: Requirements 3.4**
        """
        self._clear_env()
        os.environ['VALUATION_WACC'] = str(wacc_value)
        try:
            config = ValuationConfig()
            assert abs(config.wacc - wacc_value) < 0.0001
        finally:
            self._clear_env()
    
    @given(st.floats(min_value=0.01, max_value=0.30, allow_nan=False))
    @settings(max_examples=20)
    def test_valuation_growth_rate_env_override(self, growth_value):
        """VALUATION_GROWTH_RATE should override default growth_rate.
        
        **Property 3: Environment Variable Override**
        **Validates: Requirements 3.4**
        """
        self._clear_env()
        os.environ['VALUATION_GROWTH_RATE'] = str(growth_value)
        try:
            config = ValuationConfig()
            assert abs(config.growth_rate - growth_value) < 0.0001
        finally:
            self._clear_env()
    
    @given(st.integers(min_value=1, max_value=20))
    @settings(max_examples=15)
    def test_valuation_projection_years_env_override(self, years):
        """VALUATION_PROJECTION_YEARS should override default projection_years.
        
        **Property 3: Environment Variable Override**
        **Validates: Requirements 3.4**
        """
        self._clear_env()
        os.environ['VALUATION_PROJECTION_YEARS'] = str(years)
        try:
            config = ValuationConfig()
            assert config.projection_years == years
        finally:
            self._clear_env()
    
    @given(st.floats(min_value=0.01, max_value=0.50, allow_nan=False))
    @settings(max_examples=20)
    def test_optimization_max_weight_env_override(self, weight):
        """OPTIMIZATION_MAX_WEIGHT should override default max_weight.
        
        **Property 3: Environment Variable Override**
        **Validates: Requirements 3.4**
        """
        self._clear_env()
        os.environ['OPTIMIZATION_MAX_WEIGHT'] = str(weight)
        try:
            config = OptimizationConfig()
            assert abs(config.max_weight - weight) < 0.0001
        finally:
            self._clear_env()
    
    @given(st.integers(min_value=10, max_value=200))
    @settings(max_examples=15)
    def test_optimization_top_n_env_override(self, top_n):
        """OPTIMIZATION_TOP_N should override default top_n.
        
        **Property 3: Environment Variable Override**
        **Validates: Requirements 3.4**
        """
        self._clear_env()
        os.environ['OPTIMIZATION_TOP_N'] = str(top_n)
        try:
            config = OptimizationConfig()
            assert config.top_n == top_n
        finally:
            self._clear_env()
    
    @given(st.floats(min_value=0.5, max_value=50.0, allow_nan=False))
    @settings(max_examples=20)
    def test_rate_limit_rps_env_override(self, rps):
        """RATE_LIMIT_REQUESTS_PER_SECOND should override default.
        
        **Property 3: Environment Variable Override**
        **Validates: Requirements 3.4**
        """
        self._clear_env()
        os.environ['RATE_LIMIT_REQUESTS_PER_SECOND'] = str(rps)
        try:
            config = RateLimitConfig()
            assert abs(config.requests_per_second - rps) < 0.0001
        finally:
            self._clear_env()
    
    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=15)
    def test_rate_limit_burst_env_override(self, burst):
        """RATE_LIMIT_BURST_SIZE should override default.
        
        **Property 3: Environment Variable Override**
        **Validates: Requirements 3.4**
        """
        self._clear_env()
        os.environ['RATE_LIMIT_BURST_SIZE'] = str(burst)
        try:
            config = RateLimitConfig()
            assert config.burst_size == burst
        finally:
            self._clear_env()
    
    @given(st.sampled_from(['true', 'True', 'TRUE', '1', 'yes']))
    @settings(max_examples=5)
    def test_logging_dev_mode_true_values(self, true_value):
        """LOG_DEV_MODE should accept various true values.
        
        **Property 3: Environment Variable Override**
        **Validates: Requirements 3.4**
        """
        self._clear_env()
        os.environ['LOG_DEV_MODE'] = true_value
        try:
            config = LoggingConfig()
            assert config.dev_mode is True
        finally:
            self._clear_env()
    
    @given(st.sampled_from(['false', 'False', 'FALSE', '0', 'no']))
    @settings(max_examples=5)
    def test_logging_dev_mode_false_values(self, false_value):
        """LOG_DEV_MODE should accept various false values.
        
        **Property 3: Environment Variable Override**
        **Validates: Requirements 3.4**
        """
        self._clear_env()
        os.environ['LOG_DEV_MODE'] = false_value
        try:
            config = LoggingConfig()
            assert config.dev_mode is False
        finally:
            self._clear_env()



# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEnvironmentVariableErrors:
    """Tests for invalid environment variable handling."""
    
    def test_invalid_float_env_value_uses_default(self, clean_env):
        """Invalid float env value should not crash, uses default."""
        os.environ['VALUATION_WACC'] = 'not_a_number'
        config = ValuationConfig()
        
        # Should use default
        assert config.wacc == 0.09
    
    def test_invalid_int_env_value_uses_default(self, clean_env):
        """Invalid int env value should not crash, uses default."""
        os.environ['OPTIMIZATION_TOP_N'] = '3.14'  # Float when int expected
        config = OptimizationConfig()
        
        # Should use default (int('3.14') raises ValueError)
        assert config.top_n == 50
    
    def test_empty_env_value_uses_default(self, clean_env):
        """Empty env value should use default."""
        os.environ['VALUATION_WACC'] = ''
        config = ValuationConfig()
        
        # Empty string to float raises ValueError, use default
        assert config.wacc == 0.09


# =============================================================================
# Singleton Behavior Tests
# =============================================================================

class TestConfigSingletons:
    """Tests for config singleton pattern."""
    
    def test_get_valuation_config_returns_same_instance(self, clean_env):
        """get_valuation_config should return same instance."""
        config1 = get_valuation_config()
        config2 = get_valuation_config()
        
        assert config1 is config2
    
    def test_get_optimization_config_returns_same_instance(self, clean_env):
        """get_optimization_config should return same instance."""
        config1 = get_optimization_config()
        config2 = get_optimization_config()
        
        assert config1 is config2
    
    def test_reset_clears_singletons(self, clean_env):
        """reset_configs should clear all singleton instances."""
        config1 = get_valuation_config()
        reset_configs()
        config2 = get_valuation_config()
        
        assert config1 is not config2


# =============================================================================
# FactorConfig Specific Tests
# =============================================================================

class TestFactorConfig:
    """Tests for FactorConfig functionality."""
    
    def test_get_weights_bull_market(self, clean_env):
        """get_weights should return bull weights for bull market."""
        config = FactorConfig()
        weights = config.get_weights(is_bull_market=True)
        
        assert weights == config.bull_weights
        assert weights["vsm"] == 0.25
    
    def test_get_weights_bear_market(self, clean_env):
        """get_weights should return bear weights for bear market."""
        config = FactorConfig()
        weights = config.get_weights(is_bull_market=False)
        
        assert weights == config.bear_weights
        assert weights["qmj"] == 0.30
    
    def test_weights_sum_approximately_to_one(self, clean_env):
        """Factor weights should sum approximately to 1.0."""
        config = FactorConfig()
        
        bull_sum = sum(config.bull_weights.values())
        bear_sum = sum(config.bear_weights.values())
        
        # Allow small tolerance for floating point
        assert 0.9 <= bull_sum <= 1.1
        assert 0.9 <= bear_sum <= 1.1
