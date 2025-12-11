"""Property-Based Tests for Error Handler.

Tests using Hypothesis to verify error handling properties:
- Property 1: Retry on Transient Errors
- Property 2: Graceful Handling of Data Quality Errors

**Feature: code-quality-improvements**
"""

import pytest
import time
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.error_handler import (
    ErrorCategory,
    ErrorHandler,
    with_retry,
    handle_gracefully,
)


# =============================================================================
# Unit Tests for Error Categorization
# =============================================================================

class TestErrorCategorization:
    """Unit tests for error categorization logic."""
    
    def test_connection_error_is_transient(self):
        """ConnectionError should be categorized as TRANSIENT."""
        error = ConnectionError("Network unreachable")
        assert ErrorHandler.categorize(error) == ErrorCategory.TRANSIENT
    
    def test_timeout_error_is_transient(self):
        """TimeoutError should be categorized as TRANSIENT."""
        error = TimeoutError("Request timed out")
        assert ErrorHandler.categorize(error) == ErrorCategory.TRANSIENT
    
    def test_os_error_is_transient(self):
        """OSError should be categorized as TRANSIENT."""
        error = OSError("Temporary failure")
        assert ErrorHandler.categorize(error) == ErrorCategory.TRANSIENT
    
    def test_value_error_is_data_quality(self):
        """ValueError should be categorized as DATA_QUALITY."""
        error = ValueError("Invalid value")
        assert ErrorHandler.categorize(error) == ErrorCategory.DATA_QUALITY
    
    def test_key_error_is_data_quality(self):
        """KeyError should be categorized as DATA_QUALITY."""
        error = KeyError("missing_key")
        assert ErrorHandler.categorize(error) == ErrorCategory.DATA_QUALITY
    
    def test_type_error_is_data_quality(self):
        """TypeError should be categorized as DATA_QUALITY."""
        error = TypeError("Expected int, got str")
        assert ErrorHandler.categorize(error) == ErrorCategory.DATA_QUALITY
    
    def test_unknown_error_is_permanent(self):
        """Unknown errors should be categorized as PERMANENT."""
        error = RuntimeError("Unknown error")
        assert ErrorHandler.categorize(error) == ErrorCategory.PERMANENT
    
    def test_custom_exception_is_permanent(self):
        """Custom exceptions should be categorized as PERMANENT."""
        class CustomError(Exception):
            pass
        error = CustomError("Custom error")
        assert ErrorHandler.categorize(error) == ErrorCategory.PERMANENT


# =============================================================================
# Property 1: Retry on Transient Errors
# =============================================================================
# **Validates: Requirements 1.1, 1.4**
# For any function decorated with @with_retry and for any transient error,
# the function SHALL be retried up to max_retries times with exponential backoff.

class TestRetryOnTransientErrors:
    """Property tests for retry behavior on transient errors."""
    
    @given(st.integers(min_value=1, max_value=5))
    @settings(max_examples=10, deadline=None)
    def test_retries_on_connection_error(self, max_retries):
        """Function should retry max_retries times on ConnectionError.
        
        **Property 1: Retry on Transient Errors**
        **Validates: Requirements 1.1, 1.4**
        """
        # Use mutable list for thread-safe counter (fresh per test)
        call_count = [0]
        
        @with_retry(max_retries=max_retries, base_delay=0.01)
        def failing_function():
            call_count[0] += 1
            raise ConnectionError("Network error")
        
        with pytest.raises(ConnectionError):
            failing_function()
        
        # Should be called max_retries + 1 times (initial + retries)
        assert call_count[0] == max_retries + 1
    
    @given(st.integers(min_value=1, max_value=3))
    @settings(max_examples=5)
    def test_retries_on_timeout_error(self, max_retries):
        """Function should retry max_retries times on TimeoutError.
        
        **Property 1: Retry on Transient Errors**
        **Validates: Requirements 1.1, 1.4**
        """
        call_count = 0
        
        @with_retry(max_retries=max_retries, base_delay=0.01)
        def failing_function():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("Timed out")
        
        with pytest.raises(TimeoutError):
            failing_function()
        
        assert call_count == max_retries + 1
    
    def test_exponential_backoff_timing(self):
        """Retries should use exponential backoff delays.
        
        **Property 1: Retry on Transient Errors**
        **Validates: Requirements 1.1, 1.4**
        """
        call_times = []
        base_delay = 0.05
        
        @with_retry(max_retries=2, base_delay=base_delay)
        def failing_function():
            call_times.append(time.time())
            raise ConnectionError("Network error")
        
        with pytest.raises(ConnectionError):
            failing_function()
        
        assert len(call_times) == 3
        
        # Check delays are approximately exponential
        # First retry: ~base_delay, Second retry: ~base_delay * 2
        first_delay = call_times[1] - call_times[0]
        second_delay = call_times[2] - call_times[1]
        
        # Allow 50% tolerance for timing variations
        assert first_delay >= base_delay * 0.5
        assert second_delay >= base_delay * 1.5  # Should be ~2x first
    
    def test_success_after_retry(self):
        """Function should succeed if retry succeeds.
        
        **Property 1: Retry on Transient Errors**
        **Validates: Requirements 1.1, 1.4**
        """
        call_count = 0
        
        @with_retry(max_retries=3, base_delay=0.01)
        def eventually_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return "success"
        
        result = eventually_succeeds()
        
        assert result == "success"
        assert call_count == 3  # Failed twice, succeeded on third


# =============================================================================
# Property 2: Graceful Handling of Data Quality Errors
# =============================================================================
# **Validates: Requirements 1.2, 1.5**
# For any function processing financial data and for any malformed input
# that raises ValueError or KeyError, the function SHALL return None or
# a default value without propagating the exception.

class TestGracefulDataQualityHandling:
    """Property tests for graceful data quality error handling."""
    
    @given(st.text(min_size=1))
    @settings(max_examples=20)
    def test_returns_default_on_value_error(self, message):
        """Function should return default value on ValueError.
        
        **Property 2: Graceful Handling of Data Quality Errors**
        **Validates: Requirements 1.2, 1.5**
        """
        default_value = {"fallback": True}
        
        @with_retry(max_retries=3, on_data_error=default_value)
        def parse_data():
            raise ValueError(message)
        
        result = parse_data()
        assert result == default_value
    
    @given(st.text(min_size=1))
    @settings(max_examples=20)
    def test_returns_default_on_key_error(self, key_name):
        """Function should return default value on KeyError.
        
        **Property 2: Graceful Handling of Data Quality Errors**
        **Validates: Requirements 1.2, 1.5**
        """
        @with_retry(max_retries=3, on_data_error=None)
        def access_data():
            raise KeyError(key_name)
        
        result = access_data()
        assert result is None
    
    def test_returns_default_on_type_error(self):
        """Function should return default value on TypeError.
        
        **Property 2: Graceful Handling of Data Quality Errors**
        **Validates: Requirements 1.2, 1.5**
        """
        @with_retry(max_retries=3, on_data_error=[])
        def process_data():
            raise TypeError("Expected list, got str")
        
        result = process_data()
        assert result == []
    
    def test_no_retry_on_data_quality_error(self):
        """Data quality errors should not trigger retries.
        
        **Property 2: Graceful Handling of Data Quality Errors**
        **Validates: Requirements 1.2, 1.5**
        """
        call_count = 0
        
        @with_retry(max_retries=5, on_data_error="default")
        def parse_data():
            nonlocal call_count
            call_count += 1
            raise ValueError("Bad data")
        
        result = parse_data()
        
        assert result == "default"
        assert call_count == 1  # Should only be called once, no retries
    
    def test_handle_gracefully_decorator(self):
        """handle_gracefully should catch all errors and return default.
        
        **Property 2: Graceful Handling of Data Quality Errors**
        **Validates: Requirements 1.2, 1.5**
        """
        @handle_gracefully(default=0.0)
        def risky_calculation():
            raise RuntimeError("Unexpected error")
        
        result = risky_calculation()
        assert result == 0.0


# =============================================================================
# Additional Edge Cases
# =============================================================================

class TestEdgeCases:
    """Additional edge case tests."""
    
    def test_permanent_error_not_retried(self):
        """Permanent errors should not be retried."""
        call_count = 0
        
        @with_retry(max_retries=5, base_delay=0.01)
        def permanent_failure():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Permanent failure")
        
        with pytest.raises(RuntimeError):
            permanent_failure()
        
        assert call_count == 1  # No retries
    
    def test_max_delay_cap(self):
        """Delay should be capped at max_delay."""
        @with_retry(max_retries=3, base_delay=100.0, max_delay=0.05)
        def failing_function():
            raise ConnectionError("Error")
        
        start = time.time()
        with pytest.raises(ConnectionError):
            failing_function()
        elapsed = time.time() - start
        
        # With max_delay=0.05, total time should be < 1 second
        assert elapsed < 1.0
    
    def test_preserves_function_metadata(self):
        """Decorator should preserve function name and docstring."""
        @with_retry(max_retries=3)
        def my_function():
            """My docstring."""
            return 42
        
        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."
