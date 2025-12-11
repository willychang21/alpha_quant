"""Property-Based Tests for Structured Logger.

Tests using Hypothesis to verify logging properties:
- Property 7: Consistent Log Format
- Property 8: Correlation ID in Error Logs
- Property 9: No Emojis in Production Logs

**Feature: code-quality-improvements**
"""

import pytest
import re
import logging
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.structured_logger import (
    StructuredLogger,
    get_structured_logger,
    set_correlation_id,
    get_correlation_id,
    clear_correlation_id,
    strip_emojis,
    EMOJI_PATTERN,
)


@pytest.fixture(autouse=True)
def reset_correlation_id():
    """Clear correlation ID before and after each test."""
    clear_correlation_id()
    yield
    clear_correlation_id()


# =============================================================================
# Unit Tests for Core Functionality
# =============================================================================

class TestStripEmojis:
    """Unit tests for emoji stripping."""
    
    def test_strips_common_emojis(self):
        """Should strip common emoji characters."""
        text = "Hello 👋 World 🌍!"
        result = strip_emojis(text)
        assert result == "Hello  World !"
    
    def test_preserves_plain_text(self):
        """Should preserve plain text without emojis."""
        text = "Hello World"
        result = strip_emojis(text)
        assert result == "Hello World"
    
    def test_strips_multiple_emojis(self):
        """Should strip multiple consecutive emojis."""
        text = "Status: 🎉🎊🥳 Success"
        result = strip_emojis(text)
        assert "🎉" not in result
        assert "🎊" not in result
        assert "Success" in result


class TestCorrelationId:
    """Unit tests for correlation ID context variable."""
    
    def test_set_and_get_correlation_id(self):
        """Should be able to set and get correlation ID."""
        set_correlation_id("test-123")
        assert get_correlation_id() == "test-123"
    
    def test_clear_correlation_id(self):
        """Should be able to clear correlation ID."""
        set_correlation_id("test-123")
        clear_correlation_id()
        assert get_correlation_id() is None
    
    def test_default_is_none(self):
        """Default correlation ID should be None."""
        assert get_correlation_id() is None


# =============================================================================
# Property 7: Consistent Log Format
# =============================================================================
# **Validates: Requirements 7.1**
# For any log message produced by StructuredLogger, the message SHALL match
# the pattern `\[[\w_]+\] .+` (component prefix followed by message).

LOG_PREFIX_PATTERN = re.compile(r'\[[\w_]+\] .+')


class TestConsistentLogFormat:
    """Property tests for consistent log format."""
    
    @given(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N', 'Pc'))))
    @settings(max_examples=50)
    def test_info_messages_have_component_prefix(self, component):
        """All info messages should have [COMPONENT] prefix.
        
        **Property 7: Consistent Log Format**
        **Validates: Requirements 7.1**
        """
        assume(len(component.strip()) > 0)
        assume(re.match(r'^[\w_]+$', component))  # Valid component name
        
        mock_logger = MagicMock()
        logger = StructuredLogger(component, logger=mock_logger, dev_mode=True)
        
        logger.info("Test message")
        
        # Get the formatted message
        call_args = mock_logger.info.call_args
        formatted_message = call_args[0][0]
        
        # Should match pattern [COMPONENT] message
        assert LOG_PREFIX_PATTERN.match(formatted_message), f"Message '{formatted_message}' doesn't match pattern"
        assert formatted_message.startswith(f"[{component}]")
    
    @given(
        st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('L',))),
        st.text(min_size=1, max_size=100)
    )
    @settings(max_examples=30)
    def test_all_log_levels_have_consistent_format(self, component, message):
        """All log levels should use consistent [COMPONENT] prefix.
        
        **Property 7: Consistent Log Format**
        **Validates: Requirements 7.1**
        """
        assume(len(component.strip()) > 0)
        assume(len(message.strip()) > 0)
        
        mock_logger = MagicMock()
        logger = StructuredLogger(component, logger=mock_logger, dev_mode=True)
        
        # Test all log levels
        logger.debug(message)
        logger.info(message)
        logger.warning(message)
        logger.error(message, exc_info=False)
        
        # Verify each call has the prefix
        for method in ['debug', 'info', 'warning', 'error']:
            call_args = getattr(mock_logger, method).call_args
            formatted = call_args[0][0]
            assert f"[{component}]" in formatted, f"{method} message missing prefix"
    
    def test_prefix_format_exact_structure(self):
        """Log prefix should be exactly [COMPONENT] with space before message.
        
        **Property 7: Consistent Log Format**
        **Validates: Requirements 7.1**
        """
        mock_logger = MagicMock()
        logger = StructuredLogger("TestComponent", logger=mock_logger, dev_mode=True)
        
        logger.info("Hello World")
        
        formatted = mock_logger.info.call_args[0][0]
        assert formatted == "[TestComponent] Hello World"


# =============================================================================
# Property 8: Correlation ID in Error Logs
# =============================================================================
# **Validates: Requirements 7.2**
# For any error log message and for any request with a correlation ID set,
# the log message SHALL include the correlation ID.

class TestCorrelationIdInErrorLogs:
    """Property tests for correlation ID in error logs."""
    
    @given(st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N', 'Pd'))))
    @settings(max_examples=30)
    def test_error_logs_include_correlation_id_from_context(self, corr_id):
        """Error logs should include correlation ID from context.
        
        **Property 8: Correlation ID in Error Logs**
        **Validates: Requirements 7.2**
        """
        assume(len(corr_id.strip()) > 0)
        corr_id = corr_id.strip()
        
        mock_logger = MagicMock()
        logger = StructuredLogger("TestComponent", logger=mock_logger, dev_mode=True)
        
        set_correlation_id(corr_id)
        logger.error("Something went wrong", exc_info=False)
        
        formatted = mock_logger.error.call_args[0][0]
        assert corr_id in formatted, f"Correlation ID '{corr_id}' not in message '{formatted}'"
    
    @given(st.text(min_size=5, max_size=30, alphabet=st.characters(whitelist_categories=('L', 'N'))))
    @settings(max_examples=20)
    def test_error_logs_include_explicit_correlation_id(self, corr_id):
        """Error logs should include explicitly passed correlation ID.
        
        **Property 8: Correlation ID in Error Logs**
        **Validates: Requirements 7.2**
        """
        assume(len(corr_id.strip()) > 0)
        corr_id = corr_id.strip()
        
        mock_logger = MagicMock()
        logger = StructuredLogger("TestComponent", logger=mock_logger, dev_mode=True)
        
        # Don't set context correlation ID
        logger.error("Error occurred", exc_info=False, correlation_id=corr_id)
        
        formatted = mock_logger.error.call_args[0][0]
        assert corr_id in formatted
    
    def test_no_correlation_id_when_not_set(self):
        """Error logs should not include placeholder when no correlation ID.
        
        **Property 8: Correlation ID in Error Logs**
        **Validates: Requirements 7.2**
        """
        mock_logger = MagicMock()
        logger = StructuredLogger("TestComponent", logger=mock_logger, dev_mode=True)
        
        # Ensure no correlation ID
        clear_correlation_id()
        logger.error("Error occurred", exc_info=False)
        
        formatted = mock_logger.error.call_args[0][0]
        # Should just have single brackets for component, no double brackets
        assert formatted == "[TestComponent] Error occurred"
    
    def test_correlation_id_format_in_message(self):
        """Correlation ID should be in [COMPONENT][CORR_ID] format.
        
        **Property 8: Correlation ID in Error Logs**
        **Validates: Requirements 7.2**
        """
        mock_logger = MagicMock()
        logger = StructuredLogger("MyService", logger=mock_logger, dev_mode=True)
        
        set_correlation_id("req-12345")
        logger.error("Failed to process", exc_info=False)
        
        formatted = mock_logger.error.call_args[0][0]
        assert "[MyService][req-12345]" in formatted


# =============================================================================
# Property 9: No Emojis in Production Logs
# =============================================================================
# **Validates: Requirements 7.5**
# For any log message produced when dev_mode is False, the message SHALL
# not contain emoji characters.

class TestNoEmojisInProduction:
    """Property tests for emoji-free production logs."""
    
    # Strategy for generating text with emojis
    emoji_chars = st.sampled_from(['🎉', '🚀', '✅', '❌', '⚠️', '📊', '🔥', '💡', '🐛', '⏱️'])
    
    @given(
        st.text(min_size=1, max_size=50),
        st.lists(emoji_chars, min_size=1, max_size=5)
    )
    @settings(max_examples=50)
    def test_emojis_stripped_in_production_mode(self, message, emojis):
        """Emojis should be stripped when dev_mode is False.
        
        **Property 9: No Emojis in Production Logs**
        **Validates: Requirements 7.5**
        """
        # Insert emojis into message
        emoji_str = ''.join(emojis)
        message_with_emojis = f"{emoji_str} {message} {emoji_str}"
        
        mock_logger = MagicMock()
        logger = StructuredLogger("TestComponent", logger=mock_logger, dev_mode=False)
        
        logger.info(message_with_emojis)
        
        formatted = mock_logger.info.call_args[0][0]
        
        # Check no emojis remain
        for emoji in emojis:
            assert emoji not in formatted, f"Emoji '{emoji}' found in production log"
        
        # Verify using regex
        assert not EMOJI_PATTERN.search(formatted), "Emoji pattern found in log"
    
    @given(st.text(min_size=1, max_size=50))
    @settings(max_examples=30)
    def test_emojis_preserved_in_dev_mode(self, message):
        """Emojis should be preserved when dev_mode is True.
        
        **Property 9: No Emojis in Production Logs**
        **Validates: Requirements 7.5**
        """
        message_with_emoji = f"🚀 {message}"
        
        mock_logger = MagicMock()
        logger = StructuredLogger("TestComponent", logger=mock_logger, dev_mode=True)
        
        logger.info(message_with_emoji)
        
        formatted = mock_logger.info.call_args[0][0]
        assert "🚀" in formatted
    
    def test_all_log_levels_strip_emojis_in_production(self):
        """All log levels should strip emojis in production mode.
        
        **Property 9: No Emojis in Production Logs**
        **Validates: Requirements 7.5**
        """
        mock_logger = MagicMock()
        logger = StructuredLogger("TestComponent", logger=mock_logger, dev_mode=False)
        
        logger.debug("🔍 Debug message")
        logger.info("ℹ️ Info message")
        logger.warning("⚠️ Warning message")
        logger.error("❌ Error message", exc_info=False)
        
        for method in ['debug', 'info', 'warning', 'error']:
            formatted = getattr(mock_logger, method).call_args[0][0]
            assert not EMOJI_PATTERN.search(formatted), f"{method} log contains emoji"


# =============================================================================
# Additional Tests
# =============================================================================

class TestStructuredLoggerFactory:
    """Tests for logger factory function."""
    
    def test_get_structured_logger_returns_logger(self):
        """get_structured_logger should return configured logger."""
        logger = get_structured_logger("MyComponent")
        assert isinstance(logger, StructuredLogger)
        assert logger.component == "MyComponent"
    
    @patch('config.quant_config.get_logging_config')
    def test_factory_uses_config(self, mock_get_config):
        """Factory should use LoggingConfig for dev_mode."""
        mock_config = MagicMock()
        mock_config.dev_mode = True
        mock_get_config.return_value = mock_config
        
        logger = StructuredLogger.from_config("TestComponent")
        assert logger._dev_mode is True


class TestMetricLogging:
    """Tests for structured metric logging."""
    
    def test_metric_logs_as_json(self):
        """Metrics should be logged as JSON."""
        mock_logger = MagicMock()
        logger = StructuredLogger("TestComponent", logger=mock_logger, dev_mode=True)
        
        logger.metric("api_latency", 150.5, unit="ms", tags={"endpoint": "/api/v1"})
        
        log_message = mock_logger.info.call_args[0][0]
        assert "METRIC:" in log_message
        assert "api_latency" in log_message
        assert "150.5" in log_message
    
    def test_performance_logs_duration(self):
        """Performance method should log duration metric."""
        mock_logger = MagicMock()
        logger = StructuredLogger("TestComponent", logger=mock_logger, dev_mode=True)
        
        logger.performance("fetch_data", duration_ms=250.0, success=True, ticker="AAPL")
        
        log_message = mock_logger.info.call_args[0][0]
        assert "fetch_data_duration" in log_message
        assert "250.0" in log_message
