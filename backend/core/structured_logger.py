"""Structured Logger with Consistent Formatting.

Provides:
- Consistent [COMPONENT] prefix format
- Correlation ID support for request tracing
- Dev mode for emoji logging (disabled in production)
- Structured JSON output for metrics
"""

import logging
import re
import json
from typing import Optional, Dict, Any
from contextvars import ContextVar

# Context variable for correlation ID (thread/async-safe)
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current context.
    
    Args:
        correlation_id: Unique identifier for request tracing
    """
    correlation_id_var.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get the correlation ID for the current context.
    
    Returns:
        Correlation ID if set, None otherwise
    """
    return correlation_id_var.get()


def clear_correlation_id() -> None:
    """Clear the correlation ID for the current context."""
    correlation_id_var.set(None)


# Regex pattern for emoji detection (common Unicode emoji ranges)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed characters
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended
    "\U00002600-\U000026FF"  # misc symbols
    "]+",
    flags=re.UNICODE
)


def strip_emojis(text: str) -> str:
    """Remove emoji characters from text.
    
    Args:
        text: Input text possibly containing emojis
        
    Returns:
        Text with emojis removed
    """
    return EMOJI_PATTERN.sub('', text).strip()


class StructuredLogger:
    """Logger wrapper with consistent formatting.
    
    Features:
    - Component prefix: [COMPONENT] message
    - Correlation ID in error logs
    - Emoji stripping in production mode
    - Structured JSON for metrics/performance data
    
    Usage:
        logger = StructuredLogger("ValuationEngine")
        logger.info("Processing ticker", ticker="AAPL")
        logger.error("Failed to fetch data", correlation_id="req-123")
    """
    
    PREFIX_FORMAT = "[{component}]"
    
    def __init__(
        self, 
        component: str, 
        logger: Optional[logging.Logger] = None,
        dev_mode: bool = False,
    ):
        """Initialize structured logger.
        
        Args:
            component: Component name for prefix (e.g., "ValuationEngine")
            logger: Custom logger instance, defaults to module logger
            dev_mode: If True, allow emojis in log messages
        """
        self.component = component
        self._logger = logger or logging.getLogger(component)
        self._dev_mode = dev_mode
    
    @classmethod
    def from_config(cls, component: str) -> 'StructuredLogger':
        """Create logger with configuration from environment.
        
        Args:
            component: Component name for prefix
            
        Returns:
            Configured StructuredLogger instance
        """
        try:
            from config.quant_config import get_logging_config
            config = get_logging_config()
            dev_mode = config.dev_mode
        except ImportError:
            dev_mode = False
        
        return cls(component=component, dev_mode=dev_mode)
    
    def _format_message(self, message: str, include_correlation: bool = False) -> str:
        """Format message with component prefix and optional correlation ID.
        
        Args:
            message: The log message
            include_correlation: Whether to include correlation ID
            
        Returns:
            Formatted message string
        """
        # Strip emojis in production mode
        if not self._dev_mode:
            message = strip_emojis(message)
        
        # Build prefix
        prefix = self.PREFIX_FORMAT.format(component=self.component)
        
        # Add correlation ID if requested and available
        if include_correlation:
            corr_id = get_correlation_id()
            if corr_id:
                prefix = f"{prefix}[{corr_id}]"
        
        return f"{prefix} {message}"
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with component prefix.
        
        Args:
            message: Log message
            **kwargs: Additional context (logged as extra)
        """
        formatted = self._format_message(message)
        self._logger.debug(formatted, extra=kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with component prefix.
        
        Args:
            message: Log message
            **kwargs: Additional context (logged as extra)
        """
        formatted = self._format_message(message)
        self._logger.info(formatted, extra=kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with component prefix.
        
        Args:
            message: Log message
            **kwargs: Additional context (logged as extra)
        """
        formatted = self._format_message(message)
        self._logger.warning(formatted, extra=kwargs)
    
    def error(
        self, 
        message: str, 
        exc_info: bool = True,
        correlation_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log error message with correlation ID and stack trace.
        
        Args:
            message: Log message
            exc_info: Whether to include exception info
            correlation_id: Override correlation ID (uses context var if not provided)
            **kwargs: Additional context (logged as extra)
        """
        # Temporarily set correlation ID if provided
        original_corr_id = get_correlation_id()
        if correlation_id:
            set_correlation_id(correlation_id)
        
        try:
            formatted = self._format_message(message, include_correlation=True)
            self._logger.error(formatted, exc_info=exc_info, extra=kwargs)
        finally:
            # Restore original correlation ID
            if correlation_id:
                if original_corr_id:
                    set_correlation_id(original_corr_id)
                else:
                    clear_correlation_id()
    
    def metric(
        self, 
        metric_name: str, 
        value: float, 
        unit: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log a metric in structured JSON format.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement (e.g., "ms", "count")
            tags: Additional tags for the metric
        """
        metric_data: Dict[str, Any] = {
            "component": self.component,
            "metric": metric_name,
            "value": value,
        }
        
        if unit:
            metric_data["unit"] = unit
        
        if tags:
            metric_data["tags"] = tags
        
        corr_id = get_correlation_id()
        if corr_id:
            metric_data["correlation_id"] = corr_id
        
        # Log as JSON for structured parsing
        self._logger.info(f"METRIC: {json.dumps(metric_data)}")
    
    def performance(
        self, 
        operation: str, 
        duration_ms: float,
        success: bool = True,
        **kwargs,
    ) -> None:
        """Log performance data for an operation.
        
        Args:
            operation: Name of the operation
            duration_ms: Duration in milliseconds
            success: Whether the operation succeeded
            **kwargs: Additional context
        """
        self.metric(
            metric_name=f"{operation}_duration",
            value=duration_ms,
            unit="ms",
            tags={"success": str(success).lower(), **{k: str(v) for k, v in kwargs.items()}},
        )


# Convenience function for getting a configured logger
def get_structured_logger(component: str) -> StructuredLogger:
    """Get a structured logger for a component.
    
    Args:
        component: Component name for prefix
        
    Returns:
        Configured StructuredLogger instance
    """
    return StructuredLogger.from_config(component)
