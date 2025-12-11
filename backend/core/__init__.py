"""Core infrastructure module for DCA Quant Backend.

Provides cross-cutting concerns:
- Error handling with categorization and retry logic
- Rate limiting with token bucket algorithm
- Structured logging with correlation IDs
- Circuit breaker for external API resilience
- Data freshness monitoring
- Metrics collection
"""

from .error_handler import (
    ErrorCategory,
    ErrorHandler,
    with_retry,
    handle_gracefully,
)

from .rate_limiter import (
    RateLimitConfig,
    TokenBucketRateLimiter,
    get_yfinance_rate_limiter,
)

from .structured_logger import (
    StructuredLogger,
    get_structured_logger,
    set_correlation_id,
    get_correlation_id,
    clear_correlation_id,
    strip_emojis,
)

from .circuit_breaker import (
    CircuitState,
    CircuitBreaker,
    CircuitOpenError,
    circuit_breaker,
    get_yfinance_circuit_breaker,
)

from .freshness import (
    DataFreshnessService,
    get_data_freshness_service,
)

from .monitoring import (
    MonitoringService,
    get_monitoring_service,
)

from .metrics import (
    MetricsCollector,
    get_metrics_collector,
)

__all__ = [
    # Error handling
    "ErrorCategory",
    "ErrorHandler",
    "with_retry",
    "handle_gracefully",
    # Rate limiting
    "RateLimitConfig",
    "TokenBucketRateLimiter",
    "get_yfinance_rate_limiter",
    # Structured logging
    "StructuredLogger",
    "get_structured_logger",
    "set_correlation_id",
    "get_correlation_id",
    "clear_correlation_id",
    "strip_emojis",
    # Circuit breaker
    "CircuitState",
    "CircuitBreaker",
    "CircuitOpenError",
    "circuit_breaker",
    "get_yfinance_circuit_breaker",
    # Data freshness
    "DataFreshnessService",
    "get_data_freshness_service",
    # Monitoring
    "MonitoringService",
    "get_monitoring_service",
    # Metrics
    "MetricsCollector",
    "get_metrics_collector",
]
