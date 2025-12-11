"""Circuit Breaker for External API Resilience.

Implements the circuit breaker pattern to handle transient failures
in external services (e.g., yfinance) gracefully.
"""

import enum
import time
import logging
from typing import Callable, TypeVar, Optional
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(enum.Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking calls
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitOpenError(Exception):
    """Raised when circuit is open and call is rejected."""
    pass


class CircuitBreaker:
    """Circuit breaker for external API resilience.
    
    State machine:
    - CLOSED: Normal operation, failures increment counter
    - OPEN: Reject all calls, wait for recovery_timeout
    - HALF_OPEN: Allow one test call to check recovery
    
    Transitions:
    - CLOSED → OPEN: When failure_count >= failure_threshold
    - OPEN → HALF_OPEN: When recovery_timeout elapses
    - HALF_OPEN → CLOSED: On successful call
    - HALF_OPEN → OPEN: On failed call
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        name: str = "default"
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Consecutive failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            name: Identifier for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
    
    @property
    def state(self) -> CircuitState:
        """Get current state, checking for timeout transition."""
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to(CircuitState.HALF_OPEN)
        return self._state
    
    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result of func
            
        Raises:
            CircuitOpenError: If circuit is open
            Exception: Any exception from func (also recorded as failure)
        """
        state = self.state
        
        if state == CircuitState.OPEN:
            raise CircuitOpenError(f"Circuit {self.name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.CLOSED)
        self._failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)
        elif self._failure_count >= self.failure_threshold:
            self._transition_to(CircuitState.OPEN)
    
    def _should_attempt_reset(self) -> bool:
        """Check if recovery timeout has elapsed."""
        if self._last_failure_time is None:
            return True
        return time.time() - self._last_failure_time >= self.recovery_timeout
    
    def _transition_to(self, new_state: CircuitState):
        """Transition to new state with logging."""
        old_state = self._state
        self._state = new_state
        logger.info(
            f"[CIRCUIT:{self.name}] State transition: "
            f"{old_state.value} -> {new_state.value}"
        )
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
    
    def reset(self):
        """Manually reset circuit to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        logger.info(f"[CIRCUIT:{self.name}] Manual reset to CLOSED")


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    name: str = "default"
):
    """Decorator to wrap function with circuit breaker.
    
    Usage:
        @circuit_breaker(failure_threshold=3, name="yfinance")
        def fetch_data(ticker):
            return yf.Ticker(ticker).history()
    """
    cb = CircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        name=name
    )
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return cb.call(func, *args, **kwargs)
        
        # Attach circuit breaker for inspection
        wrapper.circuit_breaker = cb
        return wrapper
    
    return decorator


# Default circuit breaker for yfinance
_yfinance_circuit_breaker: Optional[CircuitBreaker] = None


def get_yfinance_circuit_breaker() -> CircuitBreaker:
    """Get or create yfinance circuit breaker singleton."""
    global _yfinance_circuit_breaker
    if _yfinance_circuit_breaker is None:
        from config.settings import get_settings
        settings = get_settings()
        failure_threshold = getattr(settings, 'circuit_breaker_failure_threshold', 5)
        recovery_timeout = getattr(settings, 'circuit_breaker_recovery_timeout', 60.0)
        _yfinance_circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            name="yfinance"
        )
    return _yfinance_circuit_breaker
