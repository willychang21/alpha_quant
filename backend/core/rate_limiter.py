"""Token Bucket Rate Limiter for External API Calls.

Provides centralized rate limiting with:
- Token bucket algorithm for smooth rate limiting
- Singleton pattern per API name
- Both async and sync acquire methods
- Configurable rate and burst size
"""

import asyncio
import time
import threading
import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting.
    
    Attributes:
        requests_per_second: Maximum sustained request rate
        burst_size: Maximum number of requests allowed in burst
    """
    requests_per_second: float = 2.0
    burst_size: int = 5


class TokenBucketRateLimiter:
    """Token bucket rate limiter for API calls.
    
    Implements the token bucket algorithm:
    - Tokens are added at a constant rate (requests_per_second)
    - Bucket has a maximum capacity (burst_size)
    - Each request consumes one token
    - If no tokens available, request waits until token is available
    
    Usage:
        limiter = TokenBucketRateLimiter.get_instance("yfinance")
        await limiter.acquire()  # async context
        limiter.acquire_sync()   # sync context
    """
    
    _instances: Dict[str, 'TokenBucketRateLimiter'] = {}
    _instances_lock = threading.Lock()
    
    def __init__(self, name: str, config: Optional[RateLimitConfig] = None):
        """Initialize rate limiter.
        
        Args:
            name: Identifier for this rate limiter (e.g., "yfinance", "default")
            config: Rate limit configuration, uses defaults if not provided
        """
        self.name = name
        self.config = config or RateLimitConfig()
        
        # Token bucket state
        self._tokens = float(self.config.burst_size)
        self._last_update = time.monotonic()
        
        # Locks for thread/async safety
        self._sync_lock = threading.Lock()
        self._async_lock: Optional[asyncio.Lock] = None
        
        logger.debug(
            f"[RATE_LIMITER:{self.name}] Initialized with "
            f"rate={self.config.requests_per_second}/s, "
            f"burst={self.config.burst_size}"
        )
    
    @classmethod
    def get_instance(
        cls, 
        name: str = "default",
        config: Optional[RateLimitConfig] = None, 
    ) -> 'TokenBucketRateLimiter':
        """Get or create a rate limiter instance by name.
        
        Uses singleton pattern to ensure all components share the same
        rate limiter for a given API.
        
        Args:
            name: Identifier for the rate limiter
            config: Configuration (only used on first creation)
            
        Returns:
            TokenBucketRateLimiter instance for the given name
        """
        with cls._instances_lock:
            if name not in cls._instances:
                cls._instances[name] = cls(name, config)
                logger.info(f"[RATE_LIMITER] Created new instance: {name}")
            return cls._instances[name]
    
    @classmethod
    def reset_instances(cls) -> None:
        """Reset all rate limiter instances. Useful for testing."""
        with cls._instances_lock:
            cls._instances.clear()
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        
        # Add tokens based on elapsed time
        new_tokens = elapsed * self.config.requests_per_second
        self._tokens = min(self._tokens + new_tokens, float(self.config.burst_size))
        self._last_update = now
    
    def _wait_time_for_token(self) -> float:
        """Calculate wait time needed to acquire a token.
        
        Returns:
            Wait time in seconds, 0 if token is immediately available
        """
        self._refill_tokens()
        
        if self._tokens >= 1.0:
            return 0.0
        
        # Calculate time until enough tokens are available
        tokens_needed = 1.0 - self._tokens
        wait_time = tokens_needed / self.config.requests_per_second
        return wait_time
    
    def _consume_token(self) -> bool:
        """Attempt to consume a token.
        
        Returns:
            True if token was consumed, False if not available
        """
        self._refill_tokens()
        
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False
    
    async def acquire(self) -> None:
        """Acquire a token asynchronously, waiting if necessary.
        
        This method is safe to call from async contexts and will
        not block the event loop while waiting.
        """
        # Lazy initialization of async lock
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        
        async with self._async_lock:
            wait_time = self._wait_time_for_token()
            
            if wait_time > 0:
                logger.debug(
                    f"[RATE_LIMITER:{self.name}] Request delayed by {wait_time:.2f}s"
                )
                await asyncio.sleep(wait_time)
                self._refill_tokens()
            
            self._tokens -= 1.0
            logger.debug(
                f"[RATE_LIMITER:{self.name}] Token acquired, "
                f"remaining={self._tokens:.1f}"
            )
    
    def acquire_sync(self) -> None:
        """Acquire a token synchronously, blocking if necessary.
        
        This method is safe to call from sync contexts but will
        block the thread while waiting.
        """
        with self._sync_lock:
            wait_time = self._wait_time_for_token()
            
            if wait_time > 0:
                logger.debug(
                    f"[RATE_LIMITER:{self.name}] Request delayed by {wait_time:.2f}s"
                )
                time.sleep(wait_time)
                self._refill_tokens()
            
            self._tokens -= 1.0
            logger.debug(
                f"[RATE_LIMITER:{self.name}] Token acquired, "
                f"remaining={self._tokens:.1f}"
            )
    
    def try_acquire(self) -> bool:
        """Try to acquire a token without waiting.
        
        Returns:
            True if token was acquired, False if rate limited
        """
        with self._sync_lock:
            if self._consume_token():
                logger.debug(
                    f"[RATE_LIMITER:{self.name}] Token acquired, "
                    f"remaining={self._tokens:.1f}"
                )
                return True
            logger.debug(f"[RATE_LIMITER:{self.name}] Rate limited, no tokens available")
            return False
    
    @property
    def available_tokens(self) -> float:
        """Get current number of available tokens."""
        with self._sync_lock:
            self._refill_tokens()
            return self._tokens


# Default yfinance rate limiter
_yfinance_limiter: Optional[TokenBucketRateLimiter] = None


def get_yfinance_rate_limiter() -> TokenBucketRateLimiter:
    """Get or create the yfinance rate limiter singleton.
    
    Uses configuration from environment or defaults.
    """
    global _yfinance_limiter
    if _yfinance_limiter is None:
        try:
            from config.quant_config import get_rate_limit_config
            config = get_rate_limit_config()
            rate_config = RateLimitConfig(
                requests_per_second=config.requests_per_second,
                burst_size=config.burst_size,
            )
        except ImportError:
            rate_config = RateLimitConfig()
        
        _yfinance_limiter = TokenBucketRateLimiter.get_instance("yfinance", rate_config)
    return _yfinance_limiter
