"""Rate Limiter for API Endpoints.

In-memory rate limiting with sliding window.
"""

from collections import defaultdict
from fastapi import HTTPException
import time
from typing import Dict, List


class RateLimiter:
    """In-memory rate limiter using sliding window algorithm.
    
    Tracks request timestamps per key and enforces calls_per_minute limit.
    """
    
    def __init__(self, calls_per_minute: int = 60):
        """Initialize rate limiter.
        
        Args:
            calls_per_minute: Maximum allowed requests per 60-second window
        """
        self.calls: Dict[str, List[float]] = defaultdict(list)
        self.limit = calls_per_minute
    
    def check(self, key: str) -> None:
        """Check if request is allowed, raise 429 if rate exceeded.
        
        Args:
            key: Unique identifier for rate limiting (e.g., IP, user_id)
        
        Raises:
            HTTPException: 429 Too Many Requests if limit exceeded
        """
        now = time.time()
        
        # Remove timestamps older than 60 seconds
        self.calls[key] = [t for t in self.calls[key] if now - t < 60]
        
        if len(self.calls[key]) >= self.limit:
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded"
            )
        
        self.calls[key].append(now)
    
    def reset(self, key: str) -> None:
        """Reset rate limit for a specific key."""
        if key in self.calls:
            del self.calls[key]
    
    def get_remaining(self, key: str) -> int:
        """Get remaining requests for a key in current window."""
        now = time.time()
        self.calls[key] = [t for t in self.calls[key] if now - t < 60]
        return max(0, self.limit - len(self.calls[key]))


# Default rate limiter instance (60 calls/min for yfinance)
_rate_limiter = None


def get_rate_limiter(calls_per_minute: int = 60) -> RateLimiter:
    """Get or create default rate limiter singleton."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(calls_per_minute=calls_per_minute)
    return _rate_limiter
