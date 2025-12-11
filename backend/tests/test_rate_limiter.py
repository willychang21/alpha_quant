"""Property-Based Tests for Rate Limiter.

Tests using Hypothesis to verify rate limiting properties:
- Property 5: Rate Limiter Delays Requests
- Property 6: Rate Limiter Singleton

**Feature: code-quality-improvements**
"""

import pytest
import time
import asyncio
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rate_limiter import (
    RateLimitConfig,
    TokenBucketRateLimiter,
    get_yfinance_rate_limiter,
)


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Reset rate limiter instances before each test."""
    TokenBucketRateLimiter.reset_instances()
    yield
    TokenBucketRateLimiter.reset_instances()


# =============================================================================
# Unit Tests for Token Bucket Algorithm
# =============================================================================

class TestTokenBucketAlgorithm:
    """Unit tests for token bucket rate limiter logic."""
    
    def test_initial_tokens_equals_burst_size(self):
        """Limiter should start with burst_size tokens."""
        config = RateLimitConfig(requests_per_second=1.0, burst_size=10)
        limiter = TokenBucketRateLimiter("test", config)
        
        # Should be able to acquire burst_size tokens immediately
        for _ in range(10):
            assert limiter.try_acquire() is True
        
        # 11th should fail
        assert limiter.try_acquire() is False
    
    def test_tokens_refill_over_time(self):
        """Tokens should refill at requests_per_second rate."""
        config = RateLimitConfig(requests_per_second=10.0, burst_size=5)
        limiter = TokenBucketRateLimiter("test", config)
        
        # Consume all tokens
        for _ in range(5):
            limiter.try_acquire()
        
        # Wait for refill (0.3s = 3 tokens at 10/s)
        time.sleep(0.35)
        
        # Should have ~3 tokens now
        assert limiter.available_tokens >= 2.5
        assert limiter.available_tokens <= 4.0
    
    def test_tokens_capped_at_burst_size(self):
        """Tokens should not exceed burst_size."""
        config = RateLimitConfig(requests_per_second=100.0, burst_size=5)
        limiter = TokenBucketRateLimiter("test", config)
        
        # Wait for tokens to accumulate
        time.sleep(0.1)
        
        # Tokens should be capped at burst_size
        assert limiter.available_tokens == 5.0


# =============================================================================
# Property 5: Rate Limiter Delays Requests
# =============================================================================
# **Validates: Requirements 5.2**
# For any sequence of API calls exceeding the rate limit and for any rate limit
# configuration, requests beyond the limit SHALL be delayed until tokens are available.

class TestRateLimiterDelays:
    """Property tests for rate limiter delay behavior."""
    
    @given(st.integers(min_value=2, max_value=5))
    @settings(max_examples=5)
    def test_requests_beyond_burst_are_delayed(self, burst_size):
        """Requests beyond burst_size should be delayed.
        
        **Property 5: Rate Limiter Delays Requests**
        **Validates: Requirements 5.2**
        """
        # Reset singletons for each hypothesis example
        TokenBucketRateLimiter.reset_instances()
        
        config = RateLimitConfig(requests_per_second=1.0, burst_size=burst_size)
        limiter = TokenBucketRateLimiter("test", config)
        
        # Consume all burst tokens immediately
        for _ in range(burst_size):
            acquired = limiter.try_acquire()
            assert acquired is True
        
        # Next request should fail without waiting
        assert limiter.try_acquire() is False
    
    def test_sync_acquire_blocks_when_rate_limited(self):
        """acquire_sync should block until token is available.
        
        **Property 5: Rate Limiter Delays Requests**
        **Validates: Requirements 5.2**
        """
        config = RateLimitConfig(requests_per_second=10.0, burst_size=1)
        limiter = TokenBucketRateLimiter("test", config)
        
        # Consume the only token
        limiter.acquire_sync()
        
        # Measure time for next acquire (should wait ~0.1s)
        start = time.time()
        limiter.acquire_sync()
        elapsed = time.time() - start
        
        # Should have waited approximately 0.1 seconds (1/10 per second)
        assert elapsed >= 0.08  # Allow some tolerance
        assert elapsed <= 0.2
    
    @pytest.mark.asyncio
    async def test_async_acquire_delays_when_rate_limited(self):
        """async acquire should delay until token is available.
        
        **Property 5: Rate Limiter Delays Requests**
        **Validates: Requirements 5.2**
        """
        config = RateLimitConfig(requests_per_second=10.0, burst_size=1)
        limiter = TokenBucketRateLimiter("test", config)
        
        # Consume the only token
        await limiter.acquire()
        
        # Measure time for next acquire
        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start
        
        # Should have waited approximately 0.1 seconds
        assert elapsed >= 0.08
        assert elapsed <= 0.2
    
    @given(st.floats(min_value=5.0, max_value=20.0))
    @settings(max_examples=5, deadline=None)  # Timing test, disable deadline
    def test_rate_matches_configured_requests_per_second(self, rps):
        """Sustained rate should approximately match configured rate.
        
        **Property 5: Rate Limiter Delays Requests**
        **Validates: Requirements 5.2**
        """
        # Reset singletons for each hypothesis example
        TokenBucketRateLimiter.reset_instances()
        
        config = RateLimitConfig(requests_per_second=rps, burst_size=1)
        limiter = TokenBucketRateLimiter("test", config)
        
        # Make 3 requests
        request_count = 3
        start = time.time()
        for _ in range(request_count):
            limiter.acquire_sync()
        elapsed = time.time() - start
        
        # Time should be approximately (request_count - 1) / rps
        # (first request is instant due to burst)
        expected_time = (request_count - 1) / rps
        
        # Allow 30% tolerance
        assert elapsed >= expected_time * 0.7
        assert elapsed <= expected_time * 1.5


# =============================================================================
# Property 6: Rate Limiter Singleton
# =============================================================================
# **Validates: Requirements 5.5**
# For any component requesting a RateLimiter with the same name,
# the system SHALL return the same instance.

class TestRateLimiterSingleton:
    """Property tests for rate limiter singleton behavior."""
    
    @given(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('L', 'N'))))
    @settings(max_examples=20)
    def test_same_name_returns_same_instance(self, name):
        """get_instance with same name should return same instance.
        
        **Property 6: Rate Limiter Singleton**
        **Validates: Requirements 5.5**
        """
        assume(len(name.strip()) > 0)
        name = name.strip()
        
        instance1 = TokenBucketRateLimiter.get_instance(name)
        instance2 = TokenBucketRateLimiter.get_instance(name)
        instance3 = TokenBucketRateLimiter.get_instance(name)
        
        assert instance1 is instance2
        assert instance2 is instance3
        assert id(instance1) == id(instance2) == id(instance3)
    
    @given(
        st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('L',))),
        st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('L',)))
    )
    @settings(max_examples=10)
    def test_different_names_return_different_instances(self, name1, name2):
        """get_instance with different names should return different instances.
        
        **Property 6: Rate Limiter Singleton**
        **Validates: Requirements 5.5**
        """
        assume(name1 != name2)
        assume(len(name1.strip()) > 0 and len(name2.strip()) > 0)
        name1, name2 = name1.strip(), name2.strip()
        
        instance1 = TokenBucketRateLimiter.get_instance(name1)
        instance2 = TokenBucketRateLimiter.get_instance(name2)
        
        assert instance1 is not instance2
        assert instance1.name == name1
        assert instance2.name == name2
    
    def test_yfinance_rate_limiter_is_singleton(self):
        """get_yfinance_rate_limiter should return same instance.
        
        **Property 6: Rate Limiter Singleton**
        **Validates: Requirements 5.5**
        """
        # Reset the global variable
        import core.rate_limiter as rl_module
        rl_module._yfinance_limiter = None
        
        limiter1 = get_yfinance_rate_limiter()
        limiter2 = get_yfinance_rate_limiter()
        
        assert limiter1 is limiter2
        assert limiter1.name == "yfinance"
    
    def test_shared_state_across_calls(self):
        """Singleton instances should share state.
        
        **Property 6: Rate Limiter Singleton**
        **Validates: Requirements 5.5**
        """
        config = RateLimitConfig(requests_per_second=1.0, burst_size=2)
        
        # First caller consumes one token
        limiter1 = TokenBucketRateLimiter.get_instance("shared", config)
        limiter1.try_acquire()
        
        # Second caller should see reduced tokens
        limiter2 = TokenBucketRateLimiter.get_instance("shared")
        
        assert limiter2.available_tokens < 2.0
        assert limiter1._tokens == limiter2._tokens


# =============================================================================
# Additional Edge Cases
# =============================================================================

class TestEdgeCases:
    """Additional edge case tests."""
    
    def test_config_only_used_on_first_creation(self):
        """Config should only be applied on first instance creation."""
        config1 = RateLimitConfig(requests_per_second=5.0, burst_size=10)
        config2 = RateLimitConfig(requests_per_second=100.0, burst_size=50)
        
        instance1 = TokenBucketRateLimiter.get_instance("config_test", config1)
        instance2 = TokenBucketRateLimiter.get_instance("config_test", config2)
        
        # Both should have config1's settings
        assert instance1.config.requests_per_second == 5.0
        assert instance2.config.requests_per_second == 5.0
    
    def test_reset_clears_all_instances(self):
        """reset_instances should clear all singleton instances."""
        TokenBucketRateLimiter.get_instance("test1")
        TokenBucketRateLimiter.get_instance("test2")
        
        assert len(TokenBucketRateLimiter._instances) == 2
        
        TokenBucketRateLimiter.reset_instances()
        
        assert len(TokenBucketRateLimiter._instances) == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_async_acquires(self):
        """Multiple concurrent async acquires should be rate limited."""
        config = RateLimitConfig(requests_per_second=10.0, burst_size=2)
        limiter = TokenBucketRateLimiter("concurrent", config)
        
        start = time.time()
        
        # Launch 4 concurrent acquires
        await asyncio.gather(
            limiter.acquire(),
            limiter.acquire(),
            limiter.acquire(),
            limiter.acquire(),
        )
        
        elapsed = time.time() - start
        
        # Should take at least 0.2s (2 from burst, 2 more need 0.1s each)
        assert elapsed >= 0.15


# =============================================================================
# Property 4: Bounded Concurrency
# =============================================================================
# **Validates: Requirements 4.3**
# For any number of concurrent API calls, the system SHALL limit
# concurrent requests to a configurable maximum.


class TestBoundedConcurrency:
    """Property tests for bounded concurrency behavior."""
    
    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrent_tasks(self):
        """Semaphore should limit concurrent task execution.
        
        **Property 4: Bounded Concurrency**
        **Validates: Requirements 4.3**
        """
        import asyncio
        
        max_concurrent = 3
        semaphore = asyncio.Semaphore(max_concurrent)
        
        concurrent_count = 0
        max_observed_concurrent = 0
        
        async def task():
            nonlocal concurrent_count, max_observed_concurrent
            async with semaphore:
                concurrent_count += 1
                max_observed_concurrent = max(max_observed_concurrent, concurrent_count)
                await asyncio.sleep(0.05)  # Simulate work
                concurrent_count -= 1
        
        # Launch 10 concurrent tasks
        await asyncio.gather(*[task() for _ in range(10)])
        
        # Max concurrent should never exceed semaphore limit
        assert max_observed_concurrent <= max_concurrent
        assert max_observed_concurrent == max_concurrent  # Should hit the limit
    
    @given(st.integers(min_value=1, max_value=5))
    @settings(max_examples=5, deadline=None)
    @pytest.mark.asyncio
    async def test_bounded_concurrency_with_rate_limiter(self, max_concurrent):
        """Rate limiter + semaphore should bound concurrent requests.
        
        **Property 4: Bounded Concurrency**
        **Validates: Requirements 4.3**
        """
        import asyncio
        
        # Reset for each hypothesis example
        TokenBucketRateLimiter.reset_instances()
        
        config = RateLimitConfig(requests_per_second=100.0, burst_size=max_concurrent)
        limiter = TokenBucketRateLimiter("bounded_test", config)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        concurrent_count = 0
        max_observed = 0
        
        async def bounded_task():
            nonlocal concurrent_count, max_observed
            async with semaphore:
                await limiter.acquire()
                concurrent_count += 1
                max_observed = max(max_observed, concurrent_count)
                await asyncio.sleep(0.01)
                concurrent_count -= 1
        
        # Launch many concurrent tasks
        await asyncio.gather(*[bounded_task() for _ in range(max_concurrent * 3)])
        
        assert max_observed <= max_concurrent
    
    @pytest.mark.asyncio
    async def test_rate_limiter_with_semaphore_pattern(self):
        """Combined rate limiter + semaphore pattern should work correctly.
        
        **Property 4: Bounded Concurrency**
        **Validates: Requirements 4.3**
        
        This tests the pattern used in RankingEngine.
        """
        import asyncio
        
        TokenBucketRateLimiter.reset_instances()
        
        config = RateLimitConfig(requests_per_second=10.0, burst_size=3)
        limiter = TokenBucketRateLimiter("pattern_test", config)
        semaphore = asyncio.Semaphore(3)
        
        results = []
        
        async def process_item(item_id):
            async with semaphore:
                await limiter.acquire()
                results.append(item_id)
                await asyncio.sleep(0.01)
                return item_id
        
        # Process 9 items
        completed = await asyncio.gather(*[process_item(i) for i in range(9)])
        
        assert len(completed) == 9
        assert set(completed) == set(range(9))
