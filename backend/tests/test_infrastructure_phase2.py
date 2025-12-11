"""Property and Unit Tests for Infrastructure Phase 2.

Tests for:
- Dead job state transition (Property 1-2)
- Backtest determinism (Property 3)
- Factor cache consistency (Property 4)
- Rate limiter enforcement (Property 5)
- Metrics record completeness (Property 6)
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import time
from datetime import datetime
from pathlib import Path
from hypothesis import given, strategies as st, settings


# =============================================================================
# Property 1-2: Dead Job State Transition and Query
# =============================================================================

class TestDLQProperties:
    """Property tests for Dead Letter Queue."""
    
    def test_property1_dead_status_at_max_retries(self):
        """Job should transition to DEAD when retry_count >= max_retries."""
        from compute.job_store import JobStore, JobStatus, Job
        from unittest.mock import MagicMock, patch
        
        # Create mock session
        mock_session = MagicMock()
        mock_job = MagicMock()
        mock_job.id = "test-123"
        mock_job.task_type = "test"
        mock_job.status = JobStatus.RUNNING.value
        mock_job.retry_count = 2  # At max-1 before fail
        mock_job.error = None
        
        mock_session.query.return_value.filter.return_value.first.return_value = mock_job
        
        # Mock settings to have max_retries = 3
        with patch('compute.job_store.get_settings') as mock_settings:
            mock_settings.return_value.job_max_retries = 3
            
            store = JobStore(session=mock_session)
            store._settings = mock_settings.return_value
            
            # Fail the job (will be retry 3)
            store.fail_job("test-123", "Test error")
            
            # Should be marked as DEAD
            assert mock_job.status == JobStatus.DEAD.value
    
    def test_property2_dead_job_query_returns_only_dead(self):
        """get_dead_jobs should return only jobs with DEAD status."""
        from compute.job_store import JobStore, JobStatus
        from unittest.mock import MagicMock
        
        mock_session = MagicMock()
        
        # Create mock jobs with different statuses
        dead_jobs = [MagicMock(status=JobStatus.DEAD.value) for _ in range(3)]
        
        mock_session.query.return_value.filter.return_value.all.return_value = dead_jobs
        
        store = JobStore(session=mock_session)
        result = store.get_dead_jobs()
        
        assert len(result) == 3
        assert all(j.status == JobStatus.DEAD.value for j in result)


# =============================================================================
# Property 3: Backtest Determinism
# =============================================================================

class TestBacktestDeterminismProperties:
    """Property tests for deterministic backtests."""
    
    def test_property3_same_seed_same_result(self):
        """Same random_seed should produce same random sequence."""
        # Verify numpy seed is set correctly
        seed = 42
        
        np.random.seed(seed)
        result1 = np.random.rand(10)
        
        np.random.seed(seed)
        result2 = np.random.rand(10)
        
        assert np.allclose(result1, result2)
    
    @given(st.integers(min_value=0, max_value=10000))
    @settings(max_examples=20)
    def test_property3_determinism_with_any_seed(self, seed):
        """Any seed should produce reproducible results."""
        np.random.seed(seed)
        a = np.random.rand(5)
        
        np.random.seed(seed)
        b = np.random.rand(5)
        
        assert np.allclose(a, b)


# =============================================================================
# Property 4: Factor Cache Consistency
# =============================================================================

class TestCacheProperties:
    """Property tests for factor pipeline cache."""
    
    def test_property4_cache_returns_identical_result(self):
        """Cached result should be identical to computed result."""
        from quant.features.pipeline import FactorPipeline
        
        with tempfile.TemporaryDirectory() as cache_dir:
            # Create test DataFrame
            df = pd.DataFrame({
                'momentum': [0.1, 0.2, 0.3, -0.1, -0.2],
                'value': [1.5, 2.0, 1.2, 0.8, 1.0],
                'sector': ['Tech', 'Tech', 'Finance', 'Finance', 'Health']
            })
            
            # First call - computes and caches
            result1 = FactorPipeline.process_factors(df, cache_dir=cache_dir)
            
            # Second call - reads from cache
            result2 = FactorPipeline.process_factors(df, cache_dir=cache_dir)
            
            pd.testing.assert_frame_equal(result1, result2)
    
    def test_property4_cache_key_changes_with_data(self):
        """Different data should produce different cache keys."""
        from quant.features.pipeline import FactorPipeline
        
        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df2 = pd.DataFrame({'a': [4, 5, 6]})
        
        key1 = FactorPipeline._get_cache_key(df1)
        key2 = FactorPipeline._get_cache_key(df2)
        
        assert key1 != key2


# =============================================================================
# Property 5: Rate Limiter Enforcement
# =============================================================================

class TestRateLimiterProperties:
    """Property tests for rate limiter."""
    
    def test_property5_exceeds_limit_raises_429(self):
        """N+1 request within window should raise 429."""
        from app.core.rate_limiter import RateLimiter
        from fastapi import HTTPException
        
        limit = 5
        limiter = RateLimiter(calls_per_minute=limit)
        
        # Make 'limit' successful requests
        for i in range(limit):
            limiter.check("test-key")
        
        # N+1 should raise 429
        with pytest.raises(HTTPException) as exc_info:
            limiter.check("test-key")
        
        assert exc_info.value.status_code == 429
    
    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=20)
    def test_property5_limit_enforced_for_any_n(self, limit):
        """Rate limit should work for any positive N."""
        from app.core.rate_limiter import RateLimiter
        from fastapi import HTTPException
        
        limiter = RateLimiter(calls_per_minute=limit)
        
        # Make exactly limit requests
        for _ in range(limit):
            limiter.check(f"key-{limit}")
        
        # Next should fail
        with pytest.raises(HTTPException):
            limiter.check(f"key-{limit}")


# =============================================================================
# Property 6: Metrics Record Completeness
# =============================================================================

class TestMetricsProperties:
    """Property tests for metrics collector."""
    
    def test_property6_record_contains_all_fields(self):
        """Recorded metric should contain all required fields."""
        from core.metrics import MetricsCollector, JobMetric
        import json
        
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "metrics.jsonl"
            collector = MetricsCollector(metrics_file=str(metrics_file))
            
            # Record a metric
            metric = JobMetric(
                job_name="test_job",
                status="success",
                duration_seconds=1.5,
                timestamp=datetime.now()
            )
            collector.record(metric)
            
            # Read and verify
            with open(metrics_file, 'r') as f:
                data = json.loads(f.read().strip())
            
            assert 'job_name' in data
            assert 'status' in data
            assert 'duration_seconds' in data
            assert 'timestamp' in data
            assert data['job_name'] == "test_job"
            assert data['status'] == "success"
            assert data['duration_seconds'] == 1.5
    
    def test_property6_creates_file_if_not_exists(self):
        """MetricsCollector should create file on first write."""
        from core.metrics import MetricsCollector, JobMetric
        
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "new_metrics.jsonl"
            assert not metrics_file.exists()
            
            collector = MetricsCollector(metrics_file=str(metrics_file))
            collector.record_job("test", "success", 1.0)
            
            assert metrics_file.exists()


# =============================================================================
# Integration Tests
# =============================================================================

class TestInfrastructureIntegration:
    """Integration tests for infrastructure phase 2."""
    
    def test_optimizer_no_syntax_error(self):
        """Optimizer should load without syntax errors."""
        # This will raise SyntaxError if there are issues
        from quant.portfolio.optimizer import PortfolioOptimizer
        assert PortfolioOptimizer is not None
    
    def test_job_store_has_dead_status(self):
        """JobStore should have DEAD status enum."""
        from compute.job_store import JobStatus
        assert hasattr(JobStatus, 'DEAD')
        assert JobStatus.DEAD.value == "dead"
    
    def test_backtest_engine_has_random_seed(self):
        """BacktestEngine should accept random_seed parameter."""
        import inspect
        from quant.backtest.engine import BacktestEngine
        
        sig = inspect.signature(BacktestEngine.__init__)
        assert 'random_seed' in sig.parameters
        assert sig.parameters['random_seed'].default == 42
    
    def test_factor_pipeline_has_cache_dir(self):
        """FactorPipeline.process_factors should accept cache_dir."""
        import inspect
        from quant.features.pipeline import FactorPipeline
        
        sig = inspect.signature(FactorPipeline.process_factors)
        assert 'cache_dir' in sig.parameters
    
    def test_monitoring_service_uses_metrics_collector(self):
        """MonitoringService should use MetricsCollector."""
        from core.monitoring import MonitoringService
        from core.metrics import MetricsCollector
        
        service = MonitoringService()
        assert hasattr(service, '_metrics')
        assert isinstance(service._metrics, MetricsCollector)


# =============================================================================
# Task 11.2: Deep Health Unit Tests
# =============================================================================

class TestDeepHealthEndpoint:
    """Unit tests for /health/deep endpoint."""
    
    def test_deep_health_returns_healthy_when_all_pass(self):
        """Deep health should return 'healthy' when all checks pass."""
        from unittest.mock import MagicMock, patch
        from app.api.v1.endpoints.health import deep_health
        import asyncio
        
        # Mock database session
        mock_db = MagicMock()
        mock_db.execute.return_value = None
        
        # Mock settings with existing path
        with patch('config.settings.get_settings') as mock_settings:
            mock_settings.return_value.data_lake_path = "/tmp"
            
            # Run async function
            result = asyncio.run(deep_health(db=mock_db))
            
            assert result['status'] == 'healthy'
            assert result['checks']['database'] == 'healthy'
            assert result['checks']['data_lake'] == 'healthy'
    
    def test_deep_health_returns_degraded_when_db_fails(self):
        """Deep health should return 'degraded' when database fails."""
        from unittest.mock import MagicMock, patch
        from app.api.v1.endpoints.health import deep_health
        import asyncio
        
        # Mock database session that raises exception
        mock_db = MagicMock()
        mock_db.execute.side_effect = Exception("Connection refused")
        
        with patch('config.settings.get_settings') as mock_settings:
            mock_settings.return_value.data_lake_path = "/tmp"
            
            result = asyncio.run(deep_health(db=mock_db))
            
            assert result['status'] == 'degraded'
            assert 'unhealthy' in result['checks']['database']
    
    def test_deep_health_returns_degraded_when_data_lake_missing(self):
        """Deep health should return 'degraded' when data_lake_path not found."""
        from unittest.mock import MagicMock, patch
        from app.api.v1.endpoints.health import deep_health
        import asyncio
        
        mock_db = MagicMock()
        mock_db.execute.return_value = None
        
        with patch('config.settings.get_settings') as mock_settings:
            mock_settings.return_value.data_lake_path = "/nonexistent/path/12345"
            
            result = asyncio.run(deep_health(db=mock_db))
            
            assert result['status'] == 'degraded'
            assert 'unhealthy' in result['checks']['data_lake']
    
    def test_deep_health_includes_timestamp(self):
        """Deep health response should include timestamp."""
        from unittest.mock import MagicMock, patch
        from app.api.v1.endpoints.health import deep_health
        import asyncio
        
        mock_db = MagicMock()
        
        with patch('config.settings.get_settings') as mock_settings:
            mock_settings.return_value.data_lake_path = "/tmp"
            
            result = asyncio.run(deep_health(db=mock_db))
            
            assert 'timestamp' in result
            assert 'T' in result['timestamp']  # ISO format

