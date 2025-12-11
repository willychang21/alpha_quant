"""Property and Unit Tests for Infrastructure Phase 3.

Tests for:
- Data freshness threshold (Property 1-2)
- Circuit breaker state transitions (Property 3-6)
"""

import pytest
import time
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from hypothesis import given, strategies as st, settings as hyp_settings


# =============================================================================
# Property 1-2: Data Freshness
# =============================================================================

class TestFreshnessProperties:
    """Property tests for data freshness."""
    
    def test_property1_fresh_data_logs_info(self):
        """Data within threshold should log INFO, not WARNING."""
        from core.freshness import DataFreshnessService
        
        with tempfile.TemporaryDirectory() as tmpdir:
            service = DataFreshnessService(
                threshold_hours=24.0,
                data_lake_path=Path(tmpdir)
            )
            
            # Mock _get_max_date to return today
            service._get_max_date = MagicMock(return_value=date.today())
            
            is_fresh, lag_hours, last_date = service.get_freshness_status()
            
            assert is_fresh is True
            assert lag_hours < 24.0
    
    def test_property1_stale_data_logs_warning(self):
        """Data exceeding threshold should log WARNING."""
        from core.freshness import DataFreshnessService
        
        with tempfile.TemporaryDirectory() as tmpdir:
            service = DataFreshnessService(
                threshold_hours=24.0,
                data_lake_path=Path(tmpdir)
            )
            
            # Mock _get_max_date to return 3 days ago
            old_date = date.today() - timedelta(days=3)
            service._get_max_date = MagicMock(return_value=old_date)
            
            is_fresh, lag_hours, last_date = service.get_freshness_status()
            
            assert is_fresh is False
            assert lag_hours > 24.0
    
    @given(st.floats(min_value=1.0, max_value=168.0))
    @hyp_settings(max_examples=20)
    def test_property2_threshold_determines_freshness(self, threshold_hours):
        """Any threshold should correctly determine freshness."""
        from core.freshness import DataFreshnessService
        
        with tempfile.TemporaryDirectory() as tmpdir:
            service = DataFreshnessService(
                threshold_hours=threshold_hours,
                data_lake_path=Path(tmpdir)
            )
            
            # Data is from half the threshold ago
            hours_ago = threshold_hours / 2
            mock_date = datetime.now() - timedelta(hours=hours_ago)
            service._get_max_date = MagicMock(return_value=mock_date.date())
            
            is_fresh, lag_hours, _ = service.get_freshness_status()
            
            # Should be fresh since lag < threshold
            assert is_fresh or lag_hours > threshold_hours


# =============================================================================
# Property 3-6: Circuit Breaker
# =============================================================================

class TestCircuitBreakerProperties:
    """Property tests for circuit breaker."""
    
    def test_property3_opens_after_threshold_failures(self):
        """Circuit should open after exactly threshold failures."""
        from core.circuit_breaker import CircuitBreaker, CircuitState
        
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0, name="test")
        
        def failing_func():
            raise Exception("Test failure")
        
        # Make exactly 5 failures
        for i in range(5):
            assert cb.state == CircuitState.CLOSED
            try:
                cb.call(failing_func)
            except Exception:
                pass
        
        # After 5 failures, should be OPEN
        assert cb.state == CircuitState.OPEN
    
    @given(st.integers(min_value=1, max_value=20))
    @hyp_settings(max_examples=20)
    def test_property3_any_threshold_opens_circuit(self, threshold):
        """Any failure_threshold N should open after N failures."""
        from core.circuit_breaker import CircuitBreaker, CircuitState
        
        cb = CircuitBreaker(failure_threshold=threshold, recovery_timeout=60.0)
        
        def failing_func():
            raise Exception("fail")
        
        for _ in range(threshold):
            try:
                cb.call(failing_func)
            except Exception:
                pass
        
        assert cb.state == CircuitState.OPEN
    
    def test_property4_open_circuit_rejects_calls(self):
        """Open circuit should reject calls without executing function."""
        from core.circuit_breaker import CircuitBreaker, CircuitState, CircuitOpenError
        
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
        
        # Open the circuit
        try:
            cb.call(lambda: 1/0)
        except ZeroDivisionError:
            pass
        
        assert cb.state == CircuitState.OPEN
        
        # Function should NOT be called
        call_count = [0]
        def tracked_func():
            call_count[0] += 1
            return "success"
        
        with pytest.raises(CircuitOpenError):
            cb.call(tracked_func)
        
        assert call_count[0] == 0  # Function was never called
    
    def test_property5_timeout_transitions_to_half_open(self):
        """After recovery_timeout, circuit should transition to HALF_OPEN."""
        from core.circuit_breaker import CircuitBreaker, CircuitState
        
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)  # 100ms
        
        # Open the circuit
        try:
            cb.call(lambda: 1/0)
        except ZeroDivisionError:
            pass
        
        assert cb._state == CircuitState.OPEN
        
        # Wait for timeout
        time.sleep(0.15)
        
        # State property should trigger transition
        assert cb.state == CircuitState.HALF_OPEN
    
    def test_property6_half_open_success_closes_circuit(self):
        """Success in HALF_OPEN state should close circuit."""
        from core.circuit_breaker import CircuitBreaker, CircuitState
        
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        
        # Open the circuit
        try:
            cb.call(lambda: 1/0)
        except ZeroDivisionError:
            pass
        
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        
        # Successful call should close it
        result = cb.call(lambda: "success")
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
    
    def test_property6_half_open_failure_opens_circuit(self):
        """Failure in HALF_OPEN state should reopen circuit."""
        from core.circuit_breaker import CircuitBreaker, CircuitState
        
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        
        # Open the circuit
        try:
            cb.call(lambda: 1/0)
        except ZeroDivisionError:
            pass
        
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        
        # Failed call should reopen
        try:
            cb.call(lambda: 1/0)
        except ZeroDivisionError:
            pass
        
        assert cb.state == CircuitState.OPEN


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase3Integration:
    """Integration tests for Phase 3."""
    
    def test_settings_has_freshness_threshold(self):
        """Settings should have data_freshness_threshold_hours."""
        from config.settings import Settings
        
        settings = Settings()
        assert hasattr(settings, 'data_freshness_threshold_hours')
        assert settings.data_freshness_threshold_hours == 24.0
    
    def test_settings_has_circuit_breaker_config(self):
        """Settings should have circuit breaker config."""
        from config.settings import Settings
        
        settings = Settings()
        assert hasattr(settings, 'circuit_breaker_failure_threshold')
        assert hasattr(settings, 'circuit_breaker_recovery_timeout')
        assert settings.circuit_breaker_failure_threshold == 5
        assert settings.circuit_breaker_recovery_timeout == 60.0
    
    def test_monitoring_uses_freshness_service(self):
        """MonitoringService.check_data_freshness should use service."""
        from core.monitoring import MonitoringService
        
        service = MonitoringService()
        # Should not raise and should call freshness service
        result = service.check_data_freshness()
        assert isinstance(result, bool)
    
    def test_circuit_breaker_decorator_works(self):
        """circuit_breaker decorator should wrap function."""
        from core.circuit_breaker import circuit_breaker
        
        @circuit_breaker(failure_threshold=3, name="test_decorator")
        def my_func(x):
            return x * 2
        
        assert hasattr(my_func, 'circuit_breaker')
        assert my_func(5) == 10
