"""Unit tests for health endpoints."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Tests for /health and /ready endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from main import app
        return TestClient(app)
    
    def test_health_returns_healthy(self, client):
        """Test /health returns healthy status."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_ready_returns_status(self, client):
        """Test /ready returns readiness status with checks."""
        response = client.get("/api/v1/ready")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have status and checks
        assert "status" in data
        assert "checks" in data
        assert "timestamp" in data
        
        # Status should be ready or not_ready
        assert data["status"] in ["ready", "not_ready"]
    
    def test_ready_includes_database_check(self, client):
        """Test /ready includes database connectivity check."""
        response = client.get("/api/v1/ready")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should include database check
        assert "database" in data["checks"]
    
    def test_ready_includes_data_lag(self, client):
        """Test /ready includes data freshness metric."""
        response = client.get("/api/v1/ready")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should include data_lag_hours (may be None if no data)
        assert "data_lag_hours" in data["checks"]
    
    def test_correlation_id_in_response(self, client):
        """Test that X-Request-ID is included in response headers."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
    
    def test_correlation_id_passthrough(self, client):
        """Test that provided X-Request-ID is passed through."""
        custom_id = "test-123"
        response = client.get(
            "/api/v1/health",
            headers={"X-Request-ID": custom_id}
        )
        
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == custom_id
