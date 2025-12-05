import sys
import os
import pytest
from fastapi.testclient import TestClient

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app

client = TestClient(app)

def test_ml_signals_endpoint():
    print("\nTesting ML Signals Endpoint...")
    response = client.get("/api/v1/quant/ml/signals")
    
    print("Status Code:", response.status_code)
    print("Response:", response.json())
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    # It might return "no_data" if no run exists, or "success" if the previous test run exists
    # We ran test_tier3_mlops.py which created a run, but it might have been cleaned up?
    # Actually, we used a local ./mlruns or ./test_mlruns.
    # The API uses default ./mlruns if not specified? 
    # In registry.py: tracking_uri="mlruns" default.
    # In test_tier3_mlops.py: we used ./test_mlruns and cleaned it up.
    # In test_tier2_evolution.py: we used experiment_name="test_ray_evolution" but didn't specify tracking_uri, so it used default ./mlruns?
    # Yes, likely ./mlruns exists.
    
    if data["status"] == "success":
        assert "metrics" in data
        assert "params" in data

def test_risk_metrics_endpoint():
    print("\nTesting Risk Metrics Endpoint...")
    response = client.get("/api/v1/quant/risk/metrics?portfolio_value=1000000&volatility=0.2")
    
    print("Status Code:", response.status_code)
    print("Response:", response.json())
    
    assert response.status_code == 200
    data = response.json()
    assert "var" in data
    assert "hedge" in data
    assert data["var"]["portfolio_var"] > 0
    assert data["hedge"]["total_cost"] > 0

def test_execution_vwap_endpoint():
    print("\nTesting VWAP Execution Endpoint...")
    response = client.get("/api/v1/quant/execution/vwap?shares=5000")
    
    print("Status Code:", response.status_code)
    print("Response:", response.json())
    
    assert response.status_code == 200
    data = response.json()
    assert "schedule" in data
    assert len(data["schedule"]) > 0
    assert data["total_shares"] == 5000

if __name__ == "__main__":
    test_ml_signals_endpoint()
    test_risk_metrics_endpoint()
    test_execution_vwap_endpoint()
