import sys
import os
import shutil
import pytest
import mlflow

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quant.mlops.registry import ModelRegistry

def test_mlflow_tracking():
    print("\nTesting MLflow Tracking...")
    
    # 1. Setup
    # Use a temporary directory for mlruns
    test_tracking_uri = os.path.abspath("./test_mlruns")
    if os.path.exists(test_tracking_uri):
        shutil.rmtree(test_tracking_uri)
        
    registry = ModelRegistry(experiment_name="test_experiment", tracking_uri=test_tracking_uri)
    
    # 2. Start Run
    with registry.start_run(run_name="test_run"):
        # 3. Log Params & Metrics
        registry.log_params({"param1": 5, "param2": "algo_v1"})
        registry.log_metrics({"accuracy": 0.95, "sharpe": 2.5})
        
        # 4. Log Artifact (Create dummy file)
        with open("dummy.txt", "w") as f:
            f.write("Hello MLflow")
        registry.log_artifact("dummy.txt")
        os.remove("dummy.txt")
        
    # 5. Verify
    best_run = registry.get_best_run(metric_name="sharpe", mode="max")
    
    assert best_run is not None
    print("Best Run ID:", best_run.run_id)
    print("Metrics:", best_run["metrics.sharpe"])
    print("Params:", best_run["params.param1"])
    
    assert best_run["metrics.sharpe"] == 2.5
    assert best_run["params.param1"] == "5"
    
    # Cleanup
    if os.path.exists(test_tracking_uri):
        shutil.rmtree(test_tracking_uri)
        
    print("✅ MLflow Tracking Passed")

if __name__ == "__main__":
    test_mlflow_tracking()
