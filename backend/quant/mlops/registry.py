import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, experiment_name: str = "quant_experiments", tracking_uri: str = "mlruns"):
        """
        Wrapper for MLflow Model Registry and Experiment Tracking.
        
        Args:
            experiment_name (str): Name of the experiment bucket.
            tracking_uri (str): Path to store runs (default: local ./mlruns).
        """
        # Set tracking URI (can be local file or remote server)
        # If relative path, make it absolute
        if not tracking_uri.startswith("http") and not os.path.isabs(tracking_uri):
            tracking_uri = os.path.abspath(tracking_uri)
            
        mlflow.set_tracking_uri(f"file://{tracking_uri}")
        mlflow.set_experiment(experiment_name)
        
        self.experiment_name = experiment_name
        logger.info(f"Initialized ModelRegistry: {experiment_name} @ {tracking_uri}")
        
    def start_run(self, run_name: str = None):
        """Starts a new MLflow run."""
        return mlflow.start_run(run_name=run_name)
        
    def log_params(self, params: Dict[str, Any]):
        """Logs hyperparameters."""
        mlflow.log_params(params)
        
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Logs performance metrics."""
        mlflow.log_metrics(metrics, step=step)
        
    def log_model(self, model: Any, artifact_path: str = "model"):
        """
        Logs a model artifact. Automatically detects sklearn/xgboost.
        """
        if "sklearn" in str(type(model)):
            mlflow.sklearn.log_model(model, artifact_path)
        elif "xgboost" in str(type(model)):
            mlflow.xgboost.log_model(model, artifact_path)
        else:
            # Fallback to generic python function or pickle?
            # For now, just log as sklearn if it behaves like one, or warning
            logger.warning(f"Unknown model type {type(model)}, trying sklearn logger...")
            try:
                mlflow.sklearn.log_model(model, artifact_path)
            except:
                logger.error("Failed to log model.")
                
    def log_artifact(self, local_path: str):
        """Logs a local file as an artifact."""
        mlflow.log_artifact(local_path)
        
    def end_run(self):
        """Ends the current run."""
        mlflow.end_run()
        
    def get_best_run(self, metric_name: str = "sharpe_ratio", mode: str = "max") -> Optional[pd.Series]:
        """
        Retrieves the best run based on a metric.
        
        Args:
            metric_name (str): Metric to sort by.
            mode (str): 'max' or 'min'.
            
        Returns:
            pd.Series: Metadata of the best run.
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if not experiment:
                return None
                
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            
            if runs.empty:
                return None
                
            # Sort
            ascending = True if mode == "min" else False
            runs = runs.sort_values(f"metrics.{metric_name}", ascending=ascending)
            
            return runs.iloc[0]
            
        except Exception as e:
            logger.error(f"Failed to get best run: {e}")
            return None
