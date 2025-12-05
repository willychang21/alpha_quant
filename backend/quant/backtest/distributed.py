import ray
import logging
import pandas as pd
from typing import List, Dict, Any, Callable

logger = logging.getLogger(__name__)

@ray.remote
def run_backtest_task(backtest_func: Callable, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remote function to run a single backtest.
    
    Args:
        backtest_func (Callable): Function that takes params and returns metrics.
        params (Dict): Parameters for the backtest.
        
    Returns:
        Dict: Result metrics.
    """
    try:
        # Run the backtest
        metrics = backtest_func(params)
        return {"params": params, "metrics": metrics, "status": "success"}
    except Exception as e:
        return {"params": params, "error": str(e), "status": "failed"}

class DistributedBacktester:
    def __init__(self, num_cpus: int = None):
        """
        Distributed Backtester using Ray.
        
        Args:
            num_cpus (int): Number of CPUs to use. If None, uses all available.
        """
        if not ray.is_initialized():
            ray.init(num_cpus=num_cpus, ignore_reinit_error=True)
            logger.info(f"Ray initialized with resources: {ray.cluster_resources()}")
        else:
            logger.info("Ray already initialized.")
            
    def run_batch(self, backtest_func: Callable, param_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Runs a batch of backtests in parallel.
        
        Args:
            backtest_func (Callable): Function to execute.
            param_list (List[Dict]): List of parameter dictionaries.
            
        Returns:
            List[Dict]: List of results.
        """
        logger.info(f"Submitting {len(param_list)} backtest tasks to Ray...")
        
        # Submit tasks
        futures = [run_backtest_task.remote(backtest_func, params) for params in param_list]
        
        # Wait for results
        results = ray.get(futures)
        
        logger.info(f"Completed {len(results)} tasks.")
        return results
        
    def shutdown(self):
        """Shuts down Ray."""
        ray.shutdown()
