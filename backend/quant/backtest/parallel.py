from typing import List, Dict, Any, Callable
from joblib import Parallel, delayed
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class ParallelBacktester:
    """
    Executes backtests in parallel using joblib.
    Useful for parameter sweeps or multi-ticker analysis.
    """
    
    def __init__(self, n_jobs: int = -1):
        """
        :param n_jobs: Number of parallel jobs. -1 means use all CPUs.
        """
        self.n_jobs = n_jobs
        
    def run_batch(
        self, 
        backtest_func: Callable, 
        configs: List[Dict[str, Any]]
    ) -> List[Any]:
        """
        Run a batch of backtests.
        :param backtest_func: Function to execute for each config. Must be picklable.
        :param configs: List of configuration dictionaries passed to the function.
        :return: List of results.
        """
        logger.info(f"Running {len(configs)} backtests in parallel with n_jobs={self.n_jobs}...")
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(backtest_func)(config) for config in configs
        )
        
        return results

# Example usage function (must be top-level for pickling in some envs)
def _run_simulation_wrapper(config):
    from quant.backtest.simulator import PortfolioSimulator
    # Unpack config
    history = config.get('history')
    allocation = config.get('allocation')
    monthly = config.get('monthly_amount', 1000)
    
    return PortfolioSimulator.simulate_dca(history, allocation, monthly)
