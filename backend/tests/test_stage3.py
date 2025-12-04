import sys
import os
import pandas as pd
import numpy as np
import time

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from quant.backtest.parallel import ParallelBacktester, _run_simulation_wrapper

def test_parallel_backtest():
    print("Testing Parallel Backtesting...", flush=True)
    
    # Mock Data
    dates = pd.date_range(start='2020-01-01', periods=24, freq='M')
    history = pd.DataFrame({
        'AAPL': np.random.normal(1.01, 0.05, 24).cumprod() * 100,
        'MSFT': np.random.normal(1.01, 0.04, 24).cumprod() * 100,
        'GOOGL': np.random.normal(1.01, 0.06, 24).cumprod() * 100
    }, index=dates)
    
    # Create configurations for batch run
    configs = []
    allocations = [
        {'AAPL': 1.0},
        {'MSFT': 1.0},
        {'GOOGL': 1.0},
        {'AAPL': 0.5, 'MSFT': 0.5},
        {'AAPL': 0.33, 'MSFT': 0.33, 'GOOGL': 0.33}
    ]
    
    for alloc in allocations:
        configs.append({
            'history': history,
            'allocation': alloc,
            'monthly_amount': 1000
        })
        
    print(f"Running {len(configs)} simulations...", flush=True)
    
    start_time = time.time()
    tester = ParallelBacktester(n_jobs=2) # Use 2 cores
    results = tester.run_batch(_run_simulation_wrapper, configs)
    end_time = time.time()
    
    print(f"Completed in {end_time - start_time:.4f} seconds.", flush=True)
    
    assert len(results) == len(configs)
    for i, res in enumerate(results):
        final_val = res['metrics']['final_value_dca']
        print(f"Sim {i+1} Final Value: ${final_val:,.2f}", flush=True)
        assert final_val > 0
        
    print("Parallel Backtest Test Passed!", flush=True)

if __name__ == "__main__":
    test_parallel_backtest()
