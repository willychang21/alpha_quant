import sys
import os
import time
import pytest
import ray

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quant.backtest.distributed import DistributedBacktester
from quant.research.evolution import GeneticOptimizer

def dummy_backtest(params):
    """Simulates a backtest."""
    time.sleep(0.1) # Simulate work
    return {"sharpe": params['x'] * params['y']}

def test_distributed_backtester():
    print("\nTesting Distributed Backtester...")
    
    # 1. Setup
    backtester = DistributedBacktester()
    
    # 2. Create Params
    param_list = [
        {'x': 1, 'y': 2},
        {'x': 2, 'y': 3},
        {'x': 3, 'y': 4}
    ]
    
    # 3. Run Batch
    start_time = time.time()
    results = backtester.run_batch(dummy_backtest, param_list)
    end_time = time.time()
    
    # 4. Verify
    assert len(results) == 3
    assert results[0]['metrics']['sharpe'] == 2
    assert results[2]['metrics']['sharpe'] == 12
    
    print(f"Batch Time: {end_time - start_time:.4f}s")
    print("✅ Distributed Backtester Passed")
    
    # Shutdown Ray to allow GeneticOptimizer to re-init if needed
    backtester.shutdown()

def test_genetic_optimizer_ray():
    print("\nTesting Genetic Optimizer with Ray...")
    
    # 1. Define Objective
    def eval_func(individual):
        x, y = individual
        # Maximize -(x-10)^2 - (y-5)^2
        # Add delay to make parallel worth it?
        # time.sleep(0.01) 
        fitness = -((x - 10)**2 + (y - 5)**2)
        return (fitness,)
        
    # 2. Setup Optimizer
    param_ranges = {
        'x': (0, 20, float),
        'y': (0, 10, float)
    }
    
    optimizer = GeneticOptimizer(
        eval_func=eval_func,
        param_ranges=param_ranges,
        population_size=40,
        n_generations=5,
        use_ray=True,
        experiment_name="test_ray_evolution"
    )
    
    # 3. Run
    best_params, best_fitness, log = optimizer.run()
    
    print(f"Best Params: {best_params}")
    print(f"Best Fitness: {best_fitness}")
    
    assert abs(best_params['x'] - 10.0) < 1.0
    
    print("✅ Genetic Optimizer (Ray) Passed")

if __name__ == "__main__":
    test_distributed_backtester()
    test_genetic_optimizer_ray()
