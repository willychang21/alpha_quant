import sys
import os
import pytest
import logging

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quant.research.evolution import GeneticOptimizer

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_genetic_optimization():
    print("\nTesting Genetic Optimization...")
    
    # 1. Define Objective Function
    # Maximize: -(x - 10)^2 - (y - 5)^2
    # Maximum is at x=10, y=5 (Fitness = 0)
    def eval_func(individual):
        x, y = individual
        fitness = -((x - 10)**2 + (y - 5)**2)
        return (fitness,)
        
    # 2. Define Parameter Ranges
    param_ranges = {
        'x': (0, 20, float),
        'y': (0, 10, float)
    }
    
    # 3. Initialize Optimizer
    # Use a test experiment name
    optimizer = GeneticOptimizer(
        eval_func=eval_func,
        param_ranges=param_ranges,
        population_size=50,
        n_generations=5, # Reduce for speed
        crossover_prob=0.7,
        mutation_prob=0.2,
        experiment_name="test_genetic_algo"
    )
    
    # 4. Run Optimization
    best_params, best_fitness, log = optimizer.run()
    
    print(f"Best Params: {best_params}")
    print(f"Best Fitness: {best_fitness}")
    
    # 5. Verify Results
    # Should be close to x=10, y=5
    assert abs(best_params['x'] - 10.0) < 0.5
    assert abs(best_params['y'] - 5.0) < 0.5
    assert best_fitness > -0.5
    
    print("✅ Genetic Optimization Passed")

if __name__ == "__main__":
    test_genetic_optimization()
