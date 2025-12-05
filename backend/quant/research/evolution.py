import random
import numpy as np
import logging
from typing import List, Tuple, Callable, Dict, Any, Optional
from deap import base, creator, tools, algorithms
from quant.mlops.registry import ModelRegistry

logger = logging.getLogger(__name__)

class GeneticOptimizer:
    def __init__(self, 
                 eval_func: Callable, 
                 param_ranges: Dict[str, Tuple[Any, Any, type]],
                 population_size: int = 50,
                 n_generations: int = 10,
                 crossover_prob: float = 0.5,
                 mutation_prob: float = 0.2,
                 experiment_name: Optional[str] = None,
                 use_ray: bool = False):
        """
        Genetic Algorithm Optimizer using DEAP.
        
        Args:
            eval_func (Callable): Function that takes a list of parameters and returns a tuple (fitness,).
                                  Fitness should be maximized (e.g., Sharpe Ratio).
            param_ranges (Dict): Dictionary defining parameters to optimize.
                                 Key: Name
                                 Value: (Min, Max, Type) e.g. (10, 50, int) or (0.1, 0.5, float)
            population_size (int): Number of individuals in population.
            n_generations (int): Number of generations to evolve.
            experiment_name (str): MLflow experiment name. If provided, logs results.
            use_ray (bool): If True, use Ray for parallel evaluation.
        """
        self.eval_func = eval_func
        self.param_ranges = param_ranges
        self.pop_size = population_size
        self.n_gen = n_generations
        self.cxpb = crossover_prob
        self.mutpb = mutation_prob
        self.use_ray = use_ray
        self.pool = None
        
        self.registry = ModelRegistry(experiment_name=experiment_name) if experiment_name else None
        
        self.toolbox = base.Toolbox()
        self._setup_deap()
        
    def _setup_deap(self):
        # 1. Define Fitness and Individual
        # Check if already created to avoid errors on re-init
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
            
        # 2. Register Attributes
        # We need to register a generator for each parameter
        # But DEAP's register expects a function.
        # Since we have mixed types, we'll create a custom initializer.
        
        self.param_names = list(self.param_ranges.keys())
        
        def init_individual():
            genes = []
            for name in self.param_names:
                min_val, max_val, dtype = self.param_ranges[name]
                if dtype == int:
                    val = random.randint(min_val, max_val)
                elif dtype == float:
                    val = random.uniform(min_val, max_val)
                else:
                    val = min_val # Fallback
                genes.append(val)
            return creator.Individual(genes)
            
        self.toolbox.register("individual", init_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # 3. Register Operators
        self.toolbox.register("evaluate", self.eval_func)
        self.toolbox.register("mate", tools.cxTwoPoint)
        
        # Mutation needs to handle types... this is tricky with mixed types.
        # Standard mutGaussian works for floats.
        # For mixed, we might need custom mutation.
        # For simplicity, we'll use a custom mutator that respects ranges and types.
        self.toolbox.register("mutate", self._custom_mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
        if self.use_ray:
            try:
                import ray
                from ray.util.multiprocessing import Pool
                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True)
                self.pool = Pool()
                self.toolbox.register("map", self.pool.map)
                logger.info("Ray parallelism enabled for Genetic Optimizer.")
            except ImportError:
                logger.warning("Ray not found. Falling back to serial execution.")
            except Exception as e:
                logger.error(f"Failed to initialize Ray: {e}")
        
    def _custom_mutate(self, individual, indpb=0.2):
        """
        Custom mutation operator that respects parameter types and ranges.
        """
        for i, name in enumerate(self.param_names):
            if random.random() < indpb:
                min_val, max_val, dtype = self.param_ranges[name]
                if dtype == int:
                    # Mutate by small integer step or random reset?
                    # Let's do random reset for exploration
                    individual[i] = random.randint(min_val, max_val)
                elif dtype == float:
                    # Gaussian mutation
                    sigma = (max_val - min_val) * 0.1
                    val = individual[i] + random.gauss(0, sigma)
                    val = max(min_val, min(max_val, val)) # Clip
                    individual[i] = val
                    
        return individual,

    def run(self):
        """
        Runs the genetic algorithm.
        """
        logger.info(f"Starting Genetic Optimization (Pop={self.pop_size}, Gen={self.n_gen})...")
        
        if self.registry:
            self.registry.start_run(run_name="genetic_optimization")
            self.registry.log_params({
                "pop_size": self.pop_size,
                "n_gen": self.n_gen,
                "cxpb": self.cxpb,
                "mutpb": self.mutpb,
                "param_ranges": str(self.param_ranges)
            })
        
        pop = self.toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(1)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        
        pop, log = algorithms.eaSimple(
            pop, 
            self.toolbox, 
            cxpb=self.cxpb, 
            mutpb=self.mutpb, 
            ngen=self.n_gen, 
            stats=stats, 
            halloffame=hof, 
            verbose=True
        )
        
        best_ind = hof[0]
        best_params = dict(zip(self.param_names, best_ind))
        best_fitness = best_ind.fitness.values[0]
        
        logger.info(f"Optimization Complete. Best Fitness: {best_fitness:.4f}")
        logger.info(f"Best Parameters: {best_params}")
        
        if self.registry:
            self.registry.log_metrics({"best_fitness": best_fitness})
            # Log best params individually
            for k, v in best_params.items():
                self.registry.log_params({f"best_{k}": v})
            self.registry.end_run()
        
        if self.pool:
            self.pool.close()
            
        return best_params, best_fitness, log
