"""Optimizer Plugins Package.

Contains portfolio optimization algorithm implementations:
- HRP: Hierarchical Risk Parity
- MVO: Mean-Variance Optimization
- BlackLitterman: Black-Litterman model
- Kelly: Kelly Criterion

All optimizers inherit from OptimizerBase and register via @register_optimizer decorator.
"""

# Import all optimizer modules to trigger registration
from quant.plugins.optimizers import hrp
from quant.plugins.optimizers import mvo
from quant.plugins.optimizers import black_litterman
from quant.plugins.optimizers import kelly

__all__ = ["hrp", "mvo", "black_litterman", "kelly"]
