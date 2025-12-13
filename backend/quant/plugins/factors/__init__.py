"""Factor Plugins Package.

Contains quantitative factor implementations:
- VSM: Volatility Scaled Momentum
- BAB: Betting Against Beta
- QMJ: Quality Minus Junk
- Momentum: Basic price momentum

All factors inherit from FactorBase and register via @register_factor decorator.
"""

# Import all factor modules to trigger registration
from quant.plugins.factors import vsm
from quant.plugins.factors import bab
from quant.plugins.factors import qmj
from quant.plugins.factors import momentum

__all__ = ["vsm", "bab", "qmj", "momentum"]
