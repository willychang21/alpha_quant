"""Risk Model Plugins Package.

Contains risk constraint implementations:
- MaxWeight: Maximum single position weight constraint
- Sector: Sector concentration constraint
- Beta: Portfolio beta bounds constraint

All risk models inherit from RiskModelBase and register via @register_risk_model decorator.
"""

# Import all risk model modules to trigger registration
from quant.plugins.risk_models import max_weight
from quant.plugins.risk_models import sector
from quant.plugins.risk_models import beta

__all__ = ["max_weight", "sector", "beta"]
