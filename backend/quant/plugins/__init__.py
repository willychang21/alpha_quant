"""Plugin Package - Auto-Discovery and Registration.

This package contains all plugin modules for the registry system:
- factors/: Quantitative factor plugins (VSM, BAB, QMJ, Momentum, etc.)
- optimizers/: Portfolio optimization plugins (HRP, MVO, BlackLitterman, Kelly)
- risk_models/: Risk constraint plugins (MaxWeight, Sector, Beta)

On import, this package triggers auto-discovery of all plugin modules,
causing their @register_* decorators to execute and register with the global registry.
"""

import logging

logger = logging.getLogger(__name__)

# Import subpackages to trigger their registrations
try:
    from quant.plugins import factors
except ImportError as e:
    logger.debug(f"Could not import factors subpackage: {e}")

try:
    from quant.plugins import optimizers
except ImportError as e:
    logger.debug(f"Could not import optimizers subpackage: {e}")

try:
    from quant.plugins import risk_models
except ImportError as e:
    logger.debug(f"Could not import risk_models subpackage: {e}")

# Also trigger explicit discovery for any modules that weren't imported
from quant.core.registry import registry

# Perform auto-discovery (will be a no-op if already done)
try:
    registry.discover_plugins("quant.plugins")
except Exception as e:
    logger.debug(f"Plugin discovery skipped or failed: {e}")
