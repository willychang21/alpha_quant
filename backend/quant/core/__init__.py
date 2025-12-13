"""Core module for plugin registry system.

This package provides the infrastructure for a plugin-based architecture:
- interfaces: Abstract base classes for plugins
- registry: Singleton registry with decorator-based registration
- config_models: Pydantic models for configuration
- config_loader: Configuration loading and validation
"""

from quant.core.interfaces import (
    FactorBase,
    OptimizerBase,
    PluginMetadata,
    RiskModelBase,
)
from quant.core.registry import (
    register_factor,
    register_optimizer,
    register_risk_model,
    registry,
)

__all__ = [
    "PluginMetadata",
    "FactorBase",
    "OptimizerBase",
    "RiskModelBase",
    "registry",
    "register_factor",
    "register_optimizer",
    "register_risk_model",
]

