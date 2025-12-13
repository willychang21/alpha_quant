"""Plugin Registry with Decorator-Based Registration.

Provides a global singleton registry for managing plugin components:
- Factors (quantitative signals)
- Optimizers (portfolio optimization algorithms)
- Risk Models (constraint checkers)

Example Usage:
    from quant.core.registry import register_factor, registry

    @register_factor("Momentum")
    class MomentumFactor(FactorBase):
        ...

    # Later, retrieve and instantiate
    factor_cls = registry.get_factor("Momentum")
    factor = factor_cls(params={"lookback": 252})
"""

import importlib
import logging
import pkgutil
from typing import Any, Callable, Dict, List, Type

from quant.core.interfaces import FactorBase, OptimizerBase, RiskModelBase

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Global registry for all plugin types.

    Singleton pattern ensures a single source of truth for plugins.
    Uses decorator-based registration for clean, declarative plugin definition.

    Attributes:
        _factors: Dict mapping factor names to their classes.
        _optimizers: Dict mapping optimizer names to their classes.
        _risk_models: Dict mapping risk model names to their classes.
        _initialized: Flag indicating if auto-discovery has run.
    """

    _instance: "PluginRegistry | None" = None

    def __new__(cls) -> "PluginRegistry":
        """Create singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._factors: Dict[str, Type[FactorBase]] = {}
            cls._instance._optimizers: Dict[str, Type[OptimizerBase]] = {}
            cls._instance._risk_models: Dict[str, Type[RiskModelBase]] = {}
            cls._instance._initialized: bool = False
        return cls._instance

    # =========================================================================
    # Factor Registration
    # =========================================================================

    def register_factor(self, name: str) -> Callable[[Type[FactorBase]], Type[FactorBase]]:
        """Decorator to register a factor class.

        Args:
            name: Unique name for the factor plugin.

        Returns:
            Decorator function that registers the class.

        Raises:
            TypeError: If the class doesn't inherit from FactorBase.

        Example:
            @registry.register_factor("VSM")
            class VSMFactor(FactorBase):
                ...
        """
        def decorator(cls: Type[FactorBase]) -> Type[FactorBase]:
            if not issubclass(cls, FactorBase):
                raise TypeError(f"{cls.__name__} must inherit from FactorBase")
            if name in self._factors:
                logger.warning(f"Overwriting existing factor: {name}")
            self._factors[name] = cls
            logger.info(f"Registered factor: {name}")
            return cls
        return decorator

    def get_factor(self, name: str) -> Type[FactorBase]:
        """Get a factor class by name.

        Args:
            name: Registered name of the factor.

        Returns:
            The factor class.

        Raises:
            KeyError: If the factor is not registered.
        """
        if name not in self._factors:
            available = list(self._factors.keys())
            raise KeyError(f"Factor '{name}' not found. Available: {available}")
        return self._factors[name]

    def list_factors(self) -> List[str]:
        """List all registered factor names.

        Returns:
            Sorted list of registered factor names.
        """
        return sorted(self._factors.keys())

    # =========================================================================
    # Optimizer Registration
    # =========================================================================

    def register_optimizer(
        self, name: str
    ) -> Callable[[Type[OptimizerBase]], Type[OptimizerBase]]:
        """Decorator to register an optimizer class.

        Args:
            name: Unique name for the optimizer plugin.

        Returns:
            Decorator function that registers the class.

        Raises:
            TypeError: If the class doesn't inherit from OptimizerBase.
        """
        def decorator(cls: Type[OptimizerBase]) -> Type[OptimizerBase]:
            if not issubclass(cls, OptimizerBase):
                raise TypeError(f"{cls.__name__} must inherit from OptimizerBase")
            if name in self._optimizers:
                logger.warning(f"Overwriting existing optimizer: {name}")
            self._optimizers[name] = cls
            logger.info(f"Registered optimizer: {name}")
            return cls
        return decorator

    def get_optimizer(self, name: str) -> Type[OptimizerBase]:
        """Get an optimizer class by name.

        Args:
            name: Registered name of the optimizer.

        Returns:
            The optimizer class.

        Raises:
            KeyError: If the optimizer is not registered.
        """
        if name not in self._optimizers:
            available = list(self._optimizers.keys())
            raise KeyError(f"Optimizer '{name}' not found. Available: {available}")
        return self._optimizers[name]

    def list_optimizers(self) -> List[str]:
        """List all registered optimizer names.

        Returns:
            Sorted list of registered optimizer names.
        """
        return sorted(self._optimizers.keys())

    # =========================================================================
    # Risk Model Registration
    # =========================================================================

    def register_risk_model(
        self, name: str
    ) -> Callable[[Type[RiskModelBase]], Type[RiskModelBase]]:
        """Decorator to register a risk model class.

        Args:
            name: Unique name for the risk model plugin.

        Returns:
            Decorator function that registers the class.

        Raises:
            TypeError: If the class doesn't inherit from RiskModelBase.
        """
        def decorator(cls: Type[RiskModelBase]) -> Type[RiskModelBase]:
            if not issubclass(cls, RiskModelBase):
                raise TypeError(f"{cls.__name__} must inherit from RiskModelBase")
            if name in self._risk_models:
                logger.warning(f"Overwriting existing risk model: {name}")
            self._risk_models[name] = cls
            logger.info(f"Registered risk model: {name}")
            return cls
        return decorator

    def get_risk_model(self, name: str) -> Type[RiskModelBase]:
        """Get a risk model class by name.

        Args:
            name: Registered name of the risk model.

        Returns:
            The risk model class.

        Raises:
            KeyError: If the risk model is not registered.
        """
        if name not in self._risk_models:
            available = list(self._risk_models.keys())
            raise KeyError(f"Risk model '{name}' not found. Available: {available}")
        return self._risk_models[name]

    def list_risk_models(self) -> List[str]:
        """List all registered risk model names.

        Returns:
            Sorted list of registered risk model names.
        """
        return sorted(self._risk_models.keys())

    # =========================================================================
    # Auto-Discovery
    # =========================================================================

    def discover_plugins(self, package_path: str = "quant.plugins") -> None:
        """Auto-discover and import all plugin modules.

        Walks through the specified package and imports all modules,
        triggering their registration decorators.

        Args:
            package_path: Dotted path to the plugins package.
        """
        if self._initialized:
            logger.debug("Plugin discovery already completed, skipping")
            return

        try:
            package = importlib.import_module(package_path)
        except ModuleNotFoundError:
            logger.warning(f"Plugin package not found: {package_path}")
            return

        for importer, modname, ispkg in pkgutil.walk_packages(
            package.__path__, prefix=f"{package_path}."
        ):
            try:
                importlib.import_module(modname)
                logger.debug(f"Loaded plugin module: {modname}")
            except Exception as e:
                logger.warning(f"Failed to load {modname}: {e}")

        self._initialized = True
        logger.info(
            f"Plugin discovery complete: "
            f"{len(self._factors)} factors, "
            f"{len(self._optimizers)} optimizers, "
            f"{len(self._risk_models)} risk models"
        )

    # =========================================================================
    # Metadata
    # =========================================================================

    def get_all_metadata(self) -> Dict[str, Any]:
        """Get metadata for all registered plugins.

        Returns:
            Dict with 'factors', 'optimizers', 'risk_models' keys,
            each containing a dict of name -> metadata.
        """
        result: Dict[str, Any] = {
            "factors": {},
            "optimizers": {},
            "risk_models": {},
        }

        for name, cls in self._factors.items():
            try:
                instance = cls(params={})
                result["factors"][name] = instance.metadata.to_dict()
            except Exception as e:
                logger.warning(f"Failed to get metadata for factor {name}: {e}")
                result["factors"][name] = {"name": name, "error": str(e)}

        for name, cls in self._optimizers.items():
            try:
                instance = cls(params={})
                result["optimizers"][name] = instance.metadata.to_dict()
            except Exception as e:
                logger.warning(f"Failed to get metadata for optimizer {name}: {e}")
                result["optimizers"][name] = {"name": name, "error": str(e)}

        for name, cls in self._risk_models.items():
            try:
                instance = cls(params={})
                result["risk_models"][name] = instance.metadata.to_dict()
            except Exception as e:
                logger.warning(f"Failed to get metadata for risk model {name}: {e}")
                result["risk_models"][name] = {"name": name, "error": str(e)}

        return result

    def clear(self) -> None:
        """Clear all registrations. Mainly for testing."""
        self._factors.clear()
        self._optimizers.clear()
        self._risk_models.clear()
        self._initialized = False


# Global registry instance
registry = PluginRegistry()


# Convenience decorators at module level
def register_factor(name: str) -> Callable[[Type[FactorBase]], Type[FactorBase]]:
    """Module-level convenience decorator for registering factors.

    Args:
        name: Unique name for the factor plugin.

    Returns:
        Decorator function.

    Example:
        @register_factor("VSM")
        class VSMFactor(FactorBase):
            ...
    """
    return registry.register_factor(name)


def register_optimizer(name: str) -> Callable[[Type[OptimizerBase]], Type[OptimizerBase]]:
    """Module-level convenience decorator for registering optimizers.

    Args:
        name: Unique name for the optimizer plugin.

    Returns:
        Decorator function.
    """
    return registry.register_optimizer(name)


def register_risk_model(
    name: str,
) -> Callable[[Type[RiskModelBase]], Type[RiskModelBase]]:
    """Module-level convenience decorator for registering risk models.

    Args:
        name: Unique name for the risk model plugin.

    Returns:
        Decorator function.
    """
    return registry.register_risk_model(name)
