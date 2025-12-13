"""Dynamic Factor Pipeline - Configuration-Driven Processing.

Loads factors dynamically from Registry based on configuration.
No code changes required to add new factors.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from quant.core.config_loader import load_config, ConfigurationError
from quant.core.config_models import StrategyConfig
from quant.core.interfaces import FactorBase
from quant.core.registry import registry

logger = logging.getLogger(__name__)


class DynamicFactorPipeline:
    """Configuration-driven factor computation pipeline.

    Loads factors from Registry based on YAML/JSON configuration.
    Supports dynamic factor addition/removal at runtime.

    Attributes:
        config: Loaded strategy configuration.
        factors: Dict of instantiated factor plugins.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[StrategyConfig] = None,
    ) -> None:
        """Initialize pipeline from configuration.

        Args:
            config_path: Path to YAML or JSON config file.
            config: Pre-loaded StrategyConfig (used if config_path is None).

        Raises:
            FileNotFoundError: If config_path doesn't exist.
            ConfigurationError: If configuration is invalid.
        """
        # Ensure plugins are discovered
        try:
            registry.discover_plugins()
        except Exception:
            pass  # May already be discovered or plugins not yet created

        # Load configuration
        if config_path:
            self.config = load_config(config_path)
        elif config:
            self.config = config
        else:
            self.config = StrategyConfig(name="empty")

        # Instantiate factors
        self.factors: Dict[str, FactorBase] = {}
        self._initialize_factors()

        logger.info(
            f"DynamicFactorPipeline initialized with {len(self.factors)} factors"
        )

    def _initialize_factors(self) -> None:
        """Instantiate factors from configuration."""
        for fc in self.config.factors:
            if not fc.enabled:
                logger.info(f"Skipping disabled factor: {fc.name}")
                continue

            try:
                factor_cls = registry.get_factor(fc.name)
                self.factors[fc.name] = factor_cls(params=fc.params)
                logger.debug(f"Initialized factor: {fc.name}")
            except KeyError as e:
                logger.error(f"Factor not found: {fc.name}. {e}")
                raise ConfigurationError(f"Factor not found: {fc.name}")

    def compute_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute all configured factors.

        Args:
            data: Market data DataFrame with 'ticker', 'date', 'close' columns.

        Returns:
            DataFrame with one column per factor, indexed by ticker.
        """
        results: Dict[str, pd.Series] = {}

        for name, factor in self.factors.items():
            try:
                # Validate input if method is implemented
                if hasattr(factor, "validate_input"):
                    if not factor.validate_input(data):
                        logger.warning(f"Input validation failed for {name}")

                result = factor.compute(data)
                results[name] = result
                logger.debug(f"Computed factor: {name}")

            except Exception as e:
                logger.error(f"Error computing {name}: {e}")
                # Return empty series on error
                results[name] = pd.Series(dtype=float, name=name)

        return pd.DataFrame(results)

    def add_factor(self, name: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Dynamically add a factor at runtime.

        Args:
            name: Registered factor name.
            params: Optional parameters for factor constructor.

        Raises:
            KeyError: If factor name not in Registry.
        """
        factor_cls = registry.get_factor(name)
        self.factors[name] = factor_cls(params=params or {})
        logger.info(f"Added factor: {name}")

    def remove_factor(self, name: str) -> None:
        """Remove a factor from the pipeline.

        Args:
            name: Factor name to remove.
        """
        if name in self.factors:
            del self.factors[name]
            logger.info(f"Removed factor: {name}")

    def list_factors(self) -> List[str]:
        """List all active factors in the pipeline.

        Returns:
            List of factor names.
        """
        return list(self.factors.keys())

    def get_factor_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all active factors.

        Returns:
            Dict of factor name to metadata dict.
        """
        return {
            name: factor.metadata.to_dict()
            for name, factor in self.factors.items()
        }
