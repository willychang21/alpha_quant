"""Configuration Loader for Strategy Files.

Loads and validates strategy configurations from YAML or JSON files.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

from quant.core.config_models import (
    FactorConfig,
    OptimizerConfig,
    RiskRuleConfig,
    StrategyConfig,
)
from quant.core.registry import registry

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""

    pass


def load_config(path: str | Path) -> StrategyConfig:
    """Load and validate a strategy configuration file.

    Args:
        path: Path to YAML or JSON configuration file.

    Returns:
        Validated StrategyConfig object.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If file format is unsupported.
        ConfigurationError: If configuration is invalid.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    # Load raw config
    raw_config = _load_raw_config(path)

    # Parse into StrategyConfig
    config = _parse_config(raw_config)

    # Validate all referenced plugins exist
    _validate_plugins(config)

    logger.info(f"Loaded strategy configuration: {config.name} v{config.version}")

    return config


def _load_raw_config(path: Path) -> Dict[str, Any]:
    """Load raw configuration from file."""
    with open(path, "r") as f:
        if path.suffix in (".yaml", ".yml"):
            return yaml.safe_load(f) or {}
        elif path.suffix == ".json":
            return json.load(f)
        else:
            raise ValueError(
                f"Unsupported config format: {path.suffix}. "
                f"Supported: .yaml, .yml, .json"
            )


def _parse_config(raw: Dict[str, Any]) -> StrategyConfig:
    """Parse raw config dict into StrategyConfig."""
    # Handle nested 'strategy' key
    if "strategy" in raw:
        strategy_info = raw.get("strategy", {})
    else:
        strategy_info = raw

    # Extract factors
    factors_raw = raw.get("factors", [])
    factors: List[FactorConfig] = []
    for fc in factors_raw:
        if isinstance(fc, str):
            factors.append(FactorConfig(name=fc))
        else:
            factors.append(FactorConfig(**fc))

    # Extract optimizer
    optimizer_raw = raw.get("optimizer")
    optimizer = None
    if optimizer_raw:
        optimizer = OptimizerConfig(**optimizer_raw)

    # Extract risk rules
    risk_raw = raw.get("risk_rules", [])
    risk_rules = [RiskRuleConfig(**r) for r in risk_raw]

    # Extract pipeline config
    pipeline_raw = raw.get("pipeline", {})

    return StrategyConfig(
        name=strategy_info.get("name", "unnamed_strategy"),
        version=str(strategy_info.get("version", "1.0")),
        description=strategy_info.get("description", ""),
        factors=factors,
        optimizer=optimizer,
        risk_rules=risk_rules,
        pipeline=pipeline_raw if pipeline_raw else None,
    )


def _validate_plugins(config: StrategyConfig) -> None:
    """Validate all referenced plugins exist in Registry."""
    # Ensure plugins are discovered
    try:
        registry.discover_plugins()
    except Exception:
        pass  # May fail if plugins not yet created

    errors: List[str] = []

    # Validate factors
    available_factors = registry.list_factors()
    for fc in config.factors:
        if fc.enabled and fc.name not in available_factors:
            errors.append(
                f"Factor '{fc.name}' not found. Available: {available_factors}"
            )

    # Validate optimizer
    if config.optimizer:
        available_optimizers = registry.list_optimizers()
        if config.optimizer.name not in available_optimizers:
            errors.append(
                f"Optimizer '{config.optimizer.name}' not found. "
                f"Available: {available_optimizers}"
            )

    # Validate risk rules
    available_risk_models = registry.list_risk_models()
    for rr in config.risk_rules:
        if rr.name not in available_risk_models:
            errors.append(
                f"Risk model '{rr.name}' not found. "
                f"Available: {available_risk_models}"
            )

    if errors:
        raise ConfigurationError(
            "Invalid configuration:\n" + "\n".join(f"  - {e}" for e in errors)
        )


def validate_config_file(path: str | Path) -> List[str]:
    """Validate a configuration file without raising exceptions.

    Args:
        path: Path to configuration file.

    Returns:
        List of validation error messages (empty if valid).
    """
    try:
        load_config(path)
        return []
    except FileNotFoundError as e:
        return [str(e)]
    except ValueError as e:
        return [str(e)]
    except ConfigurationError as e:
        return [str(e)]
    except Exception as e:
        return [f"Unexpected error: {e}"]
