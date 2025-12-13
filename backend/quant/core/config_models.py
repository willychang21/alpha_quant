"""Configuration Models for Strategy Definition.

Pydantic models for defining strategy composition via YAML/JSON.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FactorConfig(BaseModel):
    """Configuration for a single factor.

    Attributes:
        name: Registered name of the factor in Registry.
        enabled: Whether this factor is active (default: True).
        params: Optional parameters to pass to factor constructor.
    """

    name: str
    enabled: bool = True
    params: Dict[str, Any] = Field(default_factory=dict)


class OptimizerConfig(BaseModel):
    """Configuration for the optimizer.

    Attributes:
        name: Registered name of the optimizer in Registry.
        params: Optional parameters to pass to optimizer constructor.
    """

    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class RiskRuleConfig(BaseModel):
    """Configuration for a risk rule.

    Attributes:
        name: Registered name of the risk model in Registry.
        params: Optional parameters to pass to risk model constructor.
    """

    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """Pipeline processing configuration.

    Attributes:
        winsorize_limits: Percentile limits for winsorization.
        neutralize_by_sector: Whether to sector-neutralize factors.
        cache_enabled: Whether to enable factor caching.
    """

    winsorize_limits: List[float] = Field(default=[0.01, 0.01])
    neutralize_by_sector: bool = True
    cache_enabled: bool = True


class StrategyConfig(BaseModel):
    """Complete strategy configuration.

    Defines which factors, optimizer, and risk rules to use for a strategy.

    Attributes:
        name: Strategy name/identifier.
        version: Strategy version.
        description: Human-readable description.
        factors: List of factor configurations.
        optimizer: Optimizer configuration.
        risk_rules: List of risk rule configurations.
        pipeline: Pipeline processing configuration.
    """

    name: str
    version: str = "1.0"
    description: str = ""
    factors: List[FactorConfig] = Field(default_factory=list)
    optimizer: Optional[OptimizerConfig] = None
    risk_rules: List[RiskRuleConfig] = Field(default_factory=list)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
