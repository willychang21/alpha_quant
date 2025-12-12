"""ML Enhancement Configuration Module.

Provides unified Pydantic configuration for all ML enhancement features:
- SHAP factor attribution
- Residual Alpha model
- Constrained GBM with monotonic constraints
- Online regime detection
- Supply chain GNN factor

All configurations support environment variable overrides.
"""

from functools import lru_cache
from typing import Dict, List, Optional
import logging
import os

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class SHAPConfig(BaseModel):
    """Configuration for SHAP factor attribution.
    
    Attributes:
        enabled: Whether SHAP attribution is enabled.
        concentration_threshold: Maximum allowed factor concentration (0-1).
        background_samples: Number of background samples for SHAP explainer.
    """
    enabled: bool = Field(default=True, description="Enable SHAP attribution")
    concentration_threshold: float = Field(
        default=0.5, 
        ge=0.0, 
        le=1.0,
        description="Factor concentration warning threshold"
    )
    background_samples: int = Field(
        default=100, 
        ge=10,
        description="Background samples for SHAP explainer"
    )


class ResidualAlphaConfig(BaseModel):
    """Configuration for Residual Alpha Model.
    
    The model uses two stages:
    1. Linear decomposition using specified linear_factors
    2. ML prediction on residuals using residual_features
    
    Attributes:
        enabled: Whether residual alpha model is enabled.
        linear_factors: Factors for linear decomposition (Stage 1).
        residual_features: Alternative features for ML prediction (Stage 2).
    """
    enabled: bool = Field(default=True, description="Enable residual alpha model")
    linear_factors: List[str] = Field(
        default=['quality', 'value', 'momentum', 'volatility'],
        description="Factors for linear decomposition"
    )
    residual_features: List[str] = Field(
        default=['sentiment', 'pead', 'capital_flow', 'revisions'],
        description="Features for residual ML prediction"
    )


class ConstrainedGBMConfig(BaseModel):
    """Configuration for Constrained Gradient Boosting Model.
    
    Supports monotonic constraints to enforce economic intuition:
    - +1: monotonically increasing (higher value → higher prediction)
    - -1: monotonically decreasing (higher value → lower prediction)
    - 0: no constraint
    
    Attributes:
        enabled: Whether constrained GBM is enabled.
        monotonic_constraints: Mapping of factor name to constraint direction.
        interaction_groups: Groups of factors allowed to interact.
        num_leaves: LightGBM num_leaves parameter.
        learning_rate: LightGBM learning rate.
    """
    enabled: bool = Field(default=True, description="Enable constrained GBM")
    monotonic_constraints: Dict[str, int] = Field(
        default={
            'quality': 1,      # Higher quality → higher returns
            'value': 1,        # Higher value (cheaper) → higher returns
            'momentum': 1,     # Higher momentum → higher returns
            'volatility': -1,  # Higher volatility → lower returns
            'beta': -1,        # Higher beta → lower risk-adjusted returns
        },
        description="Monotonic constraint direction per factor"
    )
    interaction_groups: List[List[str]] = Field(
        default=[
            ['quality', 'value'],
            ['momentum', 'volatility']
        ],
        description="Groups of factors allowed to interact"
    )
    num_leaves: int = Field(default=31, ge=2, description="LightGBM num_leaves")
    learning_rate: float = Field(
        default=0.05, 
        gt=0.0,
        description="LightGBM learning rate"
    )

    @field_validator('monotonic_constraints')
    @classmethod
    def validate_constraints(cls, v: Dict[str, int]) -> Dict[str, int]:
        """Validate monotonic constraint values are in {-1, 0, 1}."""
        for factor, direction in v.items():
            if direction not in (-1, 0, 1):
                raise ValueError(
                    f"Invalid constraint for {factor}: {direction}. "
                    f"Must be -1, 0, or 1."
                )
        return v


class OnlineLearningConfig(BaseModel):
    """Configuration for Online Regime Detector.
    
    Uses exponential decay to weight recent observations more heavily.
    State is persisted to allow recovery after restart.
    
    Attributes:
        enabled: Whether online learning is enabled.
        decay_factor: Exponential decay factor (0 < decay < 1).
        state_file: Path to persist model state.
        update_threshold: Minimum probability change to trigger weight adjustment.
    """
    enabled: bool = Field(default=True, description="Enable online regime detection")
    decay_factor: float = Field(
        default=0.95, 
        gt=0.0, 
        lt=1.0,
        description="Exponential decay factor for online updates"
    )
    state_file: str = Field(
        default="data/online_regime_state.json",
        description="Path to persist model state"
    )
    update_threshold: float = Field(
        default=0.1, 
        ge=0.0,
        description="Minimum probability change to trigger weight adjustment"
    )


class SupplyChainConfig(BaseModel):
    """Configuration for Supply Chain GNN Factor.
    
    Disabled by default as it requires external supply chain relationship data.
    
    Attributes:
        enabled: Whether supply chain factor is enabled.
        data_file: Path to supply chain relationship data (CSV/Parquet).
        decay_factor: Signal decay factor for propagation.
        propagation_steps: Number of message passing iterations.
    """
    enabled: bool = Field(
        default=False,  # Disabled by default (requires data)
        description="Enable supply chain GNN factor"
    )
    data_file: Optional[str] = Field(
        default=None,
        description="Path to supply chain relationship data"
    )
    decay_factor: float = Field(
        default=0.5, 
        gt=0.0, 
        lt=1.0,
        description="Signal decay factor for propagation"
    )
    propagation_steps: int = Field(
        default=2, 
        ge=1,
        description="Number of message passing iterations"
    )


class MLEnhancementConfig(BaseModel):
    """Unified configuration for all ML enhancement features.
    
    Each sub-configuration can be individually enabled/disabled.
    All settings can be overridden via environment variables.
    
    Example:
        >>> config = get_ml_enhancement_config()
        >>> if config.shap.enabled:
        ...     # Compute SHAP attributions
        ...     pass
    """
    shap: SHAPConfig = Field(default_factory=SHAPConfig)
    residual_alpha: ResidualAlphaConfig = Field(default_factory=ResidualAlphaConfig)
    constrained_gbm: ConstrainedGBMConfig = Field(default_factory=ConstrainedGBMConfig)
    online_learning: OnlineLearningConfig = Field(default_factory=OnlineLearningConfig)
    supply_chain: SupplyChainConfig = Field(default_factory=SupplyChainConfig)

    def get_active_features(self) -> List[str]:
        """Return list of enabled feature names.
        
        Returns:
            List of feature names that are currently enabled.
        """
        features = []
        if self.shap.enabled:
            features.append("SHAP")
        if self.residual_alpha.enabled:
            features.append("ResidualAlpha")
        if self.constrained_gbm.enabled:
            features.append("ConstrainedGBM")
        if self.online_learning.enabled:
            features.append("OnlineLearning")
        if self.supply_chain.enabled:
            features.append("SupplyChain")
        return features


# Singleton instance
_ml_enhancement_config: Optional[MLEnhancementConfig] = None


def get_ml_enhancement_config() -> MLEnhancementConfig:
    """Get or create MLEnhancementConfig singleton.
    
    Reads configuration from environment variables if available:
    - ML_SHAP_ENABLED: Enable/disable SHAP attribution
    - ML_RESIDUAL_ALPHA_ENABLED: Enable/disable residual alpha
    - ML_CONSTRAINED_GBM_ENABLED: Enable/disable constrained GBM
    - ML_ONLINE_LEARNING_ENABLED: Enable/disable online learning
    - ML_SUPPLY_CHAIN_ENABLED: Enable/disable supply chain factor
    - ML_ONLINE_STATE_FILE: Path for online regime state persistence
    
    Returns:
        MLEnhancementConfig singleton instance.
    """
    global _ml_enhancement_config
    if _ml_enhancement_config is None:
        # Build config from environment overrides
        shap_config = SHAPConfig(
            enabled=os.environ.get('ML_SHAP_ENABLED', 'true').lower() == 'true',
            concentration_threshold=float(
                os.environ.get('ML_SHAP_CONCENTRATION_THRESHOLD', '0.5')
            ),
            background_samples=int(
                os.environ.get('ML_SHAP_BACKGROUND_SAMPLES', '100')
            ),
        )
        
        residual_config = ResidualAlphaConfig(
            enabled=os.environ.get('ML_RESIDUAL_ALPHA_ENABLED', 'true').lower() == 'true',
        )
        
        gbm_config = ConstrainedGBMConfig(
            enabled=os.environ.get('ML_CONSTRAINED_GBM_ENABLED', 'true').lower() == 'true',
            num_leaves=int(os.environ.get('ML_GBM_NUM_LEAVES', '31')),
            learning_rate=float(os.environ.get('ML_GBM_LEARNING_RATE', '0.05')),
        )
        
        online_config = OnlineLearningConfig(
            enabled=os.environ.get('ML_ONLINE_LEARNING_ENABLED', 'true').lower() == 'true',
            decay_factor=float(os.environ.get('ML_ONLINE_DECAY_FACTOR', '0.95')),
            state_file=os.environ.get('ML_ONLINE_STATE_FILE', 'data/online_regime_state.json'),
        )
        
        supply_chain_config = SupplyChainConfig(
            enabled=os.environ.get('ML_SUPPLY_CHAIN_ENABLED', 'false').lower() == 'true',
            data_file=os.environ.get('ML_SUPPLY_CHAIN_DATA_FILE'),
        )
        
        _ml_enhancement_config = MLEnhancementConfig(
            shap=shap_config,
            residual_alpha=residual_config,
            constrained_gbm=gbm_config,
            online_learning=online_config,
            supply_chain=supply_chain_config,
        )
        
        active = _ml_enhancement_config.get_active_features()
        logger.info(f"ML Enhancement features active: {active}")
    
    return _ml_enhancement_config


def reset_ml_enhancement_config() -> None:
    """Reset ML enhancement config singleton. Useful for testing."""
    global _ml_enhancement_config
    _ml_enhancement_config = None
