"""Factor Registry Module.

Provides a registry of available factors with metadata, user weight management,
and regime-aware blending.

Key features:
- Factor metadata (name, description, category, default weight)
- User weight overrides with validation
- Regime-multiplied effective weights
- Factor enable/disable functionality

Example:
    >>> registry = FactorRegistry()
    >>> factors = registry.get_all_factors()
    >>> registry.set_user_weight('quality', 1.2)
    >>> weights = registry.get_user_weights()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FactorCategory(str, Enum):
    """Category classification for factors."""
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    ML_DERIVED = "ml_derived"
    ALTERNATIVE = "alternative"


@dataclass
class FactorMetadata:
    """Metadata for a registered factor.
    
    Attributes:
        name: Unique factor identifier.
        description: Human-readable description.
        category: Factor category classification.
        default_weight: Default weight in composite scoring.
        user_weight: User-configured weight override.
        enabled: Whether factor is active.
        ic_5d: Information coefficient (5-day forward return).
        turnover: Average daily turnover.
    """
    name: str
    description: str
    category: FactorCategory
    default_weight: float = 1.0
    user_weight: Optional[float] = None
    enabled: bool = True
    ic_5d: Optional[float] = None
    turnover: Optional[float] = None
    
    @property
    def effective_weight(self) -> float:
        """Get effective weight (user override or default)."""
        if not self.enabled:
            return 0.0
        return self.user_weight if self.user_weight is not None else self.default_weight
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'category': self.category.value,
            'default_weight': self.default_weight,
            'user_weight': self.user_weight,
            'enabled': self.enabled,
            'effective_weight': self.effective_weight,
            'ic_5d': self.ic_5d,
            'turnover': self.turnover
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FactorMetadata':
        """Deserialize from dictionary."""
        return cls(
            name=data['name'],
            description=data['description'],
            category=FactorCategory(data['category']),
            default_weight=data.get('default_weight', 1.0),
            user_weight=data.get('user_weight'),
            enabled=data.get('enabled', True),
            ic_5d=data.get('ic_5d'),
            turnover=data.get('turnover')
        )


class FactorRegistry:
    """Registry of available factors with weight management.
    
    Provides factor discovery, weight configuration, and regime-aware
    weight blending.
    
    Attributes:
        factors: Dict of factor name to FactorMetadata.
        regime_multipliers: Multipliers by regime (e.g., 'Bull' -> 1.0).
    """
    
    # Default factors available in the system
    DEFAULT_FACTORS = [
        FactorMetadata(
            name="quality",
            description="Measures profitability, margins, and financial health",
            category=FactorCategory.FUNDAMENTAL,
            default_weight=1.0
        ),
        FactorMetadata(
            name="value",
            description="Valuation metrics like P/E, P/B, PEG ratio",
            category=FactorCategory.FUNDAMENTAL,
            default_weight=1.0
        ),
        FactorMetadata(
            name="momentum",
            description="Price momentum and trend strength",
            category=FactorCategory.TECHNICAL,
            default_weight=0.8
        ),
        FactorMetadata(
            name="volatility",
            description="Price volatility and risk metrics",
            category=FactorCategory.TECHNICAL,
            default_weight=0.6
        ),
        FactorMetadata(
            name="sentiment",
            description="Analyst ratings and sentiment indicators",
            category=FactorCategory.SENTIMENT,
            default_weight=0.7
        ),
        FactorMetadata(
            name="capital_flow",
            description="Money flow and institutional buying",
            category=FactorCategory.ALTERNATIVE,
            default_weight=0.8
        ),
        FactorMetadata(
            name="supply_chain",
            description="Supply chain network momentum spillover",
            category=FactorCategory.ML_DERIVED,
            default_weight=0.5
        ),
        FactorMetadata(
            name="residual_alpha",
            description="ML-extracted alpha after factor decomposition",
            category=FactorCategory.ML_DERIVED,
            default_weight=1.0
        )
    ]
    
    DEFAULT_REGIME_MULTIPLIERS = {
        "Bull": {"momentum": 1.2, "value": 0.8},
        "Bear": {"momentum": 0.5, "value": 1.3, "quality": 1.2},
        "Neutral": {}
    }
    
    def __init__(
        self,
        factors: Optional[List[FactorMetadata]] = None,
        regime_multipliers: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """Initialize FactorRegistry.
        
        Args:
            factors: Optional list of factors (uses defaults if None).
            regime_multipliers: Regime-specific weight multipliers.
        """
        factor_list = factors or self.DEFAULT_FACTORS
        self.factors: Dict[str, FactorMetadata] = {
            f.name: f for f in factor_list
        }
        self.regime_multipliers = regime_multipliers or self.DEFAULT_REGIME_MULTIPLIERS
    
    def get_all_factors(self) -> List[FactorMetadata]:
        """Get all registered factors.
        
        Returns:
            List of FactorMetadata for all factors.
        """
        return list(self.factors.values())
    
    def get_factor(self, name: str) -> Optional[FactorMetadata]:
        """Get a specific factor by name.
        
        Args:
            name: Factor name.
            
        Returns:
            FactorMetadata or None if not found.
        """
        return self.factors.get(name)
    
    def set_user_weight(self, factor_name: str, weight: float) -> bool:
        """Set user weight override for a factor.
        
        Args:
            factor_name: Name of factor to update.
            weight: New weight (must be >= 0).
            
        Returns:
            True if successful, False if factor not found or invalid weight.
        """
        if weight < 0:
            logger.warning(f"Invalid weight {weight} for {factor_name} (must be >= 0)")
            return False
        
        factor = self.factors.get(factor_name)
        if factor is None:
            logger.warning(f"Factor not found: {factor_name}")
            return False
        
        factor.user_weight = weight
        logger.info(f"Set user weight for {factor_name}: {weight}")
        return True
    
    def get_user_weights(self, regime: str = "Neutral") -> Dict[str, float]:
        """Get effective weights with regime multipliers applied.
        
        Args:
            regime: Current market regime (Bull/Bear/Neutral).
            
        Returns:
            Dict of factor name to adjusted weight.
        """
        regime_mult = self.regime_multipliers.get(regime, {})
        
        weights = {}
        for name, factor in self.factors.items():
            base_weight = factor.effective_weight
            multiplier = regime_mult.get(name, 1.0)
            weights[name] = base_weight * multiplier
        
        return weights
    
    def reset_user_weights(self) -> None:
        """Reset all user weights to None (use defaults)."""
        for factor in self.factors.values():
            factor.user_weight = None
        logger.info("Reset all user weights to defaults")
    
    def toggle_factor(self, factor_name: str, enabled: bool) -> bool:
        """Enable or disable a factor.
        
        Args:
            factor_name: Name of factor.
            enabled: Whether to enable.
            
        Returns:
            True if successful.
        """
        factor = self.factors.get(factor_name)
        if factor is None:
            return False
        
        factor.enabled = enabled
        logger.info(f"{'Enabled' if enabled else 'Disabled'} factor: {factor_name}")
        return True
    
    def update_performance(
        self,
        factor_name: str,
        ic_5d: Optional[float] = None,
        turnover: Optional[float] = None
    ) -> bool:
        """Update performance metrics for a factor.
        
        Args:
            factor_name: Name of factor.
            ic_5d: Information coefficient.
            turnover: Daily turnover rate.
            
        Returns:
            True if successful.
        """
        factor = self.factors.get(factor_name)
        if factor is None:
            return False
        
        if ic_5d is not None:
            factor.ic_5d = ic_5d
        if turnover is not None:
            factor.turnover = turnover
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize registry state."""
        return {
            'factors': [f.to_dict() for f in self.factors.values()],
            'regime_multipliers': self.regime_multipliers
        }
