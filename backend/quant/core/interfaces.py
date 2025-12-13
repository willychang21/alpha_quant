"""Core Interfaces for Plugin Registry System.

Defines Abstract Base Classes (ABCs) for all plugin types:
- FactorBase: For quantitative factors (momentum, value, quality, etc.)
- OptimizerBase: For portfolio optimization algorithms (HRP, MVO, B-L, Kelly)
- RiskModelBase: For risk constraint checkers

All plugins must inherit from their respective base class and implement
required abstract methods. Python will raise TypeError at class definition
time if required methods are not implemented.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class PluginMetadata:
    """Metadata for registered plugins.

    Attributes:
        name: Unique identifier for the plugin.
        description: Human-readable description of the plugin.
        version: Semantic version string.
        author: Plugin author or maintainer.
        parameters: Dict of parameter names to descriptions.
        category: Optional category classification (e.g., "momentum", "value").
    """

    name: str
    description: str
    version: str = "1.0.0"
    author: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    category: str = ""

    def __post_init__(self) -> None:
        """Ensure parameters is never None."""
        if self.parameters is None:
            self.parameters = {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to dictionary.

        Returns:
            Dict representation of metadata.
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "parameters": self.parameters,
            "category": self.category,
        }


class FactorBase(ABC):
    """Abstract base class for all factor plugins.

    All factors must implement the compute() method which takes market data
    and returns a Series of factor values per ticker.

    Example:
        @register_factor("Momentum")
        class MomentumFactor(FactorBase):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="Momentum",
                    description="12-month price momentum"
                )

            def compute(self, data: pd.DataFrame) -> pd.Series:
                # Implementation here
                ...
    """

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata.

        Returns:
            PluginMetadata instance describing this factor.
        """
        pass

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute factor values from market data.

        Args:
            data: DataFrame with columns like 'ticker', 'date', 'close',
                  'volume', and potentially fundamental data.

        Returns:
            Series indexed by ticker with factor values.
        """
        pass

    def validate_input(self, data: pd.DataFrame) -> bool:
        """Optional input validation hook.

        Override to add custom validation logic. Default always returns True.

        Args:
            data: Input DataFrame to validate.

        Returns:
            True if input is valid, False otherwise.
        """
        return True


class OptimizerBase(ABC):
    """Abstract base class for portfolio optimizers.

    All optimizers must implement the optimize() method which takes
    expected returns and covariance matrix and returns optimal weights.

    Example:
        @register_optimizer("HRP")
        class HRPOptimizer(OptimizerBase):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="HRP",
                    description="Hierarchical Risk Parity"
                )

            def optimize(self, returns, cov, **kwargs) -> pd.Series:
                # Implementation here
                ...
    """

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata.

        Returns:
            PluginMetadata instance describing this optimizer.
        """
        pass

    @abstractmethod
    def optimize(
        self,
        returns: pd.DataFrame,
        cov: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.Series:
        """Optimize portfolio weights.

        Args:
            returns: Expected returns per asset (Series or DataFrame).
            cov: Covariance matrix of asset returns.
            **kwargs: Additional parameters (alpha scores, constraints, etc.).

        Returns:
            Series of weights indexed by ticker, summing to 1.0.
        """
        pass


class RiskModelBase(ABC):
    """Abstract base class for risk constraint models.

    All risk models must implement check_constraints() which validates
    portfolio weights against specific risk rules.

    Example:
        @register_risk_model("MaxWeight")
        class MaxWeightConstraint(RiskModelBase):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="MaxWeight",
                    description="Maximum single position weight"
                )

            def check_constraints(self, weights, **context):
                max_w = self.params.get("max_weight", 0.10)
                if weights.max() > max_w:
                    return False, f"Max weight exceeded: {weights.max()}"
                return True, None
    """

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata.

        Returns:
            PluginMetadata instance describing this risk model.
        """
        pass

    @abstractmethod
    def check_constraints(
        self,
        weights: pd.Series,
        **context: Any,
    ) -> tuple[bool, Optional[str]]:
        """Check if weights satisfy constraints.

        Args:
            weights: Portfolio weights indexed by ticker.
            **context: Additional context (sectors, betas, prices, etc.).

        Returns:
            Tuple of (is_valid, error_message).
            If valid, error_message should be None.
        """
        pass

    def get_constraint_matrix(self) -> Optional[Dict[str, Any]]:
        """Return constraint matrices for CVXPY integration.

        Override to provide constraint matrices for convex optimization.
        Default returns None (no CVXPY integration).

        Returns:
            Dict with constraint specification, or None.
        """
        return None
