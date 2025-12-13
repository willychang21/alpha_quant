"""MaxWeight Risk Constraint Plugin.

Ensures no single position exceeds a maximum weight threshold.
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from quant.core.interfaces import PluginMetadata, RiskModelBase
from quant.core.registry import register_risk_model

logger = logging.getLogger(__name__)


@register_risk_model("MaxWeight")
class MaxWeightConstraint(RiskModelBase):
    """Maximum single position weight constraint.

    Ensures no individual position exceeds the configured maximum weight.

    Attributes:
        params: Configuration parameters.
        max_weight: Maximum allowed weight per position.
    """

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        """Initialize MaxWeight constraint.

        Args:
            params: Configuration dict with optional keys:
                - max_weight: Maximum weight per position (default: 0.10)
        """
        self.params = params or {}
        self.max_weight = self.params.get("max_weight", 0.10)

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="MaxWeight",
            description="Maximum single position weight constraint",
            version="1.0.0",
            author="DCA Quant",
            category="risk",
            parameters={
                "max_weight": "Maximum weight per position (default: 0.10)",
            },
        )

    def check_constraints(
        self,
        weights: pd.Series,
        **context: Any,
    ) -> tuple[bool, Optional[str]]:
        """Check if all weights are within the maximum limit.

        Args:
            weights: Portfolio weights indexed by ticker.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if weights.empty:
            return True, None

        max_actual = weights.max()

        if max_actual > self.max_weight:
            violators = weights[weights > self.max_weight]
            msg = (
                f"Max weight exceeded: {max_actual:.2%} > {self.max_weight:.2%}. "
                f"Violating positions: {list(violators.index)}"
            )
            return False, msg

        return True, None

    def get_constraint_matrix(self) -> Optional[Dict[str, Any]]:
        """Return constraint specification for CVXPY integration."""
        return {
            "type": "bound",
            "upper": self.max_weight,
            "lower": 0.0,
        }
