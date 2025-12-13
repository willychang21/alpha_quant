"""Sector Concentration Risk Constraint Plugin.

Ensures no single sector exceeds a maximum weight threshold.
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from quant.core.interfaces import PluginMetadata, RiskModelBase
from quant.core.registry import register_risk_model

logger = logging.getLogger(__name__)


@register_risk_model("Sector")
class SectorConstraint(RiskModelBase):
    """Sector concentration constraint.

    Ensures no single sector exceeds the configured maximum weight.

    Attributes:
        params: Configuration parameters.
        max_sector_weight: Maximum allowed weight per sector.
    """

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        """Initialize Sector constraint.

        Args:
            params: Configuration dict with optional keys:
                - max_sector_weight: Maximum weight per sector (default: 0.30)
        """
        self.params = params or {}
        self.max_sector_weight = self.params.get("max_sector_weight", 0.30)

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="Sector",
            description="Sector concentration constraint",
            version="1.0.0",
            author="DCA Quant",
            category="risk",
            parameters={
                "max_sector_weight": "Maximum weight per sector (default: 0.30)",
            },
        )

    def check_constraints(
        self,
        weights: pd.Series,
        **context: Any,
    ) -> tuple[bool, Optional[str]]:
        """Check if all sector weights are within the maximum limit.

        Args:
            weights: Portfolio weights indexed by ticker.
            **context: Must include 'sectors' - a dict or Series mapping ticker to sector.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if weights.empty:
            return True, None

        sectors = context.get("sectors")
        if sectors is None:
            # No sector info provided - cannot check
            logger.warning("No sectors provided for SectorConstraint check")
            return True, None

        # Convert to Series if dict
        if isinstance(sectors, dict):
            sectors = pd.Series(sectors)

        # Calculate sector weights
        sector_weights = pd.Series(dtype=float)
        for ticker, weight in weights.items():
            sector = sectors.get(ticker, "Unknown")
            if sector in sector_weights:
                sector_weights[sector] += weight
            else:
                sector_weights[sector] = weight

        # Check max
        max_sector = sector_weights.max()
        if max_sector > self.max_sector_weight:
            violators = sector_weights[sector_weights > self.max_sector_weight]
            msg = (
                f"Sector weight exceeded: {max_sector:.2%} > {self.max_sector_weight:.2%}. "
                f"Violating sectors: {dict(violators)}"
            )
            return False, msg

        return True, None

    def get_constraint_matrix(self) -> Optional[Dict[str, Any]]:
        """Return constraint specification for CVXPY integration."""
        return {
            "type": "sector",
            "max_weight": self.max_sector_weight,
        }
