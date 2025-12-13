"""Beta Bounds Risk Constraint Plugin.

Ensures portfolio beta stays within configured bounds.
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from quant.core.interfaces import PluginMetadata, RiskModelBase
from quant.core.registry import register_risk_model

logger = logging.getLogger(__name__)


@register_risk_model("Beta")
class BetaConstraint(RiskModelBase):
    """Portfolio beta bounds constraint.

    Ensures the portfolio's weighted average beta stays within bounds.

    Attributes:
        params: Configuration parameters.
        min_beta: Minimum portfolio beta.
        max_beta: Maximum portfolio beta.
    """

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        """Initialize Beta constraint.

        Args:
            params: Configuration dict with optional keys:
                - min_beta: Minimum portfolio beta (default: 0.8)
                - max_beta: Maximum portfolio beta (default: 1.2)
        """
        self.params = params or {}
        self.min_beta = self.params.get("min_beta", 0.8)
        self.max_beta = self.params.get("max_beta", 1.2)

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="Beta",
            description="Portfolio beta bounds constraint",
            version="1.0.0",
            author="DCA Quant",
            category="risk",
            parameters={
                "min_beta": "Minimum portfolio beta (default: 0.8)",
                "max_beta": "Maximum portfolio beta (default: 1.2)",
            },
        )

    def check_constraints(
        self,
        weights: pd.Series,
        **context: Any,
    ) -> tuple[bool, Optional[str]]:
        """Check if portfolio beta is within bounds.

        Args:
            weights: Portfolio weights indexed by ticker.
            **context: Must include 'betas' - a dict or Series mapping ticker to beta.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if weights.empty:
            return True, None

        betas = context.get("betas")
        if betas is None:
            # No beta info provided - cannot check
            logger.warning("No betas provided for BetaConstraint check")
            return True, None

        # Convert to Series if dict
        if isinstance(betas, dict):
            betas = pd.Series(betas)

        # Calculate portfolio beta (weighted average)
        common_tickers = weights.index.intersection(betas.index)
        if len(common_tickers) == 0:
            logger.warning("No common tickers between weights and betas")
            return True, None

        aligned_weights = weights.loc[common_tickers]
        aligned_betas = betas.loc[common_tickers]

        # Renormalize weights to sum to 1 for the subset
        if aligned_weights.sum() > 0:
            aligned_weights = aligned_weights / aligned_weights.sum()
        else:
            return True, None

        portfolio_beta = (aligned_weights * aligned_betas).sum()

        # Check bounds
        if portfolio_beta < self.min_beta:
            msg = (
                f"Portfolio beta too low: {portfolio_beta:.3f} < {self.min_beta:.3f}"
            )
            return False, msg

        if portfolio_beta > self.max_beta:
            msg = (
                f"Portfolio beta too high: {portfolio_beta:.3f} > {self.max_beta:.3f}"
            )
            return False, msg

        return True, None

    def get_constraint_matrix(self) -> Optional[Dict[str, Any]]:
        """Return constraint specification for CVXPY integration."""
        return {
            "type": "beta",
            "min": self.min_beta,
            "max": self.max_beta,
        }
