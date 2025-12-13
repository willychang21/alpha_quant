"""Kelly Criterion Optimizer Plugin.

Multivariate Kelly for optimal geometric growth.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from quant.core.interfaces import OptimizerBase, PluginMetadata
from quant.core.registry import register_optimizer

logger = logging.getLogger(__name__)

# Try to import cvxpy
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


@register_optimizer("Kelly")
class KellyOptimizer(OptimizerBase):
    """Multivariate Kelly Criterion Optimizer.

    Maximizes geometric growth rate (log wealth).
    g(w) ≈ r + w'(μ - r) - 0.5 * w'Σw

    Attributes:
        params: Configuration parameters.
        fractional_kelly: Fraction of full Kelly to use.
        risk_free_rate: Risk-free rate for excess returns.
    """

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        """Initialize Kelly optimizer.

        Args:
            params: Configuration dict with optional keys:
                - fractional_kelly: Fraction of Kelly (default: 0.5)
                - risk_free_rate: Risk-free rate (default: 0.0)
                - max_leverage: Maximum leverage (default: 1.0)
        """
        self.params = params or {}
        self.fractional_kelly = self.params.get("fractional_kelly", 0.5)
        self.risk_free_rate = self.params.get("risk_free_rate", 0.0)
        self.max_leverage = self.params.get("max_leverage", 1.0)

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="Kelly",
            description="Kelly Criterion - maximize geometric growth",
            version="1.0.0",
            author="DCA Quant",
            category="optimization",
            parameters={
                "fractional_kelly": "Fraction of full Kelly (default: 0.5)",
                "risk_free_rate": "Risk-free rate (default: 0.0)",
                "max_leverage": "Maximum leverage (default: 1.0)",
            },
        )

    def optimize(
        self,
        returns: pd.DataFrame,
        cov: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.Series:
        """Run Kelly optimization.

        Args:
            returns: Expected returns per asset.
            cov: Covariance matrix.

        Returns:
            Series of weights indexed by ticker.
        """
        # Get expected returns
        if isinstance(returns, pd.Series):
            mu = returns.reindex(cov.columns).fillna(0)
        elif isinstance(returns, pd.DataFrame):
            mu = returns.mean().reindex(cov.columns).fillna(0)
        else:
            mu = pd.Series(0.0, index=cov.columns)

        if HAS_CVXPY:
            return self._optimize_cvxpy(mu, cov)
        else:
            return self._optimize_fallback(mu, cov)

    def _optimize_cvxpy(self, mu: pd.Series, cov: pd.DataFrame) -> pd.Series:
        """Optimize using CVXPY."""
        n = len(mu)

        # Excess returns
        excess = mu.values - self.risk_free_rate

        w = cp.Variable(n)

        # Objective: maximize w'μ - 0.5 * w'Σw (equivalent to log growth)
        port_return = w @ excess
        port_risk = cp.quad_form(w, cov.values)
        objective = cp.Maximize(port_return - 0.5 * port_risk)

        constraints = [
            cp.sum(cp.abs(w)) <= self.max_leverage,  # Leverage constraint
            w >= 0,  # Long-only
        ]

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.OSQP, verbose=False)

            if prob.status == "optimal":
                weights = w.value * self.fractional_kelly
                weights[np.abs(weights) < 1e-6] = 0.0

                # Normalize to sum to 1 for comparable weights
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                else:
                    weights = np.ones(n) / n

                return pd.Series(weights, index=cov.columns)
            else:
                logger.warning(f"Kelly status: {prob.status}")
                return self._optimize_fallback(mu, cov)

        except Exception as e:
            logger.error(f"Kelly CVXPY failed: {e}")
            return self._optimize_fallback(mu, cov)

    def _optimize_fallback(self, mu: pd.Series, cov: pd.DataFrame) -> pd.Series:
        """Fallback to inverse-variance weighting."""
        variances = pd.Series(np.diag(cov), index=cov.columns)
        inv_var = 1.0 / (variances + 1e-8)
        weights = inv_var / inv_var.sum()
        return weights
