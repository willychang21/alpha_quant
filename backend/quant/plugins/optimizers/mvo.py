"""Mean-Variance Optimization (MVO) Plugin.

Classic Markowitz optimization with optional constraints.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from quant.core.interfaces import OptimizerBase, PluginMetadata
from quant.core.registry import register_optimizer

logger = logging.getLogger(__name__)

# Try to import cvxpy for convex optimization
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    logger.warning("cvxpy not available, MVO will use fallback optimization")


@register_optimizer("MVO")
class MVOOptimizer(OptimizerBase):
    """Mean-Variance Optimization (Markowitz).

    Maximizes expected return for a given risk level, or minimizes
    risk for a given return target.

    Attributes:
        params: Configuration parameters.
        risk_aversion: Risk aversion parameter.
        max_weight: Maximum single position weight.
    """

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        """Initialize MVO optimizer.

        Args:
            params: Configuration dict with optional keys:
                - risk_aversion: Risk aversion coefficient (default: 2.5)
                - max_weight: Max position weight (default: 0.20)
                - long_only: Long-only constraint (default: True)
        """
        self.params = params or {}
        self.risk_aversion = self.params.get("risk_aversion", 2.5)
        self.max_weight = self.params.get("max_weight", 0.20)
        self.long_only = self.params.get("long_only", True)

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="MVO",
            description="Mean-Variance Optimization (Markowitz)",
            version="1.0.0",
            author="DCA Quant",
            category="optimization",
            parameters={
                "risk_aversion": "Risk aversion coefficient (default: 2.5)",
                "max_weight": "Maximum position weight (default: 0.20)",
                "long_only": "Long-only constraint (default: True)",
            },
        )

    def optimize(
        self,
        returns: pd.DataFrame,
        cov: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.Series:
        """Run MVO optimization.

        Args:
            returns: Expected returns per asset.
            cov: Covariance matrix.
            **kwargs: Optional 'alpha_scores' for expected returns.

        Returns:
            Series of weights indexed by ticker.
        """
        # Get expected returns
        if "alpha_scores" in kwargs:
            mu = pd.Series(kwargs["alpha_scores"])
        elif isinstance(returns, pd.Series):
            mu = returns
        elif isinstance(returns, pd.DataFrame):
            mu = returns.mean()
        else:
            # Use zeros if no expected returns provided
            mu = pd.Series(0.0, index=cov.columns)

        # Align mu with covariance matrix
        mu = mu.reindex(cov.columns).fillna(0)

        if HAS_CVXPY:
            return self._optimize_cvxpy(mu, cov)
        else:
            return self._optimize_fallback(mu, cov)

    def _optimize_cvxpy(self, mu: pd.Series, cov: pd.DataFrame) -> pd.Series:
        """Optimize using CVXPY."""
        n = len(mu)
        w = cp.Variable(n)

        # Objective: maximize mu'w - (lambda/2) * w'Σw
        ret = mu.values @ w
        risk = cp.quad_form(w, cov.values)
        objective = cp.Maximize(ret - (self.risk_aversion / 2) * risk)

        # Constraints
        constraints = [cp.sum(w) == 1]  # Fully invested

        if self.long_only:
            constraints.append(w >= 0)

        if self.max_weight < 1.0:
            constraints.append(w <= self.max_weight)

        # Solve
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.OSQP, verbose=False)

            if prob.status == "optimal":
                weights = w.value
                weights[np.abs(weights) < 1e-6] = 0.0
                return pd.Series(weights, index=cov.columns)
            else:
                logger.warning(f"MVO status: {prob.status}, using fallback")
                return self._optimize_fallback(mu, cov)

        except Exception as e:
            logger.error(f"CVXPY failed: {e}")
            return self._optimize_fallback(mu, cov)

    def _optimize_fallback(self, mu: pd.Series, cov: pd.DataFrame) -> pd.Series:
        """Fallback to equal weight with minor tilts."""
        n = len(mu)
        base_weight = 1.0 / n

        # Small tilt toward higher expected returns
        if mu.std() > 0:
            z_scores = (mu - mu.mean()) / mu.std()
            tilts = z_scores * 0.01 * base_weight  # 1% tilt
        else:
            tilts = 0

        weights = base_weight + tilts

        # Ensure non-negative and sum to 1
        if self.long_only:
            weights = np.maximum(weights, 0)

        weights = weights / weights.sum()

        # Apply max weight constraint
        if self.max_weight < 1.0:
            weights = np.minimum(weights, self.max_weight)
            weights = weights / weights.sum()

        return pd.Series(weights, index=cov.columns)
