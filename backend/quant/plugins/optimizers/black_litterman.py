"""Black-Litterman Optimizer Plugin.

Combines market equilibrium with alpha views.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from quant.core.interfaces import OptimizerBase, PluginMetadata
from quant.core.registry import register_optimizer

logger = logging.getLogger(__name__)


@register_optimizer("BlackLitterman")
class BlackLittermanOptimizer(OptimizerBase):
    """Black-Litterman Model Optimizer.

    Combines market equilibrium returns with investor views (alpha signals)
    to produce Bayesian posterior expected returns.

    E[R] = [(τΣ)^-1 + P'Ω^-1 P]^-1 * [(τΣ)^-1 π + P'Ω^-1 Q]

    Attributes:
        params: Configuration parameters.
        tau: Scaling factor for uncertainty in equilibrium.
        risk_aversion: Market risk aversion coefficient.
    """

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        """Initialize Black-Litterman optimizer.

        Args:
            params: Configuration dict with optional keys:
                - tau: Uncertainty scaling factor (default: 0.05)
                - risk_aversion: Market risk aversion (default: 2.5)
                - ic: Information coefficient for views (default: 0.05)
        """
        self.params = params or {}
        self.tau = self.params.get("tau", 0.05)
        self.risk_aversion = self.params.get("risk_aversion", 2.5)
        self.ic = self.params.get("ic", 0.05)

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="BlackLitterman",
            description="Black-Litterman - Bayesian combining of views with equilibrium",
            version="1.0.0",
            author="DCA Quant",
            category="optimization",
            parameters={
                "tau": "Uncertainty scaling factor (default: 0.05)",
                "risk_aversion": "Market risk aversion (default: 2.5)",
                "ic": "Information coefficient for alpha views (default: 0.05)",
            },
        )

    def optimize(
        self,
        returns: pd.DataFrame,
        cov: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.Series:
        """Run Black-Litterman optimization.

        Args:
            returns: Historical returns or expected returns.
            cov: Covariance matrix.
            **kwargs: Optional keys:
                - market_caps: Market caps for equilibrium weights
                - z_scores: Alpha Z-scores for views
                - volatilities: Asset volatilities

        Returns:
            Series of weights indexed by ticker.
        """
        try:
            n = len(cov)
            tickers = cov.columns

            # Get market caps (use equal if not provided)
            market_caps = kwargs.get("market_caps")
            if market_caps is None:
                market_caps = pd.Series(1.0, index=tickers)
            else:
                market_caps = pd.Series(market_caps).reindex(tickers).fillna(1.0)

            # Calculate equilibrium returns: π = δΣw_mkt
            mkt_weights = market_caps / market_caps.sum()
            pi = self.risk_aversion * (cov @ mkt_weights)

            # Get views from z_scores if provided
            z_scores = kwargs.get("z_scores")
            if z_scores is None:
                # No views - just use equilibrium
                posterior_returns = pi
            else:
                z_scores = pd.Series(z_scores).reindex(tickers).fillna(0)
                volatilities = kwargs.get("volatilities")
                if volatilities is None:
                    volatilities = pd.Series(np.sqrt(np.diag(cov)), index=tickers)
                else:
                    volatilities = pd.Series(volatilities).reindex(tickers).fillna(0.2)

                Q, P, Omega = self._alpha_to_views(z_scores, volatilities)
                posterior_returns = self._posterior_returns(pi, cov, Q, P, Omega)

            # Optimize using posterior returns
            weights = self._optimize_weights(posterior_returns, cov)

            return weights

        except Exception as e:
            logger.error(f"Black-Litterman failed: {e}")
            n = len(cov)
            return pd.Series(1.0 / n, index=cov.columns)

    def _alpha_to_views(
        self, z_scores: pd.Series, volatilities: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert Z-scores to Black-Litterman views.

        Uses Grinold-Kahn mapping: α_i = IC × σ_i × Z_i
        """
        n = len(z_scores)

        # Views: Q = IC * σ * Z
        Q = (self.ic * volatilities * z_scores).values

        # Picking matrix: identity (absolute views)
        P = np.eye(n)

        # Uncertainty: Omega = diag(σ^2 / IC^2)
        # Higher IC = lower uncertainty in views
        view_var = (volatilities.values ** 2) / (self.ic ** 2 + 1e-10)
        Omega = np.diag(view_var)

        return Q, P, Omega

    def _posterior_returns(
        self,
        pi: pd.Series,
        cov: pd.DataFrame,
        Q: np.ndarray,
        P: np.ndarray,
        Omega: np.ndarray,
    ) -> pd.Series:
        """Calculate posterior expected returns."""
        tau_cov = self.tau * cov.values
        tau_cov_inv = np.linalg.inv(tau_cov + np.eye(len(cov)) * 1e-8)

        Omega_inv = np.linalg.inv(Omega + np.eye(len(Omega)) * 1e-8)

        # Posterior precision
        post_precision = tau_cov_inv + P.T @ Omega_inv @ P

        # Posterior mean
        post_cov = np.linalg.inv(post_precision + np.eye(len(post_precision)) * 1e-8)
        post_mean = post_cov @ (tau_cov_inv @ pi.values + P.T @ Omega_inv @ Q)

        return pd.Series(post_mean, index=cov.columns)

    def _optimize_weights(
        self, expected_returns: pd.Series, cov: pd.DataFrame
    ) -> pd.Series:
        """Optimize weights using posterior returns."""
        # Unconstrained MVO: w* = (1/λ) Σ^-1 μ
        cov_inv = np.linalg.inv(cov.values + np.eye(len(cov)) * 1e-8)
        weights = cov_inv @ expected_returns.values / self.risk_aversion

        # Rescale to sum to 1 and enforce long-only
        weights = np.maximum(weights, 0)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)

        return pd.Series(weights, index=cov.columns)
