"""Hierarchical Risk Parity (HRP) Optimizer Plugin.

Academic reference: Lopez de Prado (2016)
"Building Diversified Portfolios that Outperform Out of Sample"

Robust alternative to Mean-Variance Optimization.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from quant.core.interfaces import OptimizerBase, PluginMetadata
from quant.core.registry import register_optimizer

logger = logging.getLogger(__name__)


@register_optimizer("HRP")
class HRPOptimizer(OptimizerBase):
    """Hierarchical Risk Parity Portfolio Optimizer.

    Unlike MVO, HRP:
    - Does not require matrix inversion (stable with multicollinearity)
    - Accounts for hierarchical relationships between assets
    - More stable out of sample

    Attributes:
        params: Configuration parameters.
        linkage_method: Clustering method for hierarchy.
    """

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        """Initialize HRP optimizer.

        Args:
            params: Configuration dict with optional keys:
                - linkage_method: Clustering method (default: 'single')
                - risk_measure: 'variance' or 'mad' (default: 'variance')
        """
        self.params = params or {}
        self.linkage_method = self.params.get("linkage_method", "single")
        self.risk_measure = self.params.get("risk_measure", "variance")

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="HRP",
            description="Hierarchical Risk Parity - cluster-based diversification",
            version="1.0.0",
            author="DCA Quant",
            category="optimization",
            parameters={
                "linkage_method": "Clustering method: single, complete, average, ward",
                "risk_measure": "Risk measure: variance or mad",
            },
        )

    def optimize(
        self,
        returns: pd.DataFrame,
        cov: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.Series:
        """Run HRP optimization.

        Args:
            returns: DataFrame of asset returns (Date x Ticker).
            cov: Covariance matrix.

        Returns:
            Series of weights indexed by ticker.
        """
        try:
            # Calculate correlation matrix
            if returns is not None and not returns.empty:
                corr = returns.corr()
                if cov is None or cov.empty:
                    cov = returns.cov()
            else:
                corr = cov.copy()
                # Convert covariance to correlation
                std = np.sqrt(np.diag(cov))
                corr = cov / np.outer(std, std)

            # Get distance matrix
            dist = self._get_distance_matrix(corr)

            # Hierarchical clustering
            dist_condensed = squareform(dist, checks=False)

            # Handle numerical issues
            dist_condensed = np.nan_to_num(dist_condensed, nan=0.0)
            dist_condensed = np.maximum(dist_condensed, 0.0)

            link = linkage(dist_condensed, method=self.linkage_method)

            # Get quasi-diagonal ordering
            sort_idx = self._get_quasi_diag(link)
            sorted_tickers = [cov.columns[i] for i in sort_idx]

            # Reorder covariance matrix
            cov_sorted = cov.loc[sorted_tickers, sorted_tickers]

            # Recursive bisection
            weights = self._recursive_bisection(cov_sorted)

            return weights

        except Exception as e:
            logger.error(f"HRP optimization failed: {e}")
            # Return equal weights as fallback
            n = len(cov)
            return pd.Series(1.0 / n, index=cov.columns)

    def _get_distance_matrix(self, corr: pd.DataFrame) -> pd.DataFrame:
        """Convert correlation matrix to distance matrix."""
        # Ensure valid correlation values
        corr_clipped = np.clip(corr.values, -1.0, 1.0)
        dist = np.sqrt((1 - corr_clipped) / 2)
        return pd.DataFrame(dist, index=corr.index, columns=corr.columns)

    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        """Get quasi-diagonal ordering from linkage."""
        return list(leaves_list(link))

    def _recursive_bisection(self, cov: pd.DataFrame) -> pd.Series:
        """Recursive bisection allocation."""
        weights = pd.Series(1.0, index=cov.index)
        cluster_items = [list(cov.index)]

        while cluster_items:
            cluster = cluster_items.pop()

            if len(cluster) == 1:
                continue

            # Split cluster in half
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]

            # Get cluster variances
            left_var = self._get_cluster_variance(cov, left)
            right_var = self._get_cluster_variance(cov, right)

            # Allocate inversely proportional to variance
            alpha = 1 - left_var / (left_var + right_var) if (left_var + right_var) > 0 else 0.5

            weights[left] *= alpha
            weights[right] *= 1 - alpha

            # Add sub-clusters for further processing
            if len(left) > 1:
                cluster_items.append(left)
            if len(right) > 1:
                cluster_items.append(right)

        return weights

    def _get_cluster_variance(self, cov: pd.DataFrame, cluster: List[str]) -> float:
        """Get variance of a cluster using inverse variance weights."""
        sub_cov = cov.loc[cluster, cluster]
        ivp = 1 / np.diag(sub_cov)
        ivp /= ivp.sum()
        return float(ivp @ sub_cov @ ivp)
