"""
Hierarchical Risk Parity (HRP) Optimizer

Based on Lopez de Prado (2016): A robust alternative to Mean-Variance Optimization
that avoids matrix inversion and is more stable with correlated assets.

Steps:
1. Tree Clustering: Build dendrogram from correlation matrix
2. Quasi-Diagonalization: Reorder covariance matrix
3. Recursive Bisection: Allocate using inverse variance
"""
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class HRPOptimizer:
    """
    Hierarchical Risk Parity Portfolio Optimizer.
    
    Unlike MVO, HRP:
    - Does not require matrix inversion (stable with multicollinearity)
    - Accounts for hierarchical relationships between assets
    - Provides more diversified weights
    """
    
    def __init__(self, linkage_method: str = 'single'):
        """
        Args:
            linkage_method: Clustering method ('single', 'complete', 'average', 'ward')
        """
        self.linkage_method = linkage_method
        
    def optimize(
        self, 
        returns: pd.DataFrame,
        cov_matrix: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Run HRP optimization.
        
        Args:
            returns: DataFrame of asset returns (Date x Ticker)
            cov_matrix: Optional pre-computed covariance matrix
            
        Returns:
            Dict of ticker -> weight
        """
        if returns.empty or returns.shape[1] < 2:
            logger.warning("HRP requires at least 2 assets")
            return {}
        
        try:
            # 1. Compute correlation and covariance matrices
            if cov_matrix is None:
                cov_matrix = returns.cov()
            
            corr_matrix = returns.corr()
            
            # Handle NaN/Inf
            cov_matrix = cov_matrix.fillna(0)
            corr_matrix = corr_matrix.fillna(0)
            
            # 2. Tree Clustering
            dist_matrix = self._get_distance_matrix(corr_matrix)
            link = linkage(squareform(dist_matrix), method=self.linkage_method)
            
            # 3. Quasi-Diagonalization (Seriation)
            sorted_idx = self._get_quasi_diag(link)
            sorted_tickers = [corr_matrix.columns[i] for i in sorted_idx]
            
            # Reorder covariance matrix
            cov_sorted = cov_matrix.loc[sorted_tickers, sorted_tickers]
            
            # 4. Recursive Bisection
            weights = self._recursive_bisection(cov_sorted)
            
            # Log top allocations
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            logger.info("HRP Optimization Complete. Top 5 allocations:")
            for ticker, weight in sorted_weights[:5]:
                logger.info(f"  {ticker}: {weight:.2%}")
            
            return weights
            
        except Exception as e:
            logger.error(f"HRP optimization failed: {e}")
            return {}
    
    def _get_distance_matrix(self, corr: pd.DataFrame) -> np.ndarray:
        """Convert correlation matrix to distance matrix."""
        # Distance = sqrt(2 * (1 - correlation))
        dist = ((1 - corr) / 2) ** 0.5
        return dist.values
    
    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        """
        Quasi-diagonalization: reorder assets to place similar assets together.
        Returns indices of assets in optimal order.
        """
        return list(leaves_list(link))
    
    def _recursive_bisection(self, cov: pd.DataFrame) -> Dict[str, float]:
        """
        Recursive bisection allocation.
        
        Splits the universe in half, allocates based on inverse variance,
        then recurses on each half.
        """
        weights = pd.Series(1.0, index=cov.index)
        cluster_items = [cov.index.tolist()]
        
        while len(cluster_items) > 0:
            # Split clusters
            cluster_items = [
                item[j:k] 
                for item in cluster_items 
                for j, k in ((0, len(item) // 2), (len(item) // 2, len(item))) 
                if len(item) > 1
            ]
            
            # Allocate based on inverse variance
            for i in range(0, len(cluster_items), 2):
                if i + 1 >= len(cluster_items):
                    break
                    
                cluster_0 = cluster_items[i]
                cluster_1 = cluster_items[i + 1]
                
                # Variance of each cluster
                var_0 = self._get_cluster_variance(cov, cluster_0)
                var_1 = self._get_cluster_variance(cov, cluster_1)
                
                # Inverse variance allocation
                alpha = 1 - var_0 / (var_0 + var_1) if (var_0 + var_1) > 0 else 0.5
                
                weights[cluster_0] *= alpha
                weights[cluster_1] *= (1 - alpha)
        
        return weights.to_dict()
    
    def _get_cluster_variance(self, cov: pd.DataFrame, cluster: List[str]) -> float:
        """Get variance of a cluster using inverse variance weights within cluster."""
        cov_cluster = cov.loc[cluster, cluster]
        
        # Within-cluster: use inverse variance (risk parity)
        ivp = 1 / np.diag(cov_cluster)
        ivp /= ivp.sum()
        
        # Cluster variance = w' * Sigma * w
        cluster_var = np.dot(ivp, np.dot(cov_cluster.values, ivp))
        
        return cluster_var


class BlackLittermanModel:
    """
    Black-Litterman Model for combining market equilibrium with alpha views.
    
    E[R] = [(τΣ)^-1 + P'Ω^-1 P]^-1 * [(τΣ)^-1 π + P'Ω^-1 Q]
    
    Where:
    - π: Market equilibrium returns
    - Q: Views on expected returns
    - P: Picking matrix (which assets the views apply to)
    - Ω: Uncertainty matrix (confidence in views)
    - τ: Scaling factor (typically 0.05)
    """
    
    def __init__(self, tau: float = 0.05, risk_aversion: float = 2.5):
        """
        Args:
            tau: Scaling factor for uncertainty in equilibrium
            risk_aversion: Market risk aversion coefficient (delta)
        """
        self.tau = tau
        self.risk_aversion = risk_aversion
    
    def get_equilibrium_returns(
        self, 
        market_caps: pd.Series, 
        cov_matrix: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate implied equilibrium returns from market caps.
        
        π = δ * Σ * w_mkt
        """
        # Market cap weights
        w_mkt = market_caps / market_caps.sum()
        
        # Implied returns
        pi = self.risk_aversion * cov_matrix.dot(w_mkt)
        
        return pi
    
    def alpha_to_views(
        self, 
        z_scores: Dict[str, float],
        volatilities: Dict[str, float],
        ic: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert Z-scores to Black-Litterman views using Grinold-Kahn mapping.
        
        α_i = IC × σ_i × Z_i
        
        Args:
            z_scores: Dict of ticker -> composite Z-score
            volatilities: Dict of ticker -> annualized volatility
            ic: Information Coefficient (skill of the signal, ~0.05 is good)
            
        Returns:
            Q (views), P (picking matrix), Omega (uncertainty)
        """
        tickers = list(z_scores.keys())
        n = len(tickers)
        
        # Q: Expected active returns
        Q = np.array([
            ic * volatilities.get(t, 0.25) * z_scores[t] 
            for t in tickers
        ])
        
        # P: Identity matrix (one view per asset)
        P = np.eye(n)
        
        # Omega: Uncertainty = variance of view
        # Higher |Z| = more confident, but we also scale by vol
        # Omega_ii = (σ_i)^2 / |Z_i| for non-zero Z, else high uncertainty
        omega_diag = [
            (volatilities.get(t, 0.25) ** 2) / max(abs(z_scores[t]), 0.5)
            for t in tickers
        ]
        Omega = np.diag(omega_diag)
        
        return Q, P, Omega
    
    def posterior_returns(
        self,
        pi: pd.Series,
        cov_matrix: pd.DataFrame,
        Q: np.ndarray,
        P: np.ndarray,
        Omega: np.ndarray
    ) -> pd.Series:
        """
        Calculate posterior expected returns using Black-Litterman formula.
        """
        tau_sigma = self.tau * cov_matrix.values
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(Omega)
        
        # Posterior precision
        post_precision = tau_sigma_inv + P.T @ omega_inv @ P
        post_cov = np.linalg.inv(post_precision)
        
        # Posterior mean
        post_mean = post_cov @ (tau_sigma_inv @ pi.values + P.T @ omega_inv @ Q)
        
        return pd.Series(post_mean, index=pi.index)
    
    def optimize(
        self,
        cov_matrix: pd.DataFrame,
        market_caps: pd.Series,
        z_scores: Dict[str, float],
        volatilities: Dict[str, float],
        ic: float = 0.05
    ) -> Dict[str, float]:
        """
        Full Black-Litterman optimization.
        
        Returns optimal weights.
        """
        try:
            # 1. Equilibrium returns
            pi = self.get_equilibrium_returns(market_caps, cov_matrix)
            
            # 2. Convert signals to views
            Q, P, Omega = self.alpha_to_views(z_scores, volatilities, ic)
            
            # 3. Posterior returns
            E_R = self.posterior_returns(pi, cov_matrix, Q, P, Omega)
            
            # 4. Optimal weights (closed form for unconstrained MVO)
            # w* = (δ * Σ)^-1 * E[R]
            sigma_inv = np.linalg.inv(self.risk_aversion * cov_matrix.values)
            w_raw = sigma_inv @ E_R.values
            
            # Normalize to sum to 1 and long-only
            w_raw = np.maximum(w_raw, 0)  # Long only
            w_sum = w_raw.sum()
            if w_sum > 0:
                w_raw /= w_sum
            
            weights = dict(zip(cov_matrix.index, w_raw))
            
            logger.info("Black-Litterman Optimization Complete.")
            
            return weights
            
        except Exception as e:
            logger.error(f"Black-Litterman optimization failed: {e}")
            return {}
