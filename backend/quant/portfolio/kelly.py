import cvxpy as cp
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def optimize_multivariate_kelly(
    expected_returns: np.ndarray, 
    cov_matrix: np.ndarray, 
    risk_free_rate: float = 0.0, 
    max_leverage: float = 1.0, 
    fractional_kelly: float = 0.5,
    # New Constraints (Phase 8)
    sector_mapper: np.ndarray = None, # (n_sectors, n_assets) binary matrix
    sector_limits: np.ndarray = None, # (n_sectors, ) max weights
    beta_vector: np.ndarray = None,   # (n_assets, ) betas
    target_beta: float = None,        # e.g. 0.0 for neutral, 1.0 for market
    beta_tolerance: float = 0.1
) -> np.ndarray:
    """
    Solves the Multivariate Kelly Criterion problem using Convex Optimization.
    
    Objective: Maximize Geometric Growth Rate g(w)
    g(w) approx r + w^T(mu - r) - 0.5 * w^T Sigma w
    
    Args:
        expected_returns (np.array): Vector of expected returns (mu).
        cov_matrix (np.ndarray): Covariance matrix of returns (Sigma).
        risk_free_rate (float): The risk-free rate.
        max_leverage (float): Maximum allowed leverage (sum of absolute weights).
        fractional_kelly (float): Scalar to reduce variance (e.g., 0.5 for Half-Kelly).
        sector_mapper (np.ndarray): Binary matrix mapping assets to sectors.
        sector_limits (np.ndarray): Max weight per sector.
        beta_vector (np.ndarray): Vector of asset betas.
        target_beta (float): Target portfolio beta.
        beta_tolerance (float): Allowed deviation from target beta (+/-).
        
    Returns:
        np.array: Optimal asset weights.
    """
    n_assets = len(expected_returns)
    
    # Excess returns
    mu = expected_returns - risk_free_rate
    
    # Variable: Weights
    w = cp.Variable(n_assets)
    
    # Objective: Maximize expected log-wealth growth (Taylor expansion approx)
    port_return = w @ mu
    port_risk = cp.quad_form(w, cov_matrix)
    
    objective = cp.Maximize(port_return - 0.5 * port_risk)
    
    constraints = [
        cp.sum(cp.abs(w)) <= max_leverage,  # Gross leverage constraint
        w >= 0                              # Long-only constraint
    ]
    
    # Phase 8: Sector Constraints
    if sector_mapper is not None and sector_limits is not None:
        # sector_mapper is (n_sectors, n_assets)
        # w is (n_assets, )
        # sector_weights = sector_mapper @ w -> (n_sectors, )
        sector_weights = sector_mapper @ w
        constraints.append(sector_weights <= sector_limits)
        
    # Phase 8: Beta Constraints
    if beta_vector is not None and target_beta is not None:
        portfolio_beta = w @ beta_vector
        constraints.append(portfolio_beta >= target_beta - beta_tolerance)
        constraints.append(portfolio_beta <= target_beta + beta_tolerance)
    
    problem = cp.Problem(objective, constraints)
    
    try:
        # Use SCS or OSQP. OSQP is often faster for quadratic problems.
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status != 'optimal':
            logger.warning(f"Kelly optimization status: {problem.status}")
            # Fallback to equal weights or zero?
            return np.zeros(n_assets)
            
        # Apply Fractional Kelly Scaling
        optimal_weights = w.value * fractional_kelly
        
        # Clean small weights
        optimal_weights[np.abs(optimal_weights) < 1e-4] = 0.0
        
        return optimal_weights
        
    except Exception as e:
        logger.error(f"Kelly optimization failed: {e}")
        return np.zeros(n_assets)
