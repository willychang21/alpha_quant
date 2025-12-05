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
    fractional_kelly: float = 0.5
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
        
    Returns:
        np.array: Optimal asset weights.
    """
    n_assets = len(expected_returns)
    
    # Excess returns
    mu = expected_returns - risk_free_rate
    
    # Variable: Weights
    w = cp.Variable(n_assets)
    
    # Objective: Maximize expected log-wealth growth (Taylor expansion approx)
    # Maximize: w^T * mu - 0.5 * w^T * Sigma * w
    # Note: This is equivalent to maximizing the Sharpe Ratio squared if unconstrained?
    # No, Kelly maximizes growth. MVO maximizes utility.
    # Kelly is equivalent to MVO with Risk Aversion (lambda) = 1?
    # Actually, Full Kelly corresponds to maximizing Log Utility.
    # Taylor expansion of Log Utility is Mean - 0.5 * Variance.
    
    port_return = w @ mu
    port_risk = cp.quad_form(w, cov_matrix)
    
    objective = cp.Maximize(port_return - 0.5 * port_risk)
    
    constraints = [
        cp.sum(cp.abs(w)) <= max_leverage,  # Gross leverage constraint
        w >= 0                              # Long-only constraint (optional, but safer for now)
    ]
    
    problem = cp.Problem(objective, constraints)
    
    try:
        # Use SCS or OSQP. OSQP is often faster for quadratic problems.
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status != 'optimal':
            logger.warning(f"Kelly optimization status: {problem.status}")
            # Fallback to equal weights or zero?
            return np.zeros(n_assets)
            
        # Apply Fractional Kelly Scaling
        # We scale the *optimal* weights by the fraction.
        # This reduces the bet size, moving us closer to the risk-free asset (cash).
        optimal_weights = w.value * fractional_kelly
        
        # Clean small weights
        optimal_weights[np.abs(optimal_weights) < 1e-4] = 0.0
        
        return optimal_weights
        
    except Exception as e:
        logger.error(f"Kelly optimization failed: {e}")
        return np.zeros(n_assets)
