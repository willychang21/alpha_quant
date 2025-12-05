import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

def calculate_component_var(
    weights: np.ndarray, 
    cov_matrix: np.ndarray, 
    alpha: float = 0.05,
    portfolio_value: float = 1.0
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calculates the Component Value at Risk (CVaR) for each asset.
    
    Component VaR decomposes the total Portfolio VaR into contributions from each asset.
    Sum(Component VaR) = Portfolio VaR.
    
    Args:
        weights (np.ndarray): Asset weights (1D array).
        cov_matrix (np.ndarray): Covariance matrix of returns (2D array).
        alpha (float): Confidence level (e.g., 0.05 for 95% confidence).
        portfolio_value (float): Total value of the portfolio (to get VaR in $).
        
    Returns:
        Tuple[float, np.ndarray, np.ndarray]:
            - Portfolio VaR (float)
            - Marginal VaR (np.ndarray): Sensitivity of VaR to weight changes.
            - Component VaR (np.ndarray): Contribution of each asset to VaR.
    """
    # 1. Portfolio Volatility
    # sigma_p = sqrt(w' Sigma w)
    port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
    port_vol = np.sqrt(port_var)
    
    # 2. Portfolio VaR (Parametric / Normal)
    # VaR = Z_alpha * sigma_p * Value
    # For alpha=0.05 (95%), Z is approx 1.645
    from scipy.stats import norm
    z_score = norm.ppf(1 - alpha)
    
    port_var_val = z_score * port_vol * portfolio_value
    
    # 3. Marginal VaR
    # dVaR/dw = Z * (Sigma w) / sigma_p
    # This vector tells us how much VaR changes if we increase weight i by 1 unit.
    marginal_var = z_score * np.dot(cov_matrix, weights) / port_vol
    
    # 4. Component VaR
    # CVaR_i = w_i * MVaR_i
    # This is the amount of risk contributed by asset i.
    component_var = weights * marginal_var * portfolio_value
    
    return port_var_val, marginal_var, component_var

def get_risk_contribution_report(
    weights: pd.Series, 
    cov_matrix: pd.DataFrame, 
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Generates a readable risk contribution report.
    
    Args:
        weights (pd.Series): Asset weights with index (Tickers).
        cov_matrix (pd.DataFrame): Covariance matrix with index/columns (Tickers).
        
    Returns:
        pd.DataFrame: Report with columns ['Weight', 'Marginal VaR', 'Component VaR', '% Contribution'].
    """
    # Align weights and covariance
    common = weights.index.intersection(cov_matrix.index)
    w = weights[common].values
    cov = cov_matrix.loc[common, common].values
    
    p_var, m_var, c_var = calculate_component_var(w, cov, alpha)
    
    df = pd.DataFrame(index=common)
    df['Weight'] = w
    df['Marginal VaR'] = m_var
    df['Component VaR'] = c_var
    df['% Contribution'] = c_var / p_var
    
    return df.sort_values('% Contribution', ascending=False)
