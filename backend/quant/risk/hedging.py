import numpy as np
import pandas as pd
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)

def black_scholes_put(S, K, T, r, sigma):
    """
    Calculates the price of a European Put Option using Black-Scholes.
    
    Args:
        S: Spot Price
        K: Strike Price
        T: Time to Maturity (years)
        r: Risk-free rate
        sigma: Volatility
        
    Returns:
        float: Put Price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def calculate_tail_hedge_cost(
    portfolio_value: float,
    spot_price: float,
    volatility: float,
    hedge_fraction: float = 1.0,
    otm_pct: float = 0.20, # 20% OTM
    duration_months: int = 3,
    risk_free_rate: float = 0.04
) -> dict:
    """
    Calculates the cost of a tail hedge using OTM Puts.
    
    Args:
        portfolio_value (float): Total value to hedge.
        spot_price (float): Current price of the underlying (e.g., SPY).
        volatility (float): Annualized volatility of the underlying.
        hedge_fraction (float): Fraction of portfolio to hedge (e.g., 1.0 for 100%).
        otm_pct (float): How far OTM the puts are (e.g., 0.20 for 20% drop protection).
        duration_months (int): Duration of the options.
        risk_free_rate (float): Risk-free rate.
        
    Returns:
        dict: {
            'put_price': Price of one put option,
            'contracts_needed': Number of contracts,
            'total_cost': Total cost of hedge,
            'cost_bps': Cost in basis points of portfolio value
        }
    """
    T = duration_months / 12.0
    K = spot_price * (1 - otm_pct)
    
    # Price per share
    put_price = black_scholes_put(spot_price, K, T, risk_free_rate, volatility)
    
    # Number of shares to hedge
    shares_to_hedge = (portfolio_value * hedge_fraction) / spot_price
    
    # Total cost
    total_cost = shares_to_hedge * put_price
    
    cost_bps = (total_cost / portfolio_value) * 10000
    
    return {
        'put_price': put_price,
        'strike': K,
        'shares_hedged': shares_to_hedge,
        'total_cost': total_cost,
        'cost_bps': cost_bps,
        'annualized_cost_bps': cost_bps * (12 / duration_months)
    }

def optimize_hedge_ratio(
    portfolio_beta: float,
    market_vol: float,
    target_drawdown: float = 0.15
) -> float:
    """
    Simple heuristic to optimize hedge ratio based on Beta.
    If portfolio is high beta, we need more hedging.
    
    Args:
        portfolio_beta (float): Portfolio Beta to SPY.
        market_vol (float): SPY Volatility.
        target_drawdown (float): Max acceptable drawdown.
        
    Returns:
        float: Recommended hedge ratio (0.0 to 1.0+).
    """
    # If Beta is high, we expect portfolio to drop Beta * MarketDrop.
    # We want to cap loss at target_drawdown.
    
    # This is a complex optimization usually, but here's a heuristic:
    # If Beta > 1, we need full hedging (1.0).
    # If Beta < 0.5, we might not need hedging.
    
    if portfolio_beta > 1.2:
        return 1.0
    elif portfolio_beta > 0.8:
        return 0.5
    else:
        return 0.0
