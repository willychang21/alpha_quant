import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def apply_vol_targeting(
    weights: pd.Series, 
    prices: pd.DataFrame, 
    target_vol: float = 0.15, 
    lookback: int = 20, 
    max_leverage: float = 2.0
) -> pd.Series:
    """
    Scales portfolio weights based on volatility targeting.
    
    Formula: Scalar_t = Target_Vol / Forecasted_Vol_t
    
    Args:
        weights (pd.Series): Target weights for each asset (sum usually 1.0 or Kelly weights).
        prices (pd.DataFrame): Asset prices series (for calculating portfolio vol).
                               Should include history up to rebalance date.
        target_vol (float): Annualized target volatility (e.g., 0.15 for 15%).
        lookback (int): Window for volatility estimation (e.g., 20 days).
        max_leverage (float): Hard cap on leverage scalar.
        
    Returns:
        pd.Series: Volatility-scaled weights.
    """
    if weights.empty or prices.empty:
        return weights
        
    # 1. Calculate Portfolio Historical Returns
    # We need to estimate the volatility of the *current* portfolio mix.
    # Assuming constant weights over the lookback period (simplification).
    
    # Filter prices for assets in weights
    assets = weights.index.intersection(prices.columns)
    if len(assets) == 0:
        logger.warning("No overlapping assets between weights and prices.")
        return weights
        
    asset_prices = prices[assets]
    asset_returns = asset_prices.pct_change().dropna()
    
    # Calculate Portfolio Returns series
    # R_p = w * R
    port_returns = asset_returns.dot(weights[assets])
    
    # 2. Forecast Volatility (EWMA)
    # We use the last available volatility estimate
    if len(port_returns) < lookback:
        logger.warning("Insufficient history for volatility targeting.")
        return weights
        
    # Annualized Volatility
    rolling_vol = port_returns.ewm(span=lookback).std() * np.sqrt(252)
    current_vol = rolling_vol.iloc[-1]
    
    if pd.isna(current_vol) or current_vol == 0:
        return weights
        
    # 3. Calculate Scalar
    vol_scalar = target_vol / current_vol
    
    # Cap leverage
    vol_scalar = min(vol_scalar, max_leverage)
    
    logger.info(f"Vol Targeting: Current Vol={current_vol:.2%}, Target={target_vol:.2%}, Scalar={vol_scalar:.2f}")
    
    # 4. Apply Scalar
    scaled_weights = weights * vol_scalar
    
    return scaled_weights
