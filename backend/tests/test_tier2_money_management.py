import sys
import os
import numpy as np
import pandas as pd
import pytest

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quant.portfolio.kelly import optimize_multivariate_kelly
from quant.portfolio.risk_control import apply_vol_targeting

def test_kelly_optimization():
    print("\nTesting Multivariate Kelly Optimization...")
    
    # 1. Setup Dummy Data
    # 3 Assets: High Return/High Vol, Low Return/Low Vol, Negative Return
    expected_returns = np.array([0.20, 0.05, -0.10]) # Annualized excess return
    
    # Covariance Matrix (Uncorrelated for simplicity)
    # Asset 1: 20% vol -> Var = 0.04
    # Asset 2: 5% vol -> Var = 0.0025
    # Asset 3: 15% vol -> Var = 0.0225
    cov_matrix = np.diag([0.04, 0.0025, 0.0225])
    
    # 2. Run Optimization (Full Kelly)
    # Theoretical Uncorrelated Kelly: w_i = mu_i / sigma_i^2
    # w1 = 0.20 / 0.04 = 5.0
    # w2 = 0.05 / 0.0025 = 20.0
    # w3 = -0.10 / 0.0225 = -4.44 (Should be 0 if long only)
    
    # We use max_leverage=30.0 to allow the theoretical unconstrained weights (sum=25)
    weights = optimize_multivariate_kelly(
        expected_returns, 
        cov_matrix, 
        max_leverage=30.0, 
        fractional_kelly=1.0 
    )
    
    print("Optimal Weights (Unconstrained):", weights)
    
    # Verify constraints
    assert np.all(weights >= -0.001) # Long only (approx)
    assert weights[2] < 0.001 # Negative return asset should be 0
    
    # Verify relative sizing
    # w1 = 5.0, w2 = 20.0. Ratio w2/w1 = 4.0.
    
    if weights[0] > 0.1:
        ratio = weights[1] / weights[0]
        print(f"Ratio w2/w1: {ratio:.2f} (Expected ~4.0)")
        assert 3.5 < ratio < 4.5
    else:
        pytest.fail("Weight 1 is too small")
    
    print("✅ Kelly Optimization Passed")

def test_vol_targeting():
    print("\nTesting Volatility Targeting...")
    
    # 1. Setup Dummy Data
    # Portfolio of 1 asset
    weights = pd.Series({'AAPL': 1.0})
    
    # Prices: Generate a series with known volatility
    # Target Vol = 15%.
    # Case A: Current Vol = 30% (High). Scalar should be 0.5.
    
    np.random.seed(42)
    # Generate 30% annualized vol returns
    daily_vol = 0.30 / np.sqrt(252)
    returns = np.random.normal(0, daily_vol, 100)
    price_series = 100 * (1 + returns).cumprod()
    prices = pd.DataFrame({'AAPL': price_series})
    
    # 2. Run Vol Targeting
    scaled_weights = apply_vol_targeting(
        weights, 
        prices, 
        target_vol=0.15, 
        lookback=20,
        max_leverage=2.0
    )
    
    print("Scaled Weights (High Vol):", scaled_weights)
    
    # Check scalar
    # Since we use random data, it won't be exactly 0.5, but close.
    # Let's calculate actual vol of last 20 days
    actual_vol = pd.Series(returns).ewm(span=20).std().iloc[-1] * np.sqrt(252)
    expected_scalar = 0.15 / actual_vol
    
    print(f"Actual Vol: {actual_vol:.2%}")
    print(f"Expected Scalar: {expected_scalar:.2f}")
    print(f"Result Scalar: {scaled_weights['AAPL']:.2f}")
    
    assert abs(scaled_weights['AAPL'] - expected_scalar) < 0.01
    assert scaled_weights['AAPL'] < 1.0 # Should deleverage
    
    print("✅ Volatility Targeting Passed")

if __name__ == "__main__":
    test_kelly_optimization()
    test_vol_targeting()
