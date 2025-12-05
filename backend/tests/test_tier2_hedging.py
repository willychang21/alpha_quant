import sys
import os
import pytest

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quant.risk.hedging import calculate_tail_hedge_cost, optimize_hedge_ratio

def test_tail_hedge_cost():
    print("\nTesting Tail Hedge Cost...")
    
    # 1. Setup
    portfolio_value = 1_000_000 # $1M
    spot_price = 400.0 # SPY
    volatility = 0.20 # 20% VIX
    
    # 2. Calculate Cost (20% OTM, 3 months)
    # Strike = 320.
    cost = calculate_tail_hedge_cost(
        portfolio_value, 
        spot_price, 
        volatility, 
        otm_pct=0.20, 
        duration_months=3
    )
    
    print("Hedge Cost Details:", cost)
    
    # 3. Verify
    # 20% OTM put for 3 months with 20% vol should be cheap.
    # Strike=320. Spot=400.
    # d1 is large positive. N(-d2) is small.
    
    # Expected cost should be low, maybe < 50 bps?
    # Let's check annualized cost.
    
    assert cost['total_cost'] > 0
    assert cost['cost_bps'] < 100 # Should be cheap (< 1%)
    
    print(f"Cost (bps): {cost['cost_bps']:.2f}")
    print(f"Annualized Cost (bps): {cost['annualized_cost_bps']:.2f}")
    
    print("✅ Tail Hedge Cost Passed")

def test_hedge_optimization():
    print("\nTesting Hedge Optimization...")
    
    # High Beta -> Full Hedge
    ratio_high = optimize_hedge_ratio(portfolio_beta=1.5, market_vol=0.20)
    assert ratio_high == 1.0
    
    # Low Beta -> No Hedge
    ratio_low = optimize_hedge_ratio(portfolio_beta=0.4, market_vol=0.20)
    assert ratio_low == 0.0
    
    print("✅ Hedge Optimization Passed")

if __name__ == "__main__":
    test_tail_hedge_cost()
    test_hedge_optimization()
