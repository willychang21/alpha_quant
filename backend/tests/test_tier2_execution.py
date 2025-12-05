import sys
import os
import numpy as np
import pandas as pd
import pytest

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quant.execution.algo import VWAPExecution

def test_vwap_schedule():
    print("\nTesting VWAP Schedule...")
    
    # 1. Setup
    algo = VWAPExecution() # Use default profile
    total_shares = 10000
    
    # 2. Generate Schedule
    schedule = algo.generate_schedule(total_shares)
    
    print("Execution Schedule (First 5 bins):")
    print(schedule.head())
    print("Execution Schedule (Last 5 bins):")
    print(schedule.tail())
    
    # 3. Verify Sum
    assert schedule['Shares'].sum() == total_shares
    
    # 4. Verify Shape
    # Morning (bin 0) should be higher than Lunch (bin 5)
    assert schedule.iloc[0]['Shares'] > schedule.iloc[5]['Shares']
    # Close (last bin) should be higher than Lunch
    assert schedule.iloc[-1]['Shares'] > schedule.iloc[5]['Shares']
    
    print("✅ VWAP Schedule Passed")

def test_impact_cost():
    print("\nTesting Impact Cost Estimation...")
    
    algo = VWAPExecution()
    
    # Case 1: Small order (1% of ADV)
    # Vol = 2% (0.02)
    cost_small = algo.estimate_impact_cost(
        total_shares=1000, 
        daily_volume=100000, 
        volatility=0.02
    )
    
    # Cost = 1.0 * 0.02 * sqrt(0.01) = 0.02 * 0.1 = 0.002 = 20 bps
    print(f"Cost (Small Order): {cost_small:.2f} bps")
    assert 19 < cost_small < 21
    
    # Case 2: Large order (10% of ADV)
    cost_large = algo.estimate_impact_cost(
        total_shares=10000, 
        daily_volume=100000, 
        volatility=0.02
    )
    
    # Cost = 1.0 * 0.02 * sqrt(0.1) = 0.02 * 0.316 = 0.00632 = 63.2 bps
    print(f"Cost (Large Order): {cost_large:.2f} bps")
    assert cost_large > cost_small
    
    print("✅ Impact Cost Passed")

if __name__ == "__main__":
    test_vwap_schedule()
    test_impact_cost()
