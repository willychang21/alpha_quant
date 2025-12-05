import sys
import os
import numpy as np
import pandas as pd
import pytest

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quant.risk.var import calculate_component_var, get_risk_contribution_report

def test_component_var():
    print("\nTesting Component VaR...")
    
    # 1. Setup Dummy Data
    # Asset A: Low Vol (10%)
    # Asset B: High Vol (30%)
    # Correlation: 0.0
    weights = np.array([0.5, 0.5])
    vol_a = 0.10
    vol_b = 0.30
    cov_matrix = np.diag([vol_a**2, vol_b**2])
    
    # 2. Calculate VaR
    p_var, m_var, c_var = calculate_component_var(weights, cov_matrix, alpha=0.05, portfolio_value=10000)
    
    print(f"Portfolio VaR: ${p_var:.2f}")
    print(f"Component VaR: {c_var}")
    
    # 3. Verify Sum
    # Sum(CVaR) should equal PVaR
    assert abs(np.sum(c_var) - p_var) < 1e-5
    
    # 4. Verify Contribution
    # Asset B (High Vol) should contribute more risk
    # Var_p = wA^2 varA + wB^2 varB = 0.25*0.01 + 0.25*0.09 = 0.0025 + 0.0225 = 0.025
    # Vol_p = sqrt(0.025) = 0.158
    # MVaR_A = Z * (wA varA) / vol_p = Z * (0.5 * 0.01) / 0.158
    # MVaR_B = Z * (wB varB) / vol_p = Z * (0.5 * 0.09) / 0.158
    # Ratio MVaR_B / MVaR_A = 9.0
    
    ratio = c_var[1] / c_var[0]
    print(f"Risk Contribution Ratio (B/A): {ratio:.2f}")
    
    assert ratio > 8.0
    
    print("✅ Component VaR Passed")

def test_risk_report():
    print("\nTesting Risk Report...")
    
    weights = pd.Series({'AAPL': 0.6, 'TSLA': 0.4})
    cov = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]], 
        index=['AAPL', 'TSLA'], 
        columns=['AAPL', 'TSLA']
    )
    
    report = get_risk_contribution_report(weights, cov)
    print("\nRisk Report:\n", report)
    
    assert 'Component VaR' in report.columns
    assert '% Contribution' in report.columns
    assert report['% Contribution'].sum() > 0.99
    
    print("✅ Risk Report Passed")

if __name__ == "__main__":
    test_component_var()
    test_risk_report()
