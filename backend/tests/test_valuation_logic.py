import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.engines import valuation
from app.engines.valuation import core
from app.domain import schemas

def test_nvda_valuation_logic(mock_nvda_data):
    """
    Regression Test: Ensure NVDA valuation is reasonable given high growth inputs.
    Prevents the $49 vs $192 issue.
    """
    result = valuation.get_valuation(
        ticker=mock_nvda_data['ticker'],
        info=mock_nvda_data['info'],
        income=mock_nvda_data['income'],
        balance=mock_nvda_data['balance'],
        cashflow=mock_nvda_data['cashflow']
    )
    
    assert result is not None
    assert result.dcf is not None
    
    # Valuation should be > $100 given the inputs (Annualized FCF $50B, Growth 100%)
    # If it falls below $100, it means the model is overly conservative or data is wrong
    assert result.dcf.sharePrice > 100.0, f"NVDA Valuation too low: ${result.dcf.sharePrice}"
    
    # Check WACC
    assert 0.08 < result.dcf.wacc < 0.15, f"WACC out of reasonable range: {result.dcf.wacc}"

def test_jpm_valuation_logic(mock_jpm_data):
    """
    Test DDM model selection for Banks
    """
    result = valuation.get_valuation(
        ticker=mock_jpm_data['ticker'],
        info=mock_jpm_data['info'],
        income=mock_jpm_data['income'],
        balance=mock_jpm_data['balance'],
        cashflow=mock_jpm_data['cashflow']
    )
    
    assert result is not None
    assert result.ddm is not None
    assert result.dcf is None # Should NOT use DCF for banks
    
    # Valuation check
    assert result.ddm.fairValue > 150.0, f"JPM Valuation too low: ${result.ddm.fairValue}"

def test_wacc_calculation():
    """
    Unit test for WACC function
    """
    # Mock inputs
    beta = 1.2
    risk_free = 0.04
    market_return = 0.10
    total_debt = 1000
    market_cap = 9000
    tax_rate = 0.21
    
    wacc_res = core.calculate_wacc(
        info={
            'beta': beta,
            'marketCap': market_cap,
            'totalDebt': total_debt
        },
        balance=None, # Not needed if totalDebt is in info
        income=None,   # Not needed if we don't check interest expense details here (defaults used)
        risk_free_rate=risk_free,
        market_risk_premium=market_return - risk_free # MRP = 0.10 - 0.04 = 0.06
    )
    
    wacc = wacc_res['wacc']
    
    # Expected:
    # Ke = 0.04 + 1.2 * (0.10 - 0.04) = 0.112 (11.2%)
    # Kd = 50/1000 = 0.05 (5%)
    # Kd_after_tax = 0.05 * (1 - 0.21) = 0.0395
    # WACC = (0.9 * 0.112) + (0.1 * 0.0395) = 0.1008 + 0.00395 = 0.10475
    
    assert 0.10 < wacc < 0.11, f"WACC calculation incorrect: {wacc}"
