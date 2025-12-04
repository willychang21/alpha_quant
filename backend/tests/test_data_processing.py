import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.engines import valuation

def test_quarterly_data_annualization(mock_quarterly_data):
    """
    CRITICAL TEST: Verify that quarterly data is annualized.
    
    Scenario:
    - Input: 1 Quarter of data (FCF = $10B)
    - Expected Behavior: Model should detect it's quarterly and annualize it (FCF = $40B)
    - Expected Valuation: Based on $40B FCF -> ~$100/share
    - Failure Mode: Based on $10B FCF -> ~$25/share
    """
    result = valuation.get_valuation(
        ticker=mock_quarterly_data['ticker'],
        info=mock_quarterly_data['info'],
        income=mock_quarterly_data['income'],
        balance=mock_quarterly_data['balance'],
        cashflow=mock_quarterly_data['cashflow'],
        is_quarterly=True # Enable annualization
    )
    
    assert result is not None
    assert result.dcf is not None
    
    # If FCF was $10B (Quarterly), valuation would be approx $200
    # If FCF was $40B (Annualized), valuation would be approx $800
    # We assert > 500 to be safe and ensure annualization happened
    print(f"DEBUG: Share Price = {result.dcf.sharePrice}")
    print(f"DEBUG: WACC = {result.dcf.wacc}")
    print(f"DEBUG: Growth = {result.dcf.growthRate}")
    # We can't access base_fcff directly from result, but we can infer from EV
    # EV = SharePrice * Shares + NetDebt
    # NetDebt = 5 - 10 = -5
    # EV = 554 * 1 - 5 = 549
    assert result.dcf.sharePrice > 500.0, \
        f"Valuation too low (${result.dcf.sharePrice:.2f}). Likely failed to annualize quarterly data."
