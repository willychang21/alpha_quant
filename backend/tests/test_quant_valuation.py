import sys
import os
import pandas as pd
import logging

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from quant.valuation.orchestrator import ValuationOrchestrator
from quant.valuation.models.base import ValuationContext

def test_dcf_valuation():
    print("Testing DCF Valuation...")
    
    # Mock Data
    info = {
        'sector': 'Technology',
        'sharesOutstanding': 1000000,
        'currentPrice': 150.0,
        'beta': 1.2,
        'marketCap': 150000000,
        'totalDebt': 50000000,
        'totalCash': 10000000,
        'revenueGrowth': 0.10,
        'earningsGrowth': 0.12
    }
    
    income = pd.DataFrame({
        'Total Revenue': [100000000],
        'EBITDA': [30000000],
        'Net Income': [20000000],
        'Interest Expense': [-2000000],
        'Tax Provision': [5000000],
        'Pretax Income': [25000000]
    }, index=[pd.Timestamp.now()])
    
    balance = pd.DataFrame({
        'Total Debt': [50000000],
        'Cash And Cash Equivalents': [10000000]
    }, index=[pd.Timestamp.now()])
    
    cashflow = pd.DataFrame({
        'Total Cash From Operating Activities': [25000000],
        'Capital Expenditure': [-5000000],
        'Free Cash Flow': [20000000]
    }, index=[pd.Timestamp.now()])
    
    data = {
        'info': info,
        'income_stmt': income,
        'balance_sheet': balance,
        'cashflow': cashflow
    }
    
    orchestrator = ValuationOrchestrator()
    result = orchestrator.get_valuation('TEST', data)
    
    if result:
        print(f"Valuation Result: {result.fair_value}")
        print(f"Model: {result.model_name}")
        print(f"Details: {result.details}")
        assert result.fair_value > 0
        assert "DCF" in result.model_name
        print("DCF Test Passed!")
    else:
        print("Valuation returned None")
        print("DCF Test Failed!")

def test_reit_valuation():
    print("\nTesting REIT Valuation...")
    
    info = {
        'sector': 'Real Estate',
        'sharesOutstanding': 1000000,
        'currentPrice': 100.0,
        'revenueGrowth': 0.08
    }
    
    income = pd.DataFrame({
        'Net Income': [5000000],
        'Depreciation And Amortization': [10000000]
    }, index=[pd.Timestamp.now()])
    
    data = {
        'info': info,
        'income_stmt': income,
        'balance_sheet': None,
        'cashflow': None
    }
    
    orchestrator = ValuationOrchestrator()
    result = orchestrator.get_valuation('REIT_TEST', data)
    
    if result:
        print(f"Valuation Result: {result.fair_value}")
        print(f"Model: {result.model_name}")
        assert result.fair_value > 0
        assert "REIT" in result.model_name
        print("REIT Test Passed!")
    else:
        print("Valuation returned None")
        print("REIT Test Failed!")

if __name__ == "__main__":
    test_dcf_valuation()
    test_reit_valuation()
