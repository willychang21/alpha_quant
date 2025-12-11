import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def mock_nvda_data():
    """
    Mock data for NVDA (High Growth Tech) - ANNUALIZED
    """
    info = {
        'symbol': 'NVDA',
        'shortName': 'NVIDIA Corp',
        'sector': 'Technology',
        'industry': 'Semiconductors',
        'currentPrice': 135.0,
        'marketCap': 3300000000000,
        'sharesOutstanding': 24500000000,
        'beta': 1.3,
        'dividendYield': 0.0003,
        'targetMeanPrice': 150.0,
        'targetLowPrice': 110.0,
        'targetHighPrice': 200.0,
        'numberOfAnalystOpinions': 40,
        'earningsGrowth': 1.0, # 100%
        'revenueGrowth': 1.0,
        'freeCashflow': 50000000000, # Annualized $50B
        'ebitda': 60000000000,
        'totalCash': 30000000000,
        'totalDebt': 10000000000,
        'netDebt': -20000000000,
    }
    
    # Create Mock Financials (Annual)
    dates = pd.date_range(end=datetime.now(), periods=4, freq='Y')
    
    income = pd.DataFrame({
        'Total Revenue': [100e9, 50e9, 25e9, 15e9],
        'Net Income': [50e9, 25e9, 10e9, 5e9],
        'EBITDA': [60e9, 30e9, 12e9, 6e9]
    }, index=dates).T
    
    balance = pd.DataFrame({
        'Total Assets': [100e9, 80e9, 60e9, 50e9],
        'Total Liab': [40e9, 30e9, 20e9, 15e9],
        'Total Stockholder Equity': [60e9, 50e9, 40e9, 35e9],
        'Cash And Cash Equivalents': [30e9, 20e9, 10e9, 5e9],
        'Total Debt': [10e9, 8e9, 5e9, 2e9],
        'Ordinary Shares Number': [24.5e9, 24.5e9, 24.5e9, 24.5e9]
    }, index=dates).T
    
    cashflow = pd.DataFrame({
        'Operating Cash Flow': [55e9, 28e9, 12e9, 7e9],
        'Capital Expenditure': [-5e9, -3e9, -2e9, -1e9],
        'Free Cash Flow': [50e9, 25e9, 10e9, 6e9]
    }, index=dates).T
    
    return {
        'ticker': 'NVDA',
        'info': info,
        'income': income,
        'balance': balance,
        'cashflow': cashflow
    }

@pytest.fixture
def mock_quarterly_data():
    """
    Mock data representing a single quarter (to test annualization logic)
    """
    info = {
        'symbol': 'TEST_Q',
        'sector': 'Technology',
        'currentPrice': 100.0,
        'sharesOutstanding': 1e9,
        'marketCap': 100e9, # Added marketCap
        'beta': 1.0,
    }
    
    # Only 1 column (Quarterly)
    dates = pd.date_range(end=datetime.now(), periods=1, freq='Q')
    
    income = pd.DataFrame({
        'Total Revenue': [25e9], # $25B Quarter -> $100B Annual
        'Net Income': [10e9],
        'EBITDA': [12e9]
    }, index=dates).T
    
    cashflow = pd.DataFrame({
        'Operating Cash Flow': [12e9],
        'Capital Expenditure': [-2e9],
        'Free Cash Flow': [10e9] # $10B Quarter -> $40B Annual
    }, index=dates).T
    
    balance = pd.DataFrame({
        'Total Debt': [5e9],
        'Cash And Cash Equivalents': [10e9]
    }, index=dates).T
    
    return {
        'ticker': 'TEST_Q',
        'info': info,
        'income': income,
        'balance': balance,
        'cashflow': cashflow
    }

@pytest.fixture
def mock_jpm_data():
    """
    Mock data for JPM (Bank - DDM Model)
    """
    info = {
        'symbol': 'JPM',
        'sector': 'Financial Services',
        'currentPrice': 200.0,
        'sharesOutstanding': 2.9e9,
        'beta': 1.1,
        'dividendRate': 4.0,
        'dividendYield': 0.02,
        'payoutRatio': 0.3,
        'returnOnEquity': 0.15,
        'bookValue': 100.0,
        'targetMeanPrice': 210.0,
        'numberOfAnalystOpinions': 20,
    }
    
    # Financials not strictly needed for DDM logic in our model (uses info), but good to have
    return {
        'ticker': 'JPM',
        'info': info,
        'income': None,
        'balance': None,
        'cashflow': None
    }
