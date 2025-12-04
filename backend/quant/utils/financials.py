import pandas as pd
import numpy as np
import logging
import math
import yfinance as yf
from typing import Dict, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

def get_latest(df: pd.DataFrame, key: str) -> Optional[float]:
    """Helper to get the most recent value from a DataFrame row."""
    try:
        if key in df.index:
            # Get the most recent column (usually the first one in yfinance)
            # yfinance columns are dates, sorted descending usually
            rows = df.loc[key]
            
            # Handle duplicate keys (returns DataFrame)
            if isinstance(rows, pd.DataFrame):
                rows = rows.iloc[0] # Take first row
                
            # Now rows should be a Series (values across dates) or Scalar?
            # If df has columns, rows is a Series.
            if isinstance(rows, pd.Series):
                val = rows.iloc[0]
            else:
                val = rows
                
            if pd.isna(val):
                return None
            return float(val)
    except Exception:
        pass
    return None

def sanitize(val: Any) -> float:
    """Helper to sanitize floats for JSON (handle NaN/Inf)."""
    # Handle DataFrame/Series (take first value)
    if isinstance(val, (pd.DataFrame, pd.Series)):
        if val.empty:
            return 0.0
        try:
            val = val.iloc[0]
            if isinstance(val, pd.Series): # Handle DataFrame -> Series -> Scalar
                val = val.iloc[0]
        except Exception:
            return 0.0

    if isinstance(val, (float, int)):
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return float(val)
    return 0.0

def calculate_fcff(info: Dict[str, Any], income: pd.DataFrame, cashflow: pd.DataFrame, tax_rate: float) -> float:
    """
    Calculates Free Cash Flow to Firm (FCFF) - Unlevered Cash Flow.
    FCFF = OCF + Interest(1-t) - CapEx
    """
    try:
        # Method 1: From Cash Flow Statement (Most reliable)
        ocf = get_latest(cashflow, 'Total Cash From Operating Activities') or get_latest(cashflow, 'Operating Cash Flow')
        capex = get_latest(cashflow, 'Capital Expenditure') or 0
        
        # CapEx is usually negative in yfinance
        if capex > 0:
            capex = -capex
        
        # Add back after-tax Interest to convert FCFE → FCFF
        # OCF already has Interest subtracted (it's based on Net Income)
        interest_expense = get_latest(income, 'Interest Expense') or 0
        interest_addback = abs(interest_expense) * (1 - tax_rate)
        
        if ocf is not None:
            fcff = ocf + interest_addback + capex  # capex is negative
            return fcff
        
        # Method 2: Fallback to info (usually FCFE, not ideal)
        fcf_fallback = info.get('freeCashflow', 0)
        return float(fcf_fallback)
        
    except Exception as e:
        logger.error(f"FCFF Calculation failed: {e}")
        return 0.0

def calculate_wacc(info: Dict[str, Any], balance: pd.DataFrame, income: pd.DataFrame) -> Tuple[float, float, float, float, float]:
    """
    Calculate Weighted Average Cost of Capital (WACC) with Beta Dampening.
    Returns: (wacc, beta, risk_free_rate, market_risk_premium, tax_rate)
    """
    # 1. Cost of Equity (CAPM)
    # Beta Dampening: Adjusted Beta = 0.67 * Raw Beta + 0.33
    raw_beta = info.get('beta', 1.0)
    if raw_beta is None: raw_beta = 1.0
    beta = 0.67 * raw_beta + 0.33
    
    # Dynamic Risk Free Rate (10Y Treasury Yield)
    # Try to fetch ^TNX, fallback to 4.2%
    risk_free_rate = 0.042
    try:
        tnx = yf.Ticker("^TNX")
        # Try to get regularMarketPrice, or previousClose, or history
        tnx_price = tnx.info.get('regularMarketPrice') or tnx.info.get('previousClose')
        if not tnx_price:
            hist = tnx.history(period="1d")
            if not hist.empty:
                tnx_price = hist['Close'].iloc[-1]
        
        if tnx_price and tnx_price > 0:
            risk_free_rate = tnx_price / 100.0  # TNX is e.g. 4.25 for 4.25%
    except Exception as e:
        logger.warning(f"Could not fetch ^TNX for Risk Free Rate: {e}")
    
    market_risk_premium = 0.055  # 5.5%
    cost_of_equity = risk_free_rate + beta * market_risk_premium
    
    # 2. Cost of Debt
    # Interest Expense / Total Debt
    interest_expense = get_latest(income, 'Interest Expense')
    total_debt = get_latest(balance, 'Total Debt')
    
    cost_of_debt = 0.05  # Default 5%
    if interest_expense and total_debt and total_debt > 0:
        # Interest expense is usually negative in statements
        cost_of_debt = abs(interest_expense) / total_debt
    
    # Cap Cost of Debt reasonable range (3% to 10%)
    cost_of_debt = max(0.03, min(cost_of_debt, 0.10))
    
    # 3. Tax Rate
    # Income Tax Expense / Pretax Income
    tax_provision = get_latest(income, 'Tax Provision')
    pretax_income = get_latest(income, 'Pretax Income')
    
    tax_rate = 0.21  # Default US Corp Tax Rate
    if tax_provision and pretax_income and pretax_income > 0:
        rate = tax_provision / pretax_income
        if not pd.isna(rate):
            tax_rate = rate
    
    # Cap Tax Rate (0% to 25% effective)
    if pd.isna(tax_rate): tax_rate = 0.21
    tax_rate = max(0.0, min(tax_rate, 0.25))   
    
    # 4. Capital Structure
    market_cap = info.get('marketCap', 0)
    if not total_debt:
        total_debt = 0
    
    total_value = market_cap + total_debt
    
    if total_value > 0:
        wacc = (market_cap / total_value) * cost_of_equity + (total_debt / total_value) * cost_of_debt * (1 - tax_rate)
    else:
        wacc = 0.09  # Fallback
    
    return wacc, beta, risk_free_rate, market_risk_premium, tax_rate

def get_forward_growth_rate(ticker: str, info: Dict[str, Any], sector: str) -> float:
    """
    Extract forward growth rate using analyst estimates.
    """
    # Method 1: Try to get analyst estimates
    try:
        # Note: In a real production system, we would inject the data source rather than calling yfinance here.
        # For MVP/Refactor, we keep the yfinance call but wrap it safely.
        # Ideally 'info' should already contain this if passed from upstream.
        
        # 'earningsGrowth' is usually Forward Earnings Growth
        forward_earnings_growth = info.get('earningsGrowth')
        
        if forward_earnings_growth and forward_earnings_growth > -0.5:  # Sanity check
            return forward_earnings_growth
            
    except Exception:
        pass
    
    # Method 2: Use Forward EPS Growth as proxy
    forward_eps_growth = info.get('earningsQuarterlyGrowth')
    if forward_eps_growth and forward_eps_growth > -0.5:
        return forward_eps_growth
    
    # Method 3: Fallback to Historical Revenue Growth
    revenue_growth = info.get('revenueGrowth', 0.05)
    return revenue_growth
