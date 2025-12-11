"""Quantitative Factor Analysis Module.

Implements Citadel-tier factor scoring:
- Quality (ROE, Margins, Leverage)
- Value (P/E, P/B ratios)
- Growth (Revenue, Earnings growth)
- Momentum (Price vs 52-week high)

Also includes Residual Income Model (RIM) and Risk Metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from app.domain import schemas

# Import infrastructure
from core.structured_logger import get_structured_logger
from core.error_handler import handle_gracefully
from config.quant_config import get_factor_config, FactorConfig

logger = get_structured_logger("QuantEngine")


def sanitize(value: Any) -> float:
    """Sanitize float values for JSON serialization.
    
    Args:
        value: Any numeric value to sanitize
        
    Returns:
        0.0 if NaN/Inf, otherwise float(value)
    """
    if pd.isna(value) or np.isinf(value):
        return 0.0
    return float(value)

def linear_score(val: Optional[float], min_val: float, max_val: float) -> float:
    """
    Continuous scoring function using linear interpolation.
    
    Maps val to 0-100 scale based on range [min_val, max_val].
    Avoids discrete cliff-edge risks in factor scoring.
    
    Args:
        val: Value to score
        min_val: Value that maps to 0
        max_val: Value that maps to 100
        
    Returns:
        Score between 0 and 100
    """
    if val is None or pd.isna(val):
        return 50.0
    score = np.interp(val, [min_val, max_val], [0, 100])
    return float(np.clip(score, 0, 100))

def get_quant_analysis(
    ticker: str,
    info: Dict[str, Any],
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
    history: pd.DataFrame
) -> Tuple[schemas.QuantScore, Optional[schemas.ResidualIncomeOutput], schemas.RiskMetrics]:
    """
    Main entry point for Quant Analysis.
    
    Args:
        ticker: Stock ticker symbol
        info: Ticker info dict from yfinance
        income: Income statement DataFrame
        balance: Balance sheet DataFrame
        cashflow: Cashflow statement DataFrame
        history: Price history DataFrame
        
    Returns:
        Tuple of (QuantScore, ResidualIncomeOutput, RiskMetrics)
    """
    logger.info(f"Running quant analysis for {ticker}")
    
    # 1. Calculate Factor Scores
    quant_score = calculate_factor_scores(info, income, balance, cashflow, history)
    
    # 2. Calculate Residual Income Model
    rim_output = calculate_residual_income(info, income, balance)
    
    # 3. Calculate Risk Metrics
    risk_metrics = calculate_risk_metrics(info, history)
    
    return quant_score, rim_output, risk_metrics

def calculate_factor_scores(
    info: Dict[str, Any],
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
    history: pd.DataFrame
) -> schemas.QuantScore:
    """
    Calculates 0-100 scores using continuous mapping (avoiding discrete steps).
    Uses linear interpolation for smooth transitions between score ranges.
    
    Args:
        info: Ticker info dict
        income: Income statement DataFrame
        balance: Balance sheet DataFrame
        cashflow: Cashflow statement DataFrame
        history: Price history DataFrame
        
    Returns:
        QuantScore with quality, value, growth, momentum, and total scores
    """
    details = {}
    
    # --- 1. Quality (Profitability & Leverage) ---
    # ROE: >25% is excellent (100), <5% is poor (0)
    roe = info.get('returnOnEquity', 0)
    s_roe = linear_score(roe, 0.05, 0.25)
    
    # Gross Margin: Sector dependent, but >60% generally wide moat
    gross_margin = info.get('grossMargins', 0)
    s_gm = linear_score(gross_margin, 0.20, 0.60)
    
    # Debt/Equity: <0.5 is safe (100), >2.5 is risky (0) - Inverted
    debt_to_equity = info.get('debtToEquity', 100) / 100.0
    s_de = 100 - linear_score(debt_to_equity, 0.5, 2.5)
    
    quality_score = (s_roe * 0.4) + (s_gm * 0.3) + (s_de * 0.3)
    details['Quality'] = f"ROE: {roe:.1%}, GM: {gross_margin:.1%}, D/E: {debt_to_equity:.2f}"

    # --- 2. Value (Relative Valuation) ---
    # P/E: <10 is cheap (100), >50 is expensive (0) - Inverted
    pe = info.get('trailingPE', 30)
    if not pe or pe <= 0:
        pe = 50  # Penalize negative/missing earnings
    s_pe = 100 - linear_score(pe, 10, 50)
    
    # P/B: <1.0 is cheap (100), >10 is expensive (0) - Inverted
    pb = info.get('priceToBook', 5)
    if not pb or pb <= 0:
        pb = 10
    s_pb = 100 - linear_score(pb, 1.0, 10.0)
    
    value_score = (s_pe * 0.6) + (s_pb * 0.4)
    details['Value'] = f"P/E: {pe:.1f}, P/B: {pb:.1f}"

    # --- 3. Growth (Historical & Forward) ---
    # Revenue Growth: 0% to 30% growth range
    rev_growth = info.get('revenueGrowth', 0)
    s_rev = linear_score(rev_growth, 0.0, 0.30)
    
    # Earnings Growth: 0% to 30% growth range
    earnings_growth = info.get('earningsGrowth', 0)
    s_earn = linear_score(earnings_growth, 0.0, 0.30)
    
    growth_score = (s_rev * 0.5) + (s_earn * 0.5)
    details['Growth'] = f"Rev Growth: {rev_growth:.1%}, Earn Growth: {earnings_growth:.1%}"

    # --- 4. Momentum (Technical) ---
    # Price vs 52w High: Closer to high = Stronger Momentum
    current_price = history['Close'].iloc[-1] if not history.empty and len(history) > 0 else 0
    high_52w = info.get('fiftyTwoWeekHigh', current_price)
    
    momentum_score = 50
    dist_to_high = 0
    
    if high_52w > 0 and current_price > 0:
        dist_to_high = current_price / high_52w
        # 70% of 52w high (0 score) to 100% of 52w high (100 score)
        momentum_score = linear_score(dist_to_high, 0.70, 1.00)
        details['Momentum'] = f"Price/52wHigh: {dist_to_high:.1%}"
    else:
        details['Momentum'] = "N/A"

    # --- Total Score (Weighted) ---
    # Quality and Growth are most important for long-term value
    total_score = (quality_score * 0.3) + (value_score * 0.2) + (growth_score * 0.3) + (momentum_score * 0.2)
    
    return schemas.QuantScore(
        quality=sanitize(quality_score),
        value=sanitize(value_score),
        growth=sanitize(growth_score),
        momentum=sanitize(momentum_score),
        total=sanitize(total_score),
        details=details
    )

def calculate_residual_income(info: dict, income: pd.DataFrame, balance: pd.DataFrame) -> Optional[schemas.ResidualIncomeOutput]:
    """
    Calculates valuation using Residual Income Model (RIM).
    V0 = B0 + Sum( (ROE - r) * B(t-1) / (1+r)^t )
    
    Suitable for Financials where FCF is hard to define.
    """
    try:
        # Prefer explicit Book Value from info
        shares = info.get('sharesOutstanding')
        bv_per_share = info.get('bookValue')
        
        # Fallback: Calculate from Balance Sheet
        if not bv_per_share and not balance.empty:
            equity_keys = ['Total Stockholder Equity', 'Total Equity Gross Minority', 'Stockholders Equity']
            for key in equity_keys:
                if key in balance.index:
                    total_equity = balance.loc[key].iloc[0]
                    if shares and shares > 0:
                        bv_per_share = total_equity / shares
                    break
        
        # Fallback: Calculate shares from market cap
        if not shares:
            mcap = info.get('marketCap')
            price = info.get('currentPrice') or info.get('regularMarketPrice')
            if mcap and price and price > 0:
                shares = mcap / price
        
        # Validation
        if not bv_per_share or bv_per_share <= 0:
            logger.warning("RIM: Book Value Per Share not available or invalid")
            return None
        
        # ROE and Cost of Equity
        roe = info.get('returnOnEquity', 0.10)
        cost_of_equity = 0.08  # Should ideally come from WACC
        
        # Conservative cap on ROE (avoid unrealistic projections)
        roe = min(roe, 0.30)
        
        # 5 Year Projection
        years = 5
        residual_income_sum = 0
        current_bv = bv_per_share
        
        for i in range(1, years + 1):
            # Net Income = BV * ROE
            net_income = current_bv * roe
            # Equity Charge = BV * Cost of Equity
            equity_charge = current_bv * cost_of_equity
            # Residual Income = NI - Equity Charge
            residual_income = net_income - equity_charge
            
            # Discount to present
            residual_income_sum += residual_income / ((1 + cost_of_equity) ** i)
            
            # Update BV (Assume 40% retention ratio, 60% payout)
            current_bv += net_income * 0.40
        
        # Terminal Value (Zero Growth in RI assumption for conservatism)
        terminal_ri = (current_bv * roe) - (current_bv * cost_of_equity)
        tv = terminal_ri / cost_of_equity
        pv_tv = tv / ((1 + cost_of_equity) ** years)
        
        # Fair Value = Current BV + PV(Residual Incomes) + PV(Terminal Value)
        fair_value = bv_per_share + residual_income_sum + pv_tv
        
        return schemas.ResidualIncomeOutput(
            bookValuePerShare=sanitize(bv_per_share),
            roe=sanitize(roe),
            costOfEquity=sanitize(cost_of_equity),
            fairValue=sanitize(fair_value)
        )
    except Exception as e:
        logger.error(f"RIM Calculation Failed: {e}")
        return None

def calculate_risk_metrics(info: dict, history: pd.DataFrame) -> schemas.RiskMetrics:
    """
    Calculates Beta, Volatility, Sharpe Ratio, and Max Drawdown.
    Uses improved statistical methods for better accuracy.
    """
    beta = info.get('beta', 1.0)
    
    if history.empty or len(history) < 30:
        return schemas.RiskMetrics(
            beta=sanitize(beta),
            volatility=0,
            sharpeRatio=0,
            maxDrawdown=0
        )
    
    # Calculate Daily Returns (avoid in-place modification warning)
    returns = history['Close'].pct_change().dropna()
    
    if len(returns) == 0:
        return schemas.RiskMetrics(
            beta=sanitize(beta),
            volatility=0,
            sharpeRatio=0,
            maxDrawdown=0
        )
    
    # Annualized Volatility (252 trading days)
    volatility = returns.std() * np.sqrt(252)
    
    # Sharpe Ratio (Risk Free Rate = 4.2%)
    rf_daily = 0.042 / 252
    excess_returns = returns - rf_daily
    sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    
    # Max Drawdown (Rolling)
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    
    return schemas.RiskMetrics(
        beta=sanitize(beta),
        volatility=sanitize(volatility),
        sharpeRatio=sanitize(sharpe),
        maxDrawdown=sanitize(max_drawdown)
    )
