"""Valuation Engine Core Module.

Main entry point for stock valuation, dispatching to appropriate models
based on sector (DCF, DDM, REIT). Includes Wall Street ensemble and
quantitative analysis integration.
"""

import time
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

from app.domain import schemas
from app.engines.valuation import dcf, ddm, reit, utils
from app.engines.quant import factors as quant_engine

# Import infrastructure
from core.structured_logger import get_structured_logger
from core.error_handler import handle_gracefully
from config.quant_config import get_valuation_config, ValuationConfig

logger = get_structured_logger("ValuationEngine")


def get_valuation(ticker: str, info: dict, income: pd.DataFrame, balance: pd.DataFrame, cashflow: pd.DataFrame, history: pd.DataFrame = pd.DataFrame(), is_quarterly: bool = False, exchange_rate: float = 1.0) -> schemas.ValuationResult:
    """
    Main entry point for valuation.
    Dispatches to appropriate model based on sector/industry.
    """
    logger.info(f"Starting valuation for {ticker}")
    try:
        if info.get('quoteType') == 'ETF':
            raise ValueError(f"DCF Valuation model is not applicable for ETFs ({ticker}).")
            
        # --- 0. Currency Normalization ---
        currency = info.get('currency', 'USD')
        # Exchange rate is now passed in from the service layer
        if exchange_rate != 1.0:
            logger.info(f"Using provided exchange rate: {exchange_rate}")

        # --- 1. Inputs Extraction ---
        # Apply exchange rate to all financial figures
        revenue = (utils.get_latest(income, 'Total Revenue') or 0) * exchange_rate
        ebitda = (utils.get_latest(income, 'Normalized EBITDA') or utils.get_latest(income, 'EBITDA') or 0) * exchange_rate
        net_income = (utils.get_latest(income, 'Net Income') or 0) * exchange_rate
        last_fcf = (utils.get_latest(cashflow, 'Free Cash Flow') or 0) * exchange_rate
        
        # Annualize if quarterly
        if is_quarterly:
            logger.info(f"{ticker}: Annualizing quarterly data (x4)")
            revenue *= 4
            ebitda *= 4
            net_income *= 4
            last_fcf *= 4
            
            # Update DataFrames so calculate_fcff uses annualized values
            if income is not None:
                income = income * 4
            if cashflow is not None:
                cashflow = cashflow * 4
        
        # Robust Shares Outstanding
        shares = info.get('sharesOutstanding')
        if not shares:
            shares = info.get('impliedSharesOutstanding')
        if not shares and info.get('marketCap') and info.get('currentPrice'):
            shares = info.get('marketCap') / info.get('currentPrice')
        if not shares:
            shares = 0
            
        # Robust Net Debt
        total_debt = info.get('totalDebt')
        total_cash = info.get('totalCash')
        
        if total_debt is None and balance is not None and not balance.empty:
            total_debt = utils.get_latest(balance, 'Total Debt')
            if total_debt: total_debt *= exchange_rate # Convert from BS
            
        if total_cash is None and balance is not None and not balance.empty:
            total_cash = utils.get_latest(balance, 'Cash And Cash Equivalents')
            if total_cash: total_cash *= exchange_rate # Convert from BS
            
        # Re-fetch Debt/Cash from DF to be safe and apply rate
        if balance is not None and not balance.empty:
            total_debt_bs = utils.get_latest(balance, 'Total Debt')
            if total_debt_bs: total_debt = total_debt_bs * exchange_rate
            
            total_cash_bs = utils.get_latest(balance, 'Cash And Cash Equivalents')
            if total_cash_bs: total_cash = total_cash_bs * exchange_rate
            
        if total_debt is not None and total_cash is not None:
            net_debt = total_debt - total_cash
        else:
            net_debt = 0.0

        # WACC Calculation (now returns tax_rate for FCFF)
        wacc, beta, rf, mrp, tax_rate = utils.calculate_wacc(info, balance, income)
        
        # Re-derive WACC Details
        raw_beta = info.get('beta', 1.0)
        market_cap = info.get('marketCap', 0)
        total_debt_val = total_debt if total_debt else 0
        total_val = market_cap + total_debt_val
        equity_weight = market_cap / total_val if total_val > 0 else 1.0
        debt_weight = total_debt_val / total_val if total_val > 0 else 0.0
        
        interest_expense = utils.get_latest(income, 'Interest Expense')
        cost_of_debt = 0.05
        if interest_expense and total_debt_val > 0:
            cost_of_debt = abs(interest_expense) / total_debt_val
        cost_of_debt = max(0.03, min(cost_of_debt, 0.10))
        
        cost_of_equity = rf + beta * mrp
        after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)
        
        wacc_details = schemas.WACCDetails(
            riskFreeRate=utils.sanitize(rf),
            betaRaw=utils.sanitize(raw_beta),
            betaAdjusted=utils.sanitize(beta),
            marketRiskPremium=utils.sanitize(mrp),
            costOfEquity=utils.sanitize(cost_of_equity),
            costOfDebt=utils.sanitize(cost_of_debt),
            taxRate=utils.sanitize(tax_rate),
            afterTaxCostOfDebt=utils.sanitize(after_tax_cost_of_debt),
            equityWeight=utils.sanitize(equity_weight),
            debtWeight=utils.sanitize(debt_weight),
            wacc=utils.sanitize(wacc)
        )
        
        inputs = schemas.DCFInput(
            revenue=utils.sanitize(revenue),
            ebitda=utils.sanitize(ebitda),
            netIncome=utils.sanitize(net_income),
            fcf=utils.sanitize(last_fcf),
            totalDebt=utils.sanitize(total_debt or 0),
            totalCash=utils.sanitize(total_cash or 0),
            netDebt=utils.sanitize(net_debt),
            sharesOutstanding=utils.sanitize(shares),
            beta=utils.sanitize(beta),
            riskFreeRate=utils.sanitize(rf),
            marketRiskPremium=utils.sanitize(mrp)
        )
        
        # --- SECTOR DISPATCHER ---
        sector = info.get('sector', 'Unknown')
        logger.info(f"Valuation Sector Dispatch: {ticker} is in {sector}")
        
        if sector == 'Financial Services':
            # Use DDM / Excess Return
            result = ddm.get_financials_valuation(ticker, info, income, balance, inputs, wacc_details)
        elif sector == 'Real Estate':
            # Use REIT FFO Model
            result = reit.get_reits_valuation(ticker, info, income, inputs)
        else:
            # Use Standard DCF (Tech, Healthcare, Consumer, etc.)
            # Use Standard DCF (Tech, Healthcare, Consumer, etc.)
            result = dcf.get_standard_dcf_valuation(ticker, info, inputs, wacc_details, wacc, beta, rf, mrp, income, balance, cashflow, exchange_rate=exchange_rate)
            
        # --- Common Analysis (Multiples & Advanced) ---
        # Multiples Analysis
        eps = info.get('trailingEps') or info.get('forwardEps') or 0.0
        pe_scenarios = {}
        if eps > 0:
            for mult in [15, 20, 25, 30]:
                pe_scenarios[f"{mult}x"] = utils.sanitize(eps * mult)
                
        ebitda_val = ebitda if ebitda and ebitda > 0 else 0.0
        ev_ebitda_scenarios = {}
        if ebitda_val > 0:
            for mult in [10, 15, 20, 25]:
                implied_ev = ebitda_val * mult
                implied_equity = implied_ev - net_debt
                implied_price = implied_equity / shares if shares > 0 else 0
                ev_ebitda_scenarios[f"{mult}x"] = utils.sanitize(implied_price)
                
        multiples_analysis = schemas.MultiplesAnalysis(
            eps=utils.sanitize(eps),
            peScenarios=pe_scenarios,
            ebitda=utils.sanitize(ebitda_val),
            evEbitdaScenarios=ev_ebitda_scenarios,
            currentPE=utils.sanitize(info.get('trailingPE')),
            currentEvEbitda=utils.sanitize(info.get('enterpriseToEbitda'))
        )
        result.multiples = multiples_analysis
        
        # Apply General Advanced Analysis for all tickers
        result = get_general_advanced_analysis(result, info)
        
        # --- 7. Wall Street Consensus Ensemble (60% Model, 40% Analyst) ---
        result = apply_wall_street_ensemble(result, info)
        
        # --- 8. Quant & Risk Analysis (Citadel-Tier) ---
        try:
            quant_score, rim_output, risk_metrics = quant_engine.get_quant_analysis(ticker, info, income, balance, cashflow, history)
            result.quant = quant_score
            result.rim = rim_output
            result.risk = risk_metrics
        except Exception as e:
            logger.error(f"Quant Engine Failed: {e}")
            # Continue without quant data if it fails
            
        # Ensure top-level fields are populated (Centralized Logic)
        result.name = info.get('shortName') or info.get('longName') or "Unknown"
        result.sector = info.get('sector', 'Unknown')
        result.industry = info.get('industry', 'Unknown')
        result.price = utils.sanitize(info.get('currentPrice', 0))
        result.currency = currency
        
        # Fundamental Ratios
        result.profitMargin = utils.sanitize(info.get('profitMargins'))
        result.roe = utils.sanitize(info.get('returnOnEquity'))
        result.dividendYield = utils.sanitize(info.get('dividendYield'))
        
        # Financial Health
        result.debtToEquity = utils.sanitize(info.get('debtToEquity'))
        result.currentRatio = utils.sanitize(info.get('currentRatio'))
        result.freeCashflow = utils.sanitize(info.get('freeCashflow'))
        result.operatingCashflow = utils.sanitize(info.get('operatingCashflow'))
        
        # Valuation Extended
        result.forwardPE = utils.sanitize(info.get('forwardPE'))
        
        # PEG Ratio (Fallback calculation)
        peg = utils.sanitize(info.get('pegRatio'))
        if not peg and result.forwardPE and result.forwardPE > 0:
            # Try to get growth rate from DCF or Info
            growth = info.get('earningsGrowth') or info.get('revenueGrowth')
            if result.dcf:
                growth = result.dcf.growthRate
            
            if growth and growth > 0:
                peg = result.forwardPE / (growth * 100)
        
        result.pegRatio = utils.sanitize(peg)
        
        result.priceToBook = utils.sanitize(info.get('priceToBook'))
        result.bookValue = utils.sanitize(info.get('bookValue'))
        result.evToEbitda = utils.sanitize(info.get('enterpriseToEbitda'))
        
        # FCF Yield
        mcap = info.get('marketCap')
        fcf = info.get('freeCashflow')
        if fcf and mcap and mcap > 0:
            result.fcfYield = utils.sanitize(fcf / mcap)
            
        # Analyst
        result.targetMeanPrice = utils.sanitize(info.get('targetMeanPrice'))
        result.recommendationKey = info.get('recommendationKey')
            
        return result
        
    except Exception as e:
        logger.error(f"Valuation error for {ticker}: {e}")
        raise e

def apply_wall_street_ensemble(result: schemas.ValuationResult, info: dict) -> schemas.ValuationResult:
    """
    Blend model fair value with Wall Street analyst consensus.
    """
    # Extract current model fair value
    model_fair_value = None
    
    if result.dcf:
        model_fair_value = result.dcf.sharePrice
    elif result.ddm:
        model_fair_value = result.ddm.fairValue
    elif result.reit:
        model_fair_value = result.reit.fairValue
    
    if not model_fair_value or model_fair_value <= 0:
        return result  # No valid model fair value
    
    # Extract analyst consensus
    analyst_mean = info.get('targetMeanPrice')
    analyst_low = info.get('targetLowPrice')
    analyst_high = info.get('targetHighPrice')
    num_analysts = info.get('numberOfAnalystOpinions')
    
    if not analyst_mean or analyst_mean <= 0:
        logger.info(f"{result.ticker}: No analyst consensus available. Using model fair value only.")
        return result
        
    # Ensemble weighting
    model_weight = 0.60
    analyst_weight = 0.40
    
    # Calculate ensemble fair value
    ensemble_fair_value = (model_fair_value * model_weight) + (analyst_mean * analyst_weight)
    
    logger.info(f"\n    {result.ticker} Wall Street Ensemble:\n"
                f"      Model FV: ${model_fair_value:.2f} (60%)\n"
                f"      Analyst Mean: ${analyst_mean:.2f} (40%)\n"
                f"      Ensemble FV: ${ensemble_fair_value:.2f}\n"
                f"      Analysts: {num_analysts or 'N/A'}\n")
    
    # Update fair value in the appropriate model
    if result.dcf:
        result.dcf.sharePrice = ensemble_fair_value
    elif result.ddm:
        result.ddm.fairValue = ensemble_fair_value
    elif result.reit:
        result.reit.fairValue = ensemble_fair_value
        
    # Update fair value range (recalculate based on ensemble)
    current_price = info.get('currentPrice', 0)
    if current_price > 0:
        # Recalculate rating based on ensemble fair value
        discount = (ensemble_fair_value - current_price) / current_price
        
        if discount > 0.20:
            result.rating = "STRONG BUY"
        elif discount > 0.10:
            result.rating = "BUY"
        elif discount < -0.10:
            result.rating = "SELL"
        else:
            result.rating = "HOLD"
    
    # Update fair value range using analyst targets
    if analyst_low and analyst_high:
        result.fairValueRange = [
            utils.sanitize(analyst_low),
            utils.sanitize(ensemble_fair_value),
            utils.sanitize(analyst_high)
        ]
    else:
        # Fallback to +/- 20% of ensemble
        result.fairValueRange = [
            utils.sanitize(ensemble_fair_value * 0.8),
            utils.sanitize(ensemble_fair_value),
            utils.sanitize(ensemble_fair_value * 1.2)
        ]
    
    return result

def get_general_advanced_analysis(base_result: schemas.ValuationResult, info: dict) -> schemas.ValuationResult:
    """
    Applies a generalized "Advanced Analysis" applicable to most companies.
    """
    current_price = info.get('currentPrice')
    if not current_price:
        if base_result.dcf:
            current_price = base_result.dcf.sharePrice
        elif base_result.ddm:
            current_price = base_result.ddm.fairValue
        elif base_result.reit:
            current_price = base_result.reit.fairValue
            
    if not current_price or current_price <= 0: return base_result
    
    # --- 1. Reverse DCF (Market Implied Growth) ---
    implied_growth = None
    
    if base_result.dcf:
        def calculate_dcf_price(growth_rate):
            # Simplified DCF for solver
            wacc = base_result.dcf.wacc
            fcf = base_result.inputs.fcf
            shares = base_result.inputs.sharesOutstanding
            net_debt = base_result.inputs.netDebt
            terminal_growth = base_result.dcf.terminalGrowthRate
            projection_years = 10 # Standardize
            
            pv_sum = 0
            curr_fcf = fcf
            
            # Projection Phase
            for i in range(1, projection_years + 1):
                curr_fcf = curr_fcf * (1 + growth_rate)
                pv_sum += curr_fcf / ((1 + wacc) ** i)
                
            # Terminal Phase (Gordon)
            tv = curr_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
            pv_tv = tv / ((1 + wacc) ** projection_years)
            
            ev = pv_sum + pv_tv
            equity = ev - net_debt
            return equity / shares if shares > 0 else 0

        # Binary Search for Implied Growth
        low = -0.50
        high = 1.00
        
        for _ in range(20): # 20 iterations is enough precision
            mid = (low + high) / 2
            price = calculate_dcf_price(mid)
            if price < current_price:
                low = mid
            else:
                high = mid
        implied_growth = (low + high) / 2
    
    # --- 2. Analyst Targets ---
    target_low = info.get('targetLowPrice')
    target_mean = info.get('targetMeanPrice')
    target_high = info.get('targetHighPrice')
    recommendation = info.get('recommendationKey', 'N/A').replace('_', ' ').title()
    
    # --- 3. PEG Ratio ---
    peg_ratio = utils.sanitize(info.get('pegRatio'))
    
    models = []
    
    # Model 1: Market Implied Expectations
    if implied_growth is not None:
        models.append({
            "name": "Market Implied Growth (Reverse DCF)",
            "value": utils.sanitize(implied_growth * 100), # Display as percentage
            "details": f"To justify the current price of ${current_price:.2f}, the market expects annual growth of {implied_growth*100:.1f}% for 10 years."
        })
    
    # Model 2: Wall Street Consensus
    if target_mean:
        t_low = target_low if target_low is not None else 0.0
        t_high = target_high if target_high is not None else 0.0
        models.append({
            "name": "Wall St. Analyst Consensus",
            "value": utils.sanitize(target_mean),
            "details": f"Mean Target: ${target_mean:.2f} (Low: ${t_low:.2f} - High: ${t_high:.2f}). Rec: {recommendation}."
        })
        
    # Model 3: Growth vs Price (PEG)
    if peg_ratio:
        details = "Undervalued (< 1.0)" if peg_ratio < 1 else "Fair (1.0 - 2.0)" if peg_ratio < 2 else "Premium (> 2.0)"
        models.append({
            "name": "PEG Ratio (Growth Adjusted)",
            "value": utils.sanitize(peg_ratio),
            "details": f"PEG: {peg_ratio:.2f}. Indicates stock is {details} relative to growth."
        })
        
    special_analysis = schemas.SpecialAnalysis(
        title="Advanced Market Insights",
        description="Reverse DCF, Analyst Targets, and Growth Metrics",
        models=models,
        blendedValue=utils.sanitize(target_mean if target_mean else base_result.dcf.sharePrice) if base_result.dcf else 0.0
    )
    
    base_result.specialAnalysis = special_analysis
    return base_result
