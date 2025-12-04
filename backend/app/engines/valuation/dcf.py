import logging
from app.domain import schemas
from app.engines.valuation import utils

logger = logging.getLogger(__name__)

def get_standard_dcf_valuation(ticker, info, inputs, wacc_details, wacc, beta, rf, mrp, income, balance, cashflow, exchange_rate=1.0):
    """
    Standard DCF Logic.
    """
    # Growth Assumptions (Sector-Adjusted with Forward Estimates)
    # Use analyst forward estimates for better accuracy
    sector = info.get('sector', 'Unknown')
    
    # Get forward growth rate using analyst estimates
    forward_growth = utils.get_forward_growth_rate(ticker, info, sector)
    
    if sector == 'Technology':
        # Tech: Use higher floor and ceiling for growth
        initial_growth_rate = max(forward_growth, 0.10)  # Minimum 10%
        initial_growth_rate = min(initial_growth_rate, 0.50)  # Cap at 50%
    elif sector == 'Communication Services':
        # Similar to Tech for GOOGL, META
        initial_growth_rate = max(forward_growth, 0.08)  # Minimum 8%
        initial_growth_rate = min(initial_growth_rate, 0.40)  # Cap at 40%
    else:
        # Traditional industries: Conservative growth
        initial_growth_rate = max(forward_growth, 0.02)  # Minimum 2%
        initial_growth_rate = min(initial_growth_rate, 0.25)  # Cap at 25%
    
    # Dynamic Terminal Growth Rate (should not exceed Risk Free Rate)
    terminal_growth_rate = min(0.025, rf - 0.01)
    terminal_growth_rate = max(0.015, terminal_growth_rate)  # Floor at 1.5%
    
    # --- 2. DCF Calculation ---
    
    # Adaptive Projection Period
    # If growth > 15%, use 10 years to capture the high growth phase
    projection_years = 10 if initial_growth_rate > 0.15 else 5
    
    projected_fcf = []
    projected_ebitda = [] # For Exit Multiple
    
    # Recalculate Base FCFF (Unlevered Cash Flow)
    # inputs.fcf might be FCFE from yfinance, so we attempt to calculate FCFF
    # NOTE: calculate_fcff returns raw currency (e.g. TWD), so we must apply exchange_rate
    base_fcff_raw = utils.calculate_fcff(info, income, cashflow, wacc_details.taxRate)
    
    if base_fcff_raw is not None and base_fcff_raw != 0:
        base_fcff = base_fcff_raw * exchange_rate
    else:
        base_fcff = inputs.fcf  # Fallback to inputs (already converted in core.py)
        logger.warning(f"Using inputs.fcf as fallback (may be FCFE): {base_fcff}")
    
    current_fcf = base_fcff
    current_ebitda = inputs.ebitda
    
    for i in range(1, projection_years + 1):
        # Linear decay of growth rate towards terminal growth
        decay_factor = (i - 1) / (projection_years - 1) if projection_years > 1 else 1
        current_growth = initial_growth_rate * (1 - decay_factor) + terminal_growth_rate * decay_factor
        current_growth = max(current_growth, terminal_growth_rate)
        
        current_fcf = current_fcf * (1 + current_growth)
        projected_fcf.append(current_fcf)
        
        if current_ebitda > 0:
            current_ebitda = current_ebitda * (1 + current_growth)
            projected_ebitda.append(current_ebitda)
        
    # 6. Calculate Present Value of Future Cash Flows
    # We recalculate this to ensure we have the correct discount factors and PVs for logging
    
    pv_fcf_sum = 0
    projected_discounted_fcf = []
    
    # Reset FCF for discounting loop
    current_fcf_for_pv = base_fcff
    
    # Logging setup for NVDA
    if ticker == 'NVDA':
        logger.info(f"\n[DCF DEBUG NVDA] Detailed FCF Projection:")
        logger.info(f"  Base FCF: ${base_fcff:,.0f}")
        logger.info(f"  Initial Growth: {initial_growth_rate:.1%}")
        logger.info(f"  WACC: {wacc:.1%}")
        logger.info(f"  Shares: {inputs.sharesOutstanding:,.0f}")
    
    for i in range(1, projection_years + 1):
        # Decay growth rate (Linear decay matching original logic)
        decay_factor = (i - 1) / (projection_years - 1) if projection_years > 1 else 1
        growth_rate_t = initial_growth_rate * (1 - decay_factor) + terminal_growth_rate * decay_factor
        growth_rate_t = max(growth_rate_t, terminal_growth_rate)
        
        # Calculate FCF for year i
        current_fcf_for_pv = current_fcf_for_pv * (1 + growth_rate_t)
        
        # Discount to PV
        discount_factor = (1 + wacc) ** i
        pv_fcf = current_fcf_for_pv / discount_factor
        
        pv_fcf_sum += pv_fcf
        projected_discounted_fcf.append(pv_fcf)
        
        if ticker == 'NVDA':
            logger.info(f"  Year {i}: Growth={growth_rate_t:.1%} | FCF=${current_fcf_for_pv:,.0f} | PV=${pv_fcf:,.0f}")

    # 7. Calculate Terminal Value
    # --- Method 1: Gordon Growth Model ---
    # Use the last calculated FCF
    last_projected_fcf = current_fcf_for_pv
    terminal_value_gg = last_projected_fcf * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)
    pv_terminal_gg = terminal_value_gg / ((1 + wacc) ** projection_years)
    
    ev_gg = pv_fcf_sum + pv_terminal_gg
    eq_gg = ev_gg - inputs.netDebt
    price_gg = eq_gg / inputs.sharesOutstanding if inputs.sharesOutstanding > 0 else 0
    
    # --- Method 2: Exit Multiple Method ---
    # Default to 15x or derive from current if reasonable (e.g. 5x to 50x)
    # Current EV/EBITDA
    current_ev = info.get('enterpriseValue')
    current_ebitda_val = info.get('ebitda')
    exit_multiple = 15.0
    
    price_em = 0.0
    terminal_value_em = 0.0
    
    if current_ev and current_ebitda_val and current_ebitda_val > 0:
        implied_multiple = current_ev / current_ebitda_val
        exit_multiple = max(5.0, min(implied_multiple, 25.0))
        
    price_em = 0.0
    terminal_value_em = 0.0
    
    if projected_ebitda:
        last_projected_ebitda = projected_ebitda[-1]
        terminal_value_em = last_projected_ebitda * exit_multiple
        pv_terminal_em = terminal_value_em / ((1 + wacc) ** projection_years)
        
        ev_em = pv_fcf_sum + pv_terminal_em
        eq_em = ev_em - inputs.netDebt
        price_em = eq_em / inputs.sharesOutstanding if inputs.sharesOutstanding > 0 else 0
    else:
        # Fallback if no EBITDA data
        price_em = price_gg
        terminal_value_em = terminal_value_gg
        eq_em = eq_gg  # Use Gordon Growth values as fallback

    # --- Final Blended Valuation ---
    # Smart Blending:
    if initial_growth_rate > 0.15:
        # High Growth: 80% Exit Multiple, 20% Gordon
        share_price_est = (price_gg * 0.20) + (price_em * 0.80)
    else:
        # Mature/Stable: 50% / 50%
        share_price_est = (price_gg + price_em) / 2
    
    current_price = info.get('currentPrice', 0)
    upside = (share_price_est - current_price) / current_price if current_price > 0 else 0
    
    dcf_output = schemas.DCFOutput(
        wacc=utils.sanitize(wacc),
        growthRate=utils.sanitize(initial_growth_rate),
        terminalGrowthRate=utils.sanitize(terminal_growth_rate),
        projectedFCF=[utils.sanitize(x) for x in projected_fcf],
        projectedDiscountedFCF=[utils.sanitize(x) for x in projected_discounted_fcf],
        terminalValue=utils.sanitize(terminal_value_gg), # Show Gordon as primary for consistency
        terminalValueExitMultiple=utils.sanitize(terminal_value_em),
        presentValueSum=utils.sanitize(pv_fcf_sum + pv_terminal_gg), # Note: this is PV sum + PV TV(Gordon)
        equityValue=utils.sanitize(eq_gg),
        equityValueExitMultiple=utils.sanitize(eq_em),
        sharePrice=utils.sanitize(share_price_est), # Blended
        sharePriceGordon=utils.sanitize(price_gg),
        sharePriceExitMultiple=utils.sanitize(price_em),
        upside=utils.sanitize(upside)
    )
    
    # --- 3. Sensitivity Analysis ---
    wacc_range = [wacc - 0.01, wacc, wacc + 0.01]
    growth_range = [terminal_growth_rate - 0.005, terminal_growth_rate, terminal_growth_rate + 0.005]
    
    last_projected_ebitda = projected_ebitda[-1] if projected_ebitda else 0
    
    sensitivity_rows = []
    for w in wacc_range:
        prices = []
        for g in growth_range:
            # Recalculate TV and PV with new assumptions
            
            # Re-discount FCFs
            s_pv_fcf = 0
            for i, val in enumerate(projected_fcf):
                s_pv_fcf += val / ((1 + w) ** (i + 1))
            
            # Method 1: Gordon Growth
            s_tv_gg = last_projected_fcf * (1 + g) / (w - g)
            s_pv_tv_gg = s_tv_gg / ((1 + w) ** projection_years)
            
            s_ev_gg = s_pv_fcf + s_pv_tv_gg
            s_eq_gg = s_ev_gg - inputs.netDebt
            s_price_gg = s_eq_gg / inputs.sharesOutstanding if inputs.sharesOutstanding > 0 else 0
            
            # Method 2: Exit Multiple
            s_price_em = 0
            if last_projected_ebitda > 0:
                s_tv_em = last_projected_ebitda * exit_multiple
                s_pv_tv_em = s_tv_em / ((1 + w) ** projection_years)
                
                s_ev_em = s_pv_fcf + s_pv_tv_em
                s_eq_em = s_ev_em - inputs.netDebt
                s_price_em = s_eq_em / inputs.sharesOutstanding if inputs.sharesOutstanding > 0 else 0
            else:
                s_price_em = s_price_gg # Fallback
            
            # Blended Price (Use same logic as main valuation)
            if initial_growth_rate > 0.15:
                # High Growth: 80% Exit Multiple, 20% Gordon
                s_price_blended = (s_price_gg * 0.20) + (s_price_em * 0.80)
            else:
                # Mature/Stable: 50% / 50%
                s_price_blended = (s_price_gg + s_price_em) / 2
            
            prices.append(utils.sanitize(s_price_blended))
            
        sensitivity_rows.append(schemas.SensitivityRow(wacc=utils.sanitize(w), prices=prices))
        
    sensitivity = schemas.SensitivityAnalysis(
        growthRates=[utils.sanitize(g) for g in growth_range],
        rows=sensitivity_rows
    )
    
    return schemas.ValuationResult(
        ticker=ticker,
        inputs=inputs,
        dcf=dcf_output,
        waccDetails=wacc_details,
        rating="HOLD", # Will be updated by caller
        fairValueRange=[utils.sanitize(share_price_est * 0.85), utils.sanitize(share_price_est), utils.sanitize(share_price_est * 1.15)],
        sensitivity=sensitivity
    )
