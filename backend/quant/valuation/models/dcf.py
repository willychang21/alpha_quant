import logging
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

from quant.valuation.models.base import ValuationModel, ValuationContext, ValuationResult
from quant.utils import financials

logger = logging.getLogger(__name__)

class DCFModel(ValuationModel):
    def calculate(self, context: ValuationContext) -> Optional[ValuationResult]:
        ticker = context.ticker
        info = context.market_data.get('info', {})
        financials_data = context.financials
        
        income = financials_data.get('income_stmt')
        balance = financials_data.get('balance_sheet')
        cashflow = financials_data.get('cashflow')
        
        if income is None or balance is None or cashflow is None:
            logger.warning(f"Missing financial statements for {ticker}")
            return None

        # 1. Calculate WACC
        wacc, beta, rf, mrp, tax_rate = financials.calculate_wacc(info, balance, income)
        
        # 2. Growth Assumptions
        sector = info.get('sector', 'Unknown')
        forward_growth = financials.get_forward_growth_rate(ticker, info, sector)
        
        # Sector-specific adjustments
        if sector == 'Technology':
            initial_growth_rate = max(forward_growth, 0.10)
            initial_growth_rate = min(initial_growth_rate, 0.50)
        elif sector == 'Communication Services':
            initial_growth_rate = max(forward_growth, 0.08)
            initial_growth_rate = min(initial_growth_rate, 0.40)
        else:
            initial_growth_rate = max(forward_growth, 0.02)
            initial_growth_rate = min(initial_growth_rate, 0.25)
            
        terminal_growth_rate = min(0.025, rf - 0.01)
        terminal_growth_rate = max(0.015, terminal_growth_rate)
        
        # 3. Projections
        projection_years = 10 if initial_growth_rate > 0.15 else 5
        
        # Calculate Base FCFF
        base_fcff = financials.calculate_fcff(info, income, cashflow, tax_rate)
        
        projected_fcf = []
        projected_discounted_fcf = []
        pv_fcf_sum = 0
        
        current_fcf = base_fcff
        
        for i in range(1, projection_years + 1):
            decay_factor = (i - 1) / (projection_years - 1) if projection_years > 1 else 1
            growth_rate_t = initial_growth_rate * (1 - decay_factor) + terminal_growth_rate * decay_factor
            growth_rate_t = max(growth_rate_t, terminal_growth_rate)
            
            current_fcf = current_fcf * (1 + growth_rate_t)
            projected_fcf.append(current_fcf)
            
            discount_factor = (1 + wacc) ** i
            pv_fcf = current_fcf / discount_factor
            pv_fcf_sum += pv_fcf
            projected_discounted_fcf.append(pv_fcf)

        # 4. Terminal Value (Gordon Growth)
        last_fcf = projected_fcf[-1]
        tv_gg = last_fcf * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)
        pv_tv_gg = tv_gg / ((1 + wacc) ** projection_years)
        
        # 5. Equity Value
        enterprise_value = pv_fcf_sum + pv_tv_gg
        
        total_debt = financials.get_latest(balance, 'Total Debt') or 0
        cash = financials.get_latest(balance, 'Cash And Cash Equivalents') or 0
        net_debt = total_debt - cash
        
        equity_value = enterprise_value - net_debt
        shares = info.get('sharesOutstanding', 0)
        
        if shares <= 0:
            return None
            
        fair_value = equity_value / shares
        current_price = info.get('currentPrice', 0)
        upside = (fair_value - current_price) / current_price if current_price > 0 else 0
        
        return ValuationResult(
            fair_value=financials.sanitize(fair_value),
            upside=financials.sanitize(upside),
            model_name="DCF_GordonGrowth_v1",
            details={
                "wacc": financials.sanitize(wacc),
                "growth_rate": financials.sanitize(initial_growth_rate),
                "terminal_growth": financials.sanitize(terminal_growth_rate),
                "projection_years": projection_years,
                "equity_value": financials.sanitize(equity_value),
                "enterprise_value": financials.sanitize(enterprise_value),
                "projected_fcf": [financials.sanitize(x) for x in projected_fcf],
                "pv_fcf_sum": financials.sanitize(pv_fcf_sum),
                "pv_terminal_value": financials.sanitize(pv_tv_gg)
            }
        )
