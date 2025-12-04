import logging
from typing import Optional, Dict, Any
from quant.valuation.models.base import ValuationModel, ValuationContext, ValuationResult
from quant.utils import financials

logger = logging.getLogger(__name__)

class DDMModel(ValuationModel):
    def calculate(self, context: ValuationContext) -> Optional[ValuationResult]:
        ticker = context.ticker
        info = context.market_data.get('info', {})
        
        # 1. DDM Inputs
        dividend_rate = info.get('dividendRate')
        dividend_yield = info.get('dividendYield') or 0.0
        
        # Cost of Equity (Ke)
        # We need WACC details, but if not available, we calculate Ke on the fly
        # Ideally context.params or context.financials should have this.
        # For now, let's recalculate Ke using financials utils if not provided.
        
        # Check if WACC details are in params
        wacc_details = context.params.get('wacc_details')
        if wacc_details:
            ke = wacc_details.get('cost_of_equity', 0.10)
        else:
            # Fallback calculation
            beta = info.get('beta', 1.0)
            rf = 0.042 # Approximation
            mrp = 0.055
            ke = rf + beta * mrp
            
        # Sustainable Growth Rate
        roe = info.get('returnOnEquity', 0.10)
        payout_ratio = info.get('payoutRatio', 0.40)
        growth_rate = roe * (1 - payout_ratio)
        
        # Cap growth rate
        growth_rate = min(growth_rate, ke - 0.01)
        growth_rate = max(growth_rate, 0.02)
        
        fair_value_ddm = 0.0
        model_type = "DDM_GordonGrowth"
        
        if dividend_rate and dividend_rate > 0:
            d1 = dividend_rate * (1 + growth_rate)
            if ke > growth_rate:
                fair_value_ddm = d1 / (ke - growth_rate)
            else:
                logger.warning(f"{ticker}: Ke ({ke}) <= Growth ({growth_rate}), DDM invalid.")
                return None
        else:
            # Excess Return Model Fallback
            book_value = info.get('bookValue')
            if book_value:
                excess_return = (roe - ke) * book_value / (ke - growth_rate)
                fair_value_ddm = book_value + excess_return
                model_type = "ExcessReturnModel"
            else:
                return None
                
        current_price = info.get('currentPrice', 0)
        upside = (fair_value_ddm - current_price) / current_price if current_price > 0 else 0
        
        return ValuationResult(
            fair_value=financials.sanitize(fair_value_ddm),
            upside=financials.sanitize(upside),
            model_name=model_type,
            details={
                "dividend_yield": financials.sanitize(dividend_yield),
                "dividend_growth_rate": financials.sanitize(growth_rate),
                "cost_of_equity": financials.sanitize(ke),
                "roe": financials.sanitize(roe),
                "payout_ratio": financials.sanitize(payout_ratio)
            }
        )
