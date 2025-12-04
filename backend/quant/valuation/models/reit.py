import logging
from typing import Optional, Dict, Any
from quant.valuation.models.base import ValuationModel, ValuationContext, ValuationResult
from quant.utils import financials

logger = logging.getLogger(__name__)

class REITModel(ValuationModel):
    def calculate(self, context: ValuationContext) -> Optional[ValuationResult]:
        ticker = context.ticker
        info = context.market_data.get('info', {})
        financials_data = context.financials
        
        income = financials_data.get('income_stmt')
        # Cashflow is useful for D&A lookup
        cashflow = financials_data.get('cashflow') 
        
        if income is None:
            return None
            
        # 1. Calculate FFO
        net_income = financials.get_latest(income, 'Net Income') or 0
        
        # Try to find Depreciation
        depreciation = financials.get_latest(income, 'Reconciled Depreciation')
        if not depreciation:
            depreciation = financials.get_latest(income, 'Depreciation And Amortization')
        if not depreciation and cashflow is not None:
             depreciation = financials.get_latest(cashflow, 'Depreciation And Amortization')
             
        if not depreciation:
            depreciation = 0
            
        ffo = net_income + depreciation
        shares = info.get('sharesOutstanding', 0)
        
        if shares <= 0:
            return None
            
        ffo_per_share = ffo / shares
        
        # 2. Determine Multiple
        # Premium REITs map
        premium_multiples = {
            'EQIX': 28.0, 'DLR': 27.0, 'PLD': 22.0, 'AMT': 25.0,
            'CCI': 24.0, 'PSA': 23.0, 'WELL': 21.0, 'O': 18.0,
            'SPG': 15.0, 'VNO': 12.0
        }
        
        sector_multiple = premium_multiples.get(ticker)
        
        if sector_multiple is None:
            revenue_growth = info.get('revenueGrowth', 0)
            if revenue_growth > 0.15: sector_multiple = 25.0
            elif revenue_growth > 0.10: sector_multiple = 20.0
            elif revenue_growth > 0.05: sector_multiple = 18.0
            else: sector_multiple = 15.0
            
        fair_value = ffo_per_share * sector_multiple
        current_price = info.get('currentPrice', 0)
        upside = (fair_value - current_price) / current_price if current_price > 0 else 0
        
        return ValuationResult(
            fair_value=financials.sanitize(fair_value),
            upside=financials.sanitize(upside),
            model_name="REIT_FFO_Multiple",
            details={
                "ffo": financials.sanitize(ffo),
                "ffo_per_share": financials.sanitize(ffo_per_share),
                "sector_multiple": sector_multiple,
                "price_to_ffo": financials.sanitize(current_price / ffo_per_share if ffo_per_share else 0)
            }
        )
