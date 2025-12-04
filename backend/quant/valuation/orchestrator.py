import logging
import pandas as pd
from typing import Dict, Any, Optional, List

from quant.valuation.models.base import ValuationModel, ValuationContext, ValuationResult
from quant.valuation.models.dcf import DCFModel
from quant.valuation.models.ddm import DDMModel
from quant.valuation.models.reit import REITModel
from quant.utils import financials

logger = logging.getLogger(__name__)

class ValuationOrchestrator:
    def __init__(self):
        self.models = {
            'dcf': DCFModel(),
            'ddm': DDMModel(),
            'reit': REITModel()
        }

    def get_valuation(self, ticker: str, data: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> Optional[ValuationResult]:
        """
        Main entry point for valuation.
        """
        if params is None:
            params = {}
            
        info = data.get('info', {})
        income = data.get('income_stmt')
        balance = data.get('balance_sheet')
        cashflow = data.get('cashflow')
        
        # 0. Currency Normalization
        exchange_rate = params.get('exchange_rate', 1.0)
        if exchange_rate != 1.0:
            # Apply exchange rate to DataFrames
            # Note: This modifies the DataFrames in place or creates new ones. 
            # For safety, we should probably copy, but for performance we might not.
            # Assuming 'data' is transient for this request.
            if income is not None: income = income * exchange_rate
            if balance is not None: balance = balance * exchange_rate
            if cashflow is not None: cashflow = cashflow * exchange_rate
            
        # 1. Annualization (if quarterly)
        if params.get('is_quarterly', False):
            if income is not None: income = income * 4
            if cashflow is not None: cashflow = cashflow * 4
            # Balance sheet is a snapshot, don't multiply
            
        # 2. Context Creation
        context = ValuationContext(
            ticker=ticker,
            valuation_date=pd.Timestamp.now().date(),
            financials={
                'income_stmt': income,
                'balance_sheet': balance,
                'cashflow': cashflow
            },
            market_data={
                'info': info
            },
            params=params
        )
        
        # 3. Model Selection
        sector = info.get('sector', 'Unknown')
        quote_type = info.get('quoteType')
        
        if quote_type == 'ETF':
            logger.warning(f"Valuation not applicable for ETF: {ticker}")
            return None
            
        model_key = 'dcf' # Default
        if sector == 'Financial Services':
            model_key = 'ddm'
        elif sector == 'Real Estate':
            model_key = 'reit'
            
        # Override via params
        if 'model_type' in params:
            model_key = params['model_type']
            
        logger.info(f"Selected valuation model '{model_key}' for {ticker} ({sector})")
        
        model = self.models.get(model_key)
        if not model:
            logger.error(f"Model '{model_key}' not found.")
            return None
            
        # 4. Execution
        try:
            result = model.calculate(context)
            if result:
                # Post-process / Enrich result
                self._enrich_result(result, context)
            return result
        except Exception as e:
            logger.error(f"Error executing valuation model for {ticker}: {e}", exc_info=True)
            return None

    def _enrich_result(self, result: ValuationResult, context: ValuationContext):
        """
        Add common analysis like ratings, advanced metrics, etc.
        """
        info = context.market_data.get('info', {})
        current_price = info.get('currentPrice', 0)
        
        # Rating Logic
        if current_price > 0:
            upside = result.upside
            if upside > 0.20:
                rating = "STRONG BUY"
            elif upside > 0.10:
                rating = "BUY"
            elif upside < -0.10:
                rating = "SELL"
            else:
                rating = "HOLD"
            result.details['rating'] = rating
            result.details['current_price'] = current_price
            
        # Wall Street Ensemble (Simplified)
        target_mean = info.get('targetMeanPrice')
        if target_mean:
            model_fv = result.fair_value
            ensemble_fv = (model_fv * 0.6) + (target_mean * 0.4)
            result.details['ensemble_fair_value'] = financials.sanitize(ensemble_fv)
            result.details['analyst_target_mean'] = financials.sanitize(target_mean)
