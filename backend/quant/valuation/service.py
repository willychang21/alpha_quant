import logging
import pandas as pd
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any

from app.domain import schemas
from quant.valuation.orchestrator import ValuationOrchestrator
from quant.data.models import Security, MarketDataDaily, Fundamentals # Assuming these exist or will be moved
# Note: We might need to keep using app.data.models if they haven't been moved yet.
# The plan said "Move app/data/models.py to backend/quant/data/schema.py".
# For Stage 1, let's assume we still use the app models or the ones available.
# Checking imports in original file: `from quant.data.models import Security...` 
# Wait, the original `backend/quant/valuation/engine.py` imported from `quant.data.models`.
# But `backend/app/engines/valuation/core.py` used `app.domain.schemas`.

# Let's use the Orchestrator and map to schemas.
from app.engines.quant import factors as quant_engine # Keep this for now until refactored

logger = logging.getLogger(__name__)

class ValuationService:
    def __init__(self, db: Session):
        self.db = db
        self.orchestrator = ValuationOrchestrator()

    def get_valuation(self, ticker: str, info: dict, income: pd.DataFrame, balance: pd.DataFrame, cashflow: pd.DataFrame, history: pd.DataFrame = pd.DataFrame(), is_quarterly: bool = False, exchange_rate: float = 1.0) -> schemas.ValuationResult:
        """
        Service method to get valuation result in the API schema format.
        """
        # Prepare data for orchestrator
        data = {
            'info': info,
            'income_stmt': income,
            'balance_sheet': balance,
            'cashflow': cashflow,
            'history': history
        }
        
        params = {
            'is_quarterly': is_quarterly,
            'exchange_rate': exchange_rate
        }
        
        # Call Orchestrator
        quant_result = self.orchestrator.get_valuation(ticker, data, params)
        
        if not quant_result:
            raise ValueError(f"Could not calculate valuation for {ticker}")
            
        # Map to API Schema (schemas.ValuationResult)
        # This is the "Adapter" pattern to keep API stable while changing the core.
        
        # Re-construct inputs (simplified for now, or extract from quant_result details)
        # In a full refactor, we would update schemas.ValuationResult to match quant_result.
        # For now, we try to populate the existing schema.
        
        # We need to reconstruct DCFOutput / DDMOutput / REITOutput based on model_name
        dcf_output = None
        ddm_output = None
        reit_output = None
        
        details = quant_result.details
        
        if "DCF" in quant_result.model_name:
            dcf_output = schemas.DCFOutput(
                wacc=details.get('wacc', 0),
                growthRate=details.get('growth_rate', 0),
                terminalGrowthRate=details.get('terminal_growth', 0),
                projectedFCF=details.get('projected_fcf', []),
                projectedDiscountedFCF=[], # Not passed back for brevity/simplicity in v1 refactor
                terminalValue=details.get('pv_terminal_value', 0), # Using PV TV for now
                terminalValueExitMultiple=0,
                presentValueSum=details.get('enterprise_value', 0),
                equityValue=details.get('equity_value', 0),
                equityValueExitMultiple=0,
                sharePrice=quant_result.fair_value,
                sharePriceGordon=quant_result.fair_value,
                sharePriceExitMultiple=0,
                upside=quant_result.upside
            )
        elif "DDM" in quant_result.model_name:
            ddm_output = schemas.DDMOutput(
                dividendYield=details.get('dividend_yield', 0),
                dividendGrowthRate=details.get('dividend_growth_rate', 0),
                costOfEquity=details.get('cost_of_equity', 0),
                fairValue=quant_result.fair_value,
                modelType=quant_result.model_name
            )
        elif "REIT" in quant_result.model_name:
            reit_output = schemas.REITOutput(
                ffo=details.get('ffo', 0),
                ffoPerShare=details.get('ffo_per_share', 0),
                priceToFFO=details.get('price_to_ffo', 0),
                fairValue=quant_result.fair_value,
                sectorAveragePFFO=details.get('sector_multiple', 0)
            )
            
        # Construct Inputs (Dummy for now or extracted)
        inputs = schemas.DCFInput(
            revenue=0, ebitda=0, netIncome=0, fcf=0, totalDebt=0, totalCash=0, netDebt=0, sharesOutstanding=0, beta=0, riskFreeRate=0, marketRiskPremium=0
        )
        
        # WACC Details (Dummy)
        wacc_details = schemas.WACCDetails(
            riskFreeRate=0, betaRaw=0, betaAdjusted=0, marketRiskPremium=0, costOfEquity=0, costOfDebt=0, taxRate=0, afterTaxCostOfDebt=0, equityWeight=0, debtWeight=0, wacc=details.get('wacc', 0)
        )
        
        # Fair Value Range
        fv = quant_result.fair_value
        fv_range = [fv * 0.85, fv, fv * 1.15]
        
        result = schemas.ValuationResult(
            ticker=ticker,
            inputs=inputs,
            dcf=dcf_output,
            ddm=ddm_output,
            reit=reit_output,
            waccDetails=wacc_details,
            rating=details.get('rating', 'HOLD'),
            fairValueRange=fv_range,
            sensitivity=None
        )
        
        # Populate common fields
        result.price = info.get('currentPrice', 0)
        result.name = info.get('shortName')
        result.sector = info.get('sector')
        
        # --- Quant & Risk Analysis (Legacy Integration) ---
        try:
            quant_score, rim_output, risk_metrics = quant_engine.get_quant_analysis(ticker, info, income, balance, cashflow, history)
            result.quant = quant_score
            result.rim = rim_output
            result.risk = risk_metrics
        except Exception as e:
            logger.error(f"Quant Engine Failed: {e}")

        return result
