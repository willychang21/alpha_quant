import logging
import pandas as pd
from app.domain import schemas
from app.engines.valuation import utils

logger = logging.getLogger(__name__)

def get_financials_valuation(ticker: str, info: dict, income: pd.DataFrame, balance: pd.DataFrame, inputs: schemas.DCFInput, wacc_details: schemas.WACCDetails) -> schemas.ValuationResult:
    """
    Valuation for Banks/Insurance (Financials).
    Uses Dividend Discount Model (DDM) and Excess Return Model.
    """
    # 1. DDM (Gordon Growth)
    dividend_rate = info.get('dividendRate')
    dividend_yield = info.get('dividendYield') or 0.0  # Default to 0 for non-dividend stocks
    
    # Cost of Equity (from WACC details)
    ke = wacc_details.costOfEquity
    
    # Sustainable Growth Rate = ROE * (1 - Payout Ratio)
    roe = info.get('returnOnEquity', 0.10)
    payout_ratio = info.get('payoutRatio', 0.40)
    growth_rate = roe * (1 - payout_ratio)
    
    # Cap growth rate to be conservative (must be < Ke)
    growth_rate = min(growth_rate, ke - 0.01) 
    growth_rate = max(growth_rate, 0.02) # Min 2%
    
    fair_value_ddm = 0.0
    if dividend_rate and dividend_rate > 0:
        # V = D1 / (Ke - g)
        d1 = dividend_rate * (1 + growth_rate)
        fair_value_ddm = d1 / (ke - growth_rate)
    else:
        # Fallback to Book Value + Excess Returns if no dividend
        book_value = info.get('bookValue')
        if book_value:
            # Excess Return = (ROE - Ke) * BV / (Ke - g)
            excess_return = (roe - ke) * book_value / (ke - growth_rate)
            fair_value_ddm = book_value + excess_return
        else:
            # If no book value and no dividend, this stock is not suitable for DDM
            logger.warning(f"{ticker}: No dividend and no book value. DDM not applicable.")
            
    ddm_output = schemas.DDMOutput(
        dividendYield=utils.sanitize(dividend_yield),
        dividendGrowthRate=utils.sanitize(growth_rate),
        costOfEquity=utils.sanitize(ke),
        fairValue=utils.sanitize(fair_value_ddm),
        modelType="DDM / Excess Return"
    )
    
    # Range
    fair_value_range = [
        utils.sanitize(fair_value_ddm * 0.85),
        utils.sanitize(fair_value_ddm),
        utils.sanitize(fair_value_ddm * 1.15)
    ]
    
    rating = "HOLD"
    current_price = info.get('currentPrice', 0)
    if current_price < fair_value_range[0]: rating = "STRONG BUY"
    elif current_price < fair_value_range[1] * 0.9: rating = "BUY"
    elif current_price > fair_value_range[2]: rating = "SELL"
    
    return schemas.ValuationResult(
        ticker=ticker,
        inputs=inputs,
        dcf=None, # No DCF for banks
        ddm=ddm_output,
        waccDetails=wacc_details,
        rating=rating,
        fairValueRange=fair_value_range,
        sensitivity=None
    )
