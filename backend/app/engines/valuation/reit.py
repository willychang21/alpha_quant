import logging
import pandas as pd
from app.domain import schemas
from app.engines.valuation import utils

logger = logging.getLogger(__name__)

def get_reits_valuation(ticker: str, info: dict, income: pd.DataFrame, inputs: schemas.DCFInput) -> schemas.ValuationResult:
    """
    Valuation for Real Estate (REITs).
    Uses Price/FFO Multiples.
    """
    # Enhanced FFO (Funds From Operations) Calculation
    # FFO = Net Income + Depreciation & Amortization - Gains on Sales
    net_income = inputs.netIncome
    
    # Try multiple sources for Depreciation & Amortization
    depreciation = None
    
    # Source 1: Income Statement
    depreciation = utils.get_latest(income, 'Reconciled Depreciation')
    
    # Source 2: Try 'Depreciation And Amortization' directly
    if not depreciation or depreciation == 0:
        depreciation = utils.get_latest(income, 'Depreciation And Amortization')
    
    # Source 3: Cashflow Statement (most reliable for REITs)
    if not depreciation or depreciation == 0:
        import pandas as pd
        # Import cashflow if not passed (defensive)
        try:
            stock = info.get('_ticker_obj')  # Internal yfinance object if available
            if stock:
                cashflow_stmt = stock.quarterly_cashflow
                depreciation = utils.get_latest(cashflow_stmt, 'Depreciation And Amortization')
        except:
            pass
    
    # Default to 0 if still not found
    if not depreciation:
        depreciation = 0
        logger.warning(f"{ticker}: Could not find Depreciation for FFO calculation. FFO may be understated.")
    
    logger.info(f"{ticker}: FFO Calculation - Net Income: {net_income}, D&A: {depreciation}")
    
    # Note: inputs are already currency normalized
    ffo = net_income + depreciation
    shares = inputs.sharesOutstanding
    ffo_per_share = ffo / shares if shares > 0 else 0
    
    # Dynamic FFO Multiple (Sub-Sector Specific)
    # Based on backtest calibration - previous 15x was too conservative
    
    # Sub-Sector Mapping (Premium REITs)
    premium_multiples = {
        'EQIX': 28.0,  # Data Center - Premium (Network Effects)
        'DLR': 27.0,   # Data Center
        'PLD': 22.0,   # Industrial/Logistics - Above Average
        'AMT': 25.0,   # Tower REITs - Premium
        'CCI': 24.0,   # Tower REITs
        'PSA': 23.0,   # Self Storage
        'WELL': 21.0,  # Healthcare
        'O': 18.0,     # Retail (Net Lease)
        'SPG': 15.0,   # Mall REITs
        'VNO': 12.0,   # Office REITs
    }
    
    # Get ticker-specific multiple or calculate based on growth
    sector_multiple = premium_multiples.get(ticker, None)
    
    if sector_multiple is None:
        # Dynamic calculation based on revenue growth
        revenue_growth = info.get('revenueGrowth', 0)
        
        if revenue_growth > 0.15:
            sector_multiple = 25.0  # High Growth REIT
        elif revenue_growth > 0.10:
            sector_multiple = 20.0  # Above Average Growth
        elif revenue_growth > 0.05:
            sector_multiple = 18.0  # Average Growth
        else:
            sector_multiple = 15.0  # Low/No Growth
        
    fair_value = ffo_per_share * sector_multiple
    
    current_price = info.get('currentPrice', 0)
    current_p_ffo = current_price / ffo_per_share if ffo_per_share > 0 else 0
    
    reit_output = schemas.REITOutput(
        ffo=utils.sanitize(ffo),
        ffoPerShare=utils.sanitize(ffo_per_share),
        priceToFFO=utils.sanitize(current_p_ffo),
        fairValue=utils.sanitize(fair_value),
        sectorAveragePFFO=sector_multiple
    )
    
    fair_value_range = [
        utils.sanitize(ffo_per_share * (sector_multiple - 3)),
        utils.sanitize(fair_value),
        utils.sanitize(ffo_per_share * (sector_multiple + 3))
    ]
    
    rating = "HOLD"
    if current_price < fair_value_range[0]: rating = "STRONG BUY"
    elif current_price < fair_value_range[1] * 0.9: rating = "BUY"
    elif current_price > fair_value_range[2]: rating = "SELL"
    
    return schemas.ValuationResult(
        ticker=ticker,
        inputs=inputs,
        dcf=None,
        reit=reit_output,
        rating=rating,
        fairValueRange=fair_value_range,
        sensitivity=None
    )
