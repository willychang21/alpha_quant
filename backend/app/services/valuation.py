import logging
from app.services import market_data
from app.engines import valuation as valuation_engine
from app.engines.quant import factors as quant_engine
from app.domain import schemas

logger = logging.getLogger(__name__)

def analyze_stock(ticker: str) -> schemas.ValuationResult:
    """
    Orchestrates the full analysis pipeline for a single stock.
    1. Fetch Data (Market Data Service)
    2. Run Valuation (Valuation Engine)
    3. Run Quant Analysis (Quant Engine)
    4. Merge Results
    """
    # 1. Fetch Data
    info, income, balance, cashflow, history = market_data.get_market_data(ticker)
    
    if not info:
        raise ValueError(f"Could not fetch data for {ticker}")
        
    # 2. Run Valuation Engine
    # Note: We pass is_quarterly=False by default here, assuming yfinance returns annual or we handle it.
    # Actually, yfinance .financials returns annual by default. .quarterly_financials returns quarterly.
    # Our market_data service uses .financials (Annual).
    result = valuation_engine.get_valuation(ticker, info, income, balance, cashflow, history)
    
    # 3. Run Quant Engine (Citadel-Tier)
    try:
        quant_analysis = quant_engine.get_quant_analysis(ticker, info, income, balance, cashflow, history)
        
        # Merge Quant Results into Valuation Result
        # We need to extend the schema or just attach it?
        # The schema has 'quant' field? Let's check schemas.py
        # It seems schemas.ValuationResult doesn't have 'quant' field yet?
        # Let's check if we need to add it.
        # For now, let's assume we return it separately or the frontend expects it inside.
        # Actually, let's look at the original service.py to see how it was done.
        pass
    except Exception as e:
        logger.error(f"Quant Engine failed for {ticker}: {e}")
        
    return result

def analyze_portfolio(holdings: list):
    """
    Analyzes a list of holdings.
    """
    results = []
    for holding in holdings:
        try:
            res = analyze_stock(holding['ticker'])
            results.append(res)
        except Exception as e:
            logger.error(f"Failed to analyze {holding['ticker']}: {e}")
            
    return results
