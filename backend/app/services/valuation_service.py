import logging
import asyncio
from quant.valuation.service import ValuationService
from app.engines.backtest import engine as backtest_engine
from app.services.market_data import MarketDataService
from app.domain import schemas

logger = logging.getLogger(__name__)
market_data_service = MarketDataService()
# Initialize Quant Service (DB not strictly needed for pure valuation logic yet)
quant_valuation_service = ValuationService(db=None)

async def analyze_stock(ticker: str) -> schemas.ValuationResult:
    try:
        # 1. Fetch all market data in parallel (async)
        data = await market_data_service.get_ticker_data(ticker)
        
        info = data['info']
        income = data['income']
        balance = data['balance']
        cashflow = data['cashflow']
        history = data['history']
        
        # 2. Handle Currency Conversion (Async)
        currency = info.get('currency', 'USD')
        fin_currency = info.get('financialCurrency', currency)
        exchange_rate = 1.0
        
        if currency != fin_currency:
            exchange_rate = await market_data_service.get_exchange_rate(fin_currency, currency)
            
        # 3. Call Valuation Engine (New Quant Library)
        return quant_valuation_service.get_valuation(
            ticker, info, income, balance, cashflow, history, 
            is_quarterly=False, # Default
            exchange_rate=exchange_rate
        )
        
    except Exception as e:
        logger.error(f"Valuation error for {ticker}: {e}")
        raise e

def analyze_portfolio(request: schemas.AnalysisRequest) -> schemas.AnalysisResponse:
    try:
        # Backtest engine is still sync for now, can be optimized later
        return backtest_engine.analyze_portfolio(request)
    except Exception as e:
        logger.error(f"Portfolio analysis error: {e}")
        raise e
