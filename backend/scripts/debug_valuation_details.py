import asyncio
import logging
import pandas as pd
from quant.valuation.orchestrator import ValuationOrchestrator
from core.adapters.yfinance_provider import YFinanceProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_valuation():
    tickers = ["F"]
    provider = YFinanceProvider()
    orchestrator = ValuationOrchestrator()
    
    for ticker in tickers:
        print(f"\n--- Debugging {ticker} ---")
        try:
            data = await provider.get_ticker_data(ticker)
            
            info = data.get('info', {})
            price = info.get('currentPrice')
            print(f"Current Price: {price}")
            
            # Check Financials
            income = data.get('income')
            if income is not None and not income.empty:
                print(f"Income Stmt Shape: {income.shape}")
                print(f"Latest Net Income: {income.iloc[0, 0] if not income.empty else 'N/A'}")
            else:
                print("Income Stmt: Empty/None")
                
            valuation_data = {
                'info': info,
                'income_stmt': data.get('income'),
                'balance_sheet': data.get('balance'),
                'cashflow': data.get('cashflow')
            }
            
            result = orchestrator.get_valuation(ticker, valuation_data)
            
            if result:
                print(f"Model: {result.model_name}")
                print(f"Fair Value: {result.fair_value}")
                print(f"Upside: {result.upside}")
                print(f"Details: {result.details}")
            else:
                print("Valuation Result: None")
                
        except Exception as e:
            logger.error(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(debug_valuation())
