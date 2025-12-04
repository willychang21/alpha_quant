import pandas as pd
import yfinance as yf
from quant.features.volatility import VolatilityScaledMomentum
import logging

logging.basicConfig(level=logging.INFO)

async def debug_vsm():
    tickers = ["GOOGL", "GOOG", "HOOD", "NEM", "STX", "WDC"]
    vsm_gen = VolatilityScaledMomentum()
    
    print(f"Fetching data for {tickers}...")
    data = yf.download(tickers, period="2y", group_by='ticker', progress=False)
    
    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        try:
            if len(tickers) > 1:
                history = data[ticker]
            else:
                history = data
                
            print(f"History Shape: {history.shape}")
            print(f"Last Close: {history['Close'].iloc[-1]}")
            
            score = vsm_gen.compute(history)
            print(f"VSM Score: {score}")
            
            # Manual Check
            returns = history['Close'].pct_change().fillna(0)
            ret_12m = history['Close'].pct_change(periods=252).iloc[-1]
            vol_1y = returns.rolling(window=252).std().iloc[-1] * (252**0.5)
            
            print(f"Manual 12m Ret: {ret_12m}")
            print(f"Manual 1y Vol: {vol_1y}")
            print(f"Manual Ratio: {ret_12m / vol_1y}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(debug_vsm())
