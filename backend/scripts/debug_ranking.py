import asyncio
import logging
import sys
import os
from datetime import date
import pandas as pd

# Add backend to path
sys.path.append(os.getcwd())

from app.core.database import SessionLocal
from quant.selection.ranking import RankingEngine
from quant.data.models import Security

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_ranking():
    db = SessionLocal()
    
    # Create a dummy engine
    engine = RankingEngine(db)
    
    # Mock the securities list to just a few
    securities = [
        Security(sid=1, ticker="AAPL", active=True),
        Security(sid=2, ticker="MSFT", active=True),
        Security(sid=3, ticker="NVDA", active=True)
    ]
    
    # We need to monkey-patch the db query to return our mock securities
    # Or just copy the logic from run_ranking but use our list
    
    print("Testing bulk download...")
    import yfinance as yf
    tickers = [s.ticker for s in securities]
    bulk_history = yf.download(tickers, period="2y", group_by='ticker', threads=True)
    print(f"Bulk history shape: {bulk_history.shape}")
    print(f"Columns: {bulk_history.columns}")
    
    for sec in securities:
        print(f"\nProcessing {sec.ticker}...")
        if len(tickers) > 1:
            try:
                history = bulk_history[sec.ticker].copy()
                print(f"History shape: {history.shape}")
            except KeyError:
                print("KeyError in bulk_history")
                continue
        else:
            history = bulk_history.copy()
            
        if history.empty:
            print("History is empty")
            continue
            
        mom_score = engine.momentum_gen.compute(history)
        print(f"Momentum Score: {mom_score}")
        
    db.close()

if __name__ == "__main__":
    asyncio.run(test_ranking())
