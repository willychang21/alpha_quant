import sys
import os
import pandas as pd
import yfinance as yf
from sqlalchemy import func
from datetime import date

# Add backend to path
sys.path.append(os.getcwd())

from app.core.database import SessionLocal
from quant.data.models import Security, ModelSignals

def debug_missing():
    db = SessionLocal()
    
    # Get all active tickers
    all_tickers = [s.ticker for s in db.query(Security).filter(Security.active == True).all()]
    print(f"Total Active Tickers: {len(all_tickers)}")
    
    # Get tickers with signals for today (or latest date)
    latest_date = db.query(func.max(ModelSignals.date)).scalar()
    print(f"Latest Date: {latest_date}")
    
    existing_signals = db.query(ModelSignals).filter(ModelSignals.date == latest_date).all()
    existing_tickers = set([s.security.ticker for s in existing_signals]) # Need to join? No, ModelSignals has sid.
    
    # Wait, ModelSignals has sid, need to map to ticker
    # Let's just get sids
    existing_sids = [s.sid for s in existing_signals]
    existing_tickers = [s.ticker for s in db.query(Security).filter(Security.sid.in_(existing_sids)).all()]
    
    print(f"Tickers with Signals: {len(existing_tickers)}")
    
    missing_tickers = list(set(all_tickers) - set(existing_tickers))
    print(f"Missing Tickers: {len(missing_tickers)}")
    
    if missing_tickers:
        print(f"Sample Missing: {missing_tickers[:5]}")
        
        # Test one missing ticker
        test_ticker = missing_tickers[0]
        print(f"\nTesting download for missing ticker: {test_ticker}")
        
        data = yf.download([test_ticker], period="2y", group_by='ticker', threads=True)
        print(f"Download Shape: {data.shape}")
        if not data.empty:
            print("Columns:", data.columns)
            try:
                # Check if we can access it like RankingEngine does
                if isinstance(data.columns, pd.MultiIndex):
                     hist = data[test_ticker]
                else:
                     hist = data
                print(f"History Shape: {hist.shape}")
                print("Head:", hist.head())
            except Exception as e:
                print(f"Access Error: {e}")
        else:
            print("Download returned empty dataframe.")

    db.close()

if __name__ == "__main__":
    debug_missing()
