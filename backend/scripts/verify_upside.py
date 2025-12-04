import asyncio
import logging
import pandas as pd
from unittest.mock import MagicMock
from quant.selection.ranking import RankingEngine
from quant.data.models import Security

logging.basicConfig(level=logging.INFO)

async def verify_upside():
    # Mock DB Session
    db = MagicMock()
    
    # Mock Securities
    securities = [
        Security(sid=1, ticker="AAPL", active=True),
        Security(sid=2, ticker="GOOGL", active=True),
        Security(sid=3, ticker="F", active=True),
        Security(sid=4, ticker="VZ", active=True),
        Security(sid=5, ticker="BRK-B", active=True)
    ]
    
    # Mock Query
    db.query.return_value.filter.return_value.all.return_value = securities
    
    engine = RankingEngine(db)
    
    print("Running Ranking for AAPL and GOOGL...")
    df = await engine.run_ranking(pd.Timestamp.now().date())
    
    if df is not None and not df.empty:
        print("\nResult DataFrame:")
        print(df[['ticker', 'upside', 'score']])
        
        if (df['upside'] != 0).any():
            print("\nSUCCESS: Upside is being calculated!")
        else:
            print("\nFAILURE: Upside is still 0.0")
    else:
        print("\nFAILURE: No results returned.")

if __name__ == "__main__":
    asyncio.run(verify_upside())
