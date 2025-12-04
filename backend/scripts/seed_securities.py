import pandas as pd
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from quant.data.models import Security
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import requests
from io import StringIO

def get_sp500_tickers():
    """Fetch S&P 500 tickers from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        tables = pd.read_html(StringIO(response.text))
        df = tables[0]
        tickers = df['Symbol'].tolist()
        # Clean tickers (e.g., BRK.B -> BRK-B for yfinance)
        tickers = [t.replace('.', '-') for t in tickers]
        logger.info(f"Fetched {len(tickers)} S&P 500 tickers.")
        return tickers
    except Exception as e:
        logger.error(f"Error fetching S&P 500: {e}")
        return []

def get_nasdaq100_tickers():
    """Fetch Nasdaq 100 tickers from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        tables = pd.read_html(StringIO(response.text))
        # The table index might vary, usually it's the 4th or 5th table
        # Look for a table with 'Ticker' or 'Symbol' column
        for table in tables:
            if 'Ticker' in table.columns:
                return table['Ticker'].tolist()
            if 'Symbol' in table.columns:
                return table['Symbol'].tolist()
        
        logger.warning("Could not find Nasdaq 100 table.")
        return []
    except Exception as e:
        logger.error(f"Error fetching Nasdaq 100: {e}")
        return []

def seed_securities():
    db: Session = SessionLocal()
    
    logger.info("Fetching ticker lists...")
    sp500 = get_sp500_tickers()
    nasdaq = get_nasdaq100_tickers()
    
    # Combine and deduplicate
    all_tickers = sorted(list(set(sp500 + nasdaq)))
    logger.info(f"Total unique tickers to seed: {len(all_tickers)}")
    
    added_count = 0
    for ticker in all_tickers:
        exists = db.query(Security).filter(Security.ticker == ticker).first()
        if not exists:
            sec = Security(ticker=ticker, name=ticker, type="Equity")
            db.add(sec)
            added_count += 1
    
    db.commit()
    logger.info(f"Successfully added {added_count} new securities.")
    db.close()

if __name__ == "__main__":
    seed_securities()
