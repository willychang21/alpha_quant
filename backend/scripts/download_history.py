"""
Historical Data Download Script
Downloads 5 years of S&P 500 price data for walk-forward backtesting.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import text
import logging

from app.core.database import SessionLocal
from quant.data.models import Security, MarketDataDaily

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_sp500_tickers() -> list:
    """Fetch S&P 500 tickers from Wikipedia."""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
        return tickers
    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 list: {e}")
        # Fallback to cached list
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK-B', 'JPM', 'JNJ']


def download_price_history(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download historical price data for multiple tickers.
    
    Args:
        tickers: List of stock symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with columns: date, ticker, open, high, low, close, volume
    """
    logger.info(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")
    
    all_data = []
    
    # Download in batches to avoid rate limits
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        logger.info(f"Downloading batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}")
        
        try:
            df = yf.download(
                batch,
                start=start_date,
                end=end_date,
                group_by='ticker',
                auto_adjust=True,
                progress=False
            )
            
            if df.empty:
                continue
                
            # Reshape multi-index to long format
            for ticker in batch:
                try:
                    if len(batch) == 1:
                        ticker_df = df.copy()
                    else:
                        if ticker not in df.columns.get_level_values(0):
                            continue
                        ticker_df = df[ticker].copy()
                    
                    ticker_df = ticker_df.reset_index()
                    ticker_df['ticker'] = ticker
                    ticker_df.columns = [c.lower() for c in ticker_df.columns]
                    ticker_df = ticker_df.rename(columns={'date': 'date'})
                    
                    # Keep only needed columns
                    cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
                    ticker_df = ticker_df[[c for c in cols if c in ticker_df.columns]]
                    
                    all_data.append(ticker_df)
                except Exception as e:
                    logger.warning(f"Failed to process {ticker}: {e}")
                    
        except Exception as e:
            logger.error(f"Batch download failed: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    result = pd.concat(all_data, ignore_index=True)
    logger.info(f"Downloaded {len(result)} price records")
    return result


def store_price_data(df: pd.DataFrame, db):
    """
    Store price data in MarketDataDaily table.
    Uses bulk insert for efficiency.
    """
    if df.empty:
        return
    
    # Get ticker -> sid mapping
    securities = db.query(Security.sid, Security.ticker).all()
    ticker_to_sid = {s.ticker: s.sid for s in securities}
    
    # Filter to known tickers
    df = df[df['ticker'].isin(ticker_to_sid.keys())]
    df['sid'] = df['ticker'].map(ticker_to_sid)
    
    logger.info(f"Storing {len(df)} records for {df['ticker'].nunique()} tickers...")
    
    # Delete existing data in date range
    min_date = pd.to_datetime(df['date'].min()).date()
    max_date = pd.to_datetime(df['date'].max()).date()
    
    db.execute(
        text("""
            DELETE FROM market_data_daily 
            WHERE date >= :min_date AND date <= :max_date
        """),
        {'min_date': str(min_date), 'max_date': str(max_date)}
    )
    db.commit()
    
    # Insert new data
    records = []
    for _, row in df.iterrows():
        records.append(MarketDataDaily(
            sid=row['sid'],
            date=row['date'].date() if hasattr(row['date'], 'date') else row['date'],
            open=row.get('open'),
            high=row.get('high'),
            low=row.get('low'),
            close=row.get('close'),
            volume=row.get('volume')
        ))
    
    db.bulk_save_objects(records)
    db.commit()
    logger.info(f"✅ Stored {len(records)} price records")


def main():
    """Main entry point."""
    db = SessionLocal()
    
    # Calculate date range (5 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    logger.info("=" * 60)
    logger.info("  HISTORICAL DATA DOWNLOAD")
    logger.info("=" * 60)
    logger.info(f"Period: {start_date} to {end_date}")
    
    # Get tickers from database (use existing securities)
    securities = db.query(Security.ticker).all()
    tickers = [s.ticker for s in securities]
    
    if not tickers:
        logger.warning("No securities in database. Fetching S&P 500 list...")
        tickers = get_sp500_tickers()
    
    logger.info(f"Tickers to download: {len(tickers)}")
    
    # Download price data
    price_df = download_price_history(tickers, start_date, end_date)
    
    if price_df.empty:
        logger.error("No data downloaded!")
        db.close()
        return
    
    # Store in database
    store_price_data(price_df, db)
    
    # Summary stats
    unique_dates = price_df['date'].nunique()
    unique_tickers = price_df['ticker'].nunique()
    
    logger.info("=" * 60)
    logger.info("  DOWNLOAD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Records: {len(price_df)}")
    logger.info(f"Tickers: {unique_tickers}")
    logger.info(f"Trading Days: {unique_dates}")
    
    db.close()


if __name__ == "__main__":
    main()
