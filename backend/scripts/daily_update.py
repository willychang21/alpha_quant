#!/usr/bin/env python3
"""
Daily Data Update Service

Fetches latest market data from yfinance and updates both SQLite and Parquet data lakes.
Designed to be run as a daily cron job or scheduled task.

Usage:
    # Run manually
    cd backend
    python scripts/daily_update.py
    
    # Cron job (run at 6 PM EST after market close)
    0 18 * * 1-5 cd /path/to/backend && python scripts/daily_update.py >> logs/daily_update.log 2>&1

Features:
- Incremental update (only fetches missing days)
- Dual write to SQLite + Parquet
- Automatic retry with exponential backoff
- Email/Slack notification on failure (configurable)
"""

import os
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
import argparse
import logging
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import yfinance as yf

# Setup logging
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / f'daily_update_{date.today().isoformat()}.log')
    ]
)
logger = logging.getLogger(__name__)


class DailyUpdateService:
    """
    Service for daily data updates.
    
    Supports:
    - Market prices (OHLCV)
    - Fundamentals (quarterly updates)
    - Parquet data lake storage
    """
    
    def __init__(
        self,
        update_parquet: bool = True,
        max_retries: int = 3
    ):
        self.update_parquet = update_parquet
        self.max_retries = max_retries
        
        # Stats
        self.stats = {
            'tickers_updated': 0,
            'rows_added': 0,
            'errors': [],
            'start_time': datetime.now()
        }
        
        # Initialize Parquet writer
        if update_parquet:
            from quant.data.parquet_io import ParquetWriter, get_data_lake_path
            self.parquet_writer = ParquetWriter(str(get_data_lake_path()))
            self.data_lake_path = get_data_lake_path()
    
    def get_universe(self) -> list:
        """Get list of tickers to update."""
        # Default: S&P 500 top 50
        default_tickers = [
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "LLY", "AVGO",
            "JPM", "XOM", "UNH", "V", "PG", "MA", "COST", "JNJ", "HD", "MRK",
            "ABBV", "CVX", "BAC", "WMT", "CRM", "AMD", "PEP", "KO", "NFLX", "TMO",
            "ADBE", "WFC", "LIN", "ACN", "MCD", "DIS", "CSCO", "ABT", "INTC", "VZ",
            "CMCSA", "INTU", "QCOM", "IBM", "TXN", "AMGN", "NOW", "GE", "SPGI", "CAT"
        ]
        
        # Try to get additional tickers from Parquet securities
        if self.update_parquet:
            try:
                import duckdb
                securities_path = self.data_lake_path / 'raw' / 'securities'
                if securities_path.exists():
                    conn = duckdb.connect(':memory:')
                    result = conn.execute(f"""
                        SELECT ticker FROM read_parquet('{securities_path}/*.parquet')
                        WHERE active = true OR active IS NULL
                    """).fetchall()
                    db_tickers = [r[0] for r in result]
                    return sorted(list(set(default_tickers + db_tickers)))
            except Exception as e:
                logger.warning(f"Could not read securities from Parquet: {e}")
        
        return sorted(default_tickers)
    
    def get_last_date(self, ticker: str) -> date:
        """Get the last date we have data for this ticker."""
        if self.update_parquet:
            try:
                import duckdb
                
                prices_path = self.data_lake_path / 'raw' / 'prices'
                
                if prices_path.exists():
                    conn = duckdb.connect(':memory:')
                    result = conn.execute(f"""
                        SELECT MAX(date) FROM read_parquet('{prices_path}/**/*.parquet')
                        WHERE ticker = '{ticker}'
                    """).fetchone()
                    
                    if result and result[0]:
                        if isinstance(result[0], str):
                            return datetime.strptime(result[0], '%Y-%m-%d').date()
                        return result[0]
            except Exception as e:
                logger.debug(f"Error checking last date for {ticker}: {e}")
        
        return date.today() - timedelta(days=7)
    
    def fetch_incremental(self, tickers: list, days_back: int = 5) -> pd.DataFrame:
        """
        Fetch only recent data (incremental update).
        
        Args:
            tickers: List of tickers to fetch
            days_back: Number of days to fetch (buffer for weekends/holidays)
        """
        start_date = (date.today() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        logger.info(f"Fetching data for {len(tickers)} tickers from {start_date}...")
        
        for attempt in range(self.max_retries):
            try:
                data = yf.download(
                    tickers,
                    start=start_date,
                    group_by='ticker',
                    progress=False,
                    threads=True
                )
                return data
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
    
    def update_prices(self, days_back: int = 5):
        """Update price data."""
        logger.info("=" * 50)
        logger.info("Starting price update...")
        
        tickers = self.get_universe()
        
        # Batch to avoid overwhelming yfinance
        batch_size = 50
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}...")
            
            try:
                data = self.fetch_incremental(batch, days_back)
                
                if data.empty:
                    continue
                
                # Process each ticker
                for ticker in batch:
                    try:
                        # Handle single vs multi-ticker response
                        if len(batch) == 1:
                            df = data
                        else:
                            if ticker not in data.columns.levels[0]:
                                continue
                            df = data[ticker]
                        
                        if df.empty:
                            continue
                        
                        # Reset index
                        df = df.reset_index()
                        df['ticker'] = ticker
                        
                        # Get last date in our data
                        last_date = self.get_last_date(ticker)
                        
                        # Filter to only new rows
                        df['date'] = pd.to_datetime(df['Date']).dt.date
                        df = df[df['date'] > last_date]
                        
                        if df.empty:
                            continue
                        
                        # Prepare data
                        records = []
                        for _, row in df.iterrows():
                            if pd.isna(row['Close']):
                                continue
                            records.append({
                                'ticker': ticker,
                                'date': row['date'],
                                'open': row['Open'],
                                'high': row['High'],
                                'low': row['Low'],
                                'close': row['Close'],
                                'adj_close': row['Close'],  # yfinance auto-adjusts
                                'volume': int(row['Volume']) if not pd.isna(row['Volume']) else 0
                            })
                        
                        if not records:
                            continue
                        
                        # Write to Parquet
                        if self.update_parquet:
                            self._write_parquet_prices(records)
                        
                        self.stats['tickers_updated'] += 1
                        self.stats['rows_added'] += len(records)
                        
                    except Exception as e:
                        logger.error(f"Error processing {ticker}: {e}")
                        self.stats['errors'].append(f"{ticker}: {e}")
                
            except Exception as e:
                logger.error(f"Batch fetch failed: {e}")
                self.stats['errors'].append(str(e))
    
    def _write_parquet_prices(self, records: list):
        """Write prices to Parquet."""
        df = pd.DataFrame(records)
        self.parquet_writer.write_prices(df, mode='append')
    
    def report(self):
        """Print summary report."""
        duration = (datetime.now() - self.stats['start_time']).total_seconds()
        
        logger.info("=" * 50)
        logger.info("Daily Update Complete")
        logger.info("=" * 50)
        logger.info(f"Duration: {duration:.1f} seconds")
        logger.info(f"Tickers updated: {self.stats['tickers_updated']}")
        logger.info(f"Rows added: {self.stats['rows_added']}")
        
        if self.stats['errors']:
            logger.warning(f"Errors: {len(self.stats['errors'])}")
            for e in self.stats['errors'][:5]:
                logger.warning(f"  - {e}")
        else:
            logger.info("No errors ✓")
    
    def close(self):
        """Cleanup (no-op for Parquet-only mode)."""
        pass


def main():
    parser = argparse.ArgumentParser(description='Daily data update to Parquet')
    parser.add_argument('--days-back', type=int, default=5,
                        help='Days of data to fetch (default: 5)')
    args = parser.parse_args()
    
    service = DailyUpdateService()
    
    try:
        service.update_prices(days_back=args.days_back)
        service.report()
    finally:
        service.close()
    
    return 0 if len(service.stats['errors']) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

