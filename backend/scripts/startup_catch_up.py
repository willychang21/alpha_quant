#!/usr/bin/env python3
"""Startup Catch-Up Script

Example integration of SmartCatchUpService at application startup.
Demonstrates how to detect data gaps and perform automatic backfill.

Usage:
    python scripts/startup_catch_up.py [--dry-run]
    
Requirements: 5.3
"""

import argparse
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quant.data.integrity import (
    CatchUpResult,
    MarketCalendar,
    OHLCVValidator,
    SmartCatchUpService,
)
from quant.data.parquet_io import ParquetReader, ParquetWriter, get_data_lake_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleDataFetcher:
    """Simple yfinance data fetcher for catch-up operations."""
    
    def fetch_range(self, tickers, start, end):
        """Fetch historical data from yfinance."""
        import yfinance as yf
        import pandas as pd
        
        logger.info(f"Fetching {len(tickers)} tickers from {start} to {end}")
        
        try:
            # Download all tickers at once for efficiency
            data = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                group_by='ticker',
                auto_adjust=False,
                progress=False
            )
            
            if data.empty:
                return pd.DataFrame()
            
            # Reshape to long format
            records = []
            
            if len(tickers) == 1:
                # Single ticker case - different structure
                ticker = tickers[0]
                for date_idx in data.index:
                    row = data.loc[date_idx]
                    records.append({
                        'ticker': ticker,
                        'date': date_idx.date(),
                        'open': row['Open'],
                        'high': row['High'],
                        'low': row['Low'],
                        'close': row['Close'],
                        'adj_close': row['Adj Close'],
                        'volume': int(row['Volume'])
                    })
            else:
                # Multiple tickers
                for ticker in tickers:
                    if ticker not in data.columns.get_level_values(0):
                        continue
                    ticker_data = data[ticker]
                    for date_idx in ticker_data.index:
                        row = ticker_data.loc[date_idx]
                        if pd.isna(row['Close']):
                            continue
                        records.append({
                            'ticker': ticker,
                            'date': date_idx.date(),
                            'open': row['Open'],
                            'high': row['High'],
                            'low': row['Low'],
                            'close': row['Close'],
                            'adj_close': row['Adj Close'],
                            'volume': int(row['Volume']) if pd.notna(row['Volume']) else 0
                        })
            
            df = pd.DataFrame(records)
            logger.info(f"Fetched {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            raise


class SimpleDataProvider:
    """Simple data provider for getting ticker universe."""
    
    def __init__(self, reader: ParquetReader):
        self.reader = reader
    
    def get_universe(self, as_of_date):
        """Get unique tickers from existing data."""
        try:
            df = self.reader.read_prices(columns=['ticker'])
            if df is None or df.empty:
                # Default universe if no data
                return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
            return df['ticker'].unique().tolist()
        except Exception:
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']


def run_catch_up(dry_run: bool = False) -> CatchUpResult:
    """
    Run the catch-up process.
    
    Args:
        dry_run: If True, only check gap status without backfilling
        
    Returns:
        CatchUpResult with operation details
    """
    # Initialize components
    data_lake_path = get_data_lake_path()
    logger.info(f"Using data lake at: {data_lake_path}")
    
    reader = ParquetReader(data_lake_path)
    writer = ParquetWriter(data_lake_path)
    
    # Create service
    service = SmartCatchUpService(
        data_provider=SimpleDataProvider(reader),
        data_fetcher=SimpleDataFetcher(),
        parquet_reader=reader,
        parquet_writer=writer,
        validator=OHLCVValidator(),
        market_calendar=MarketCalendar('NYSE'),
        max_retries=3,
        drop_rate_threshold=0.10,
        initial_lookback_days=365 * 2  # 2 years for initial load
    )
    
    # Check gap status
    status = service.get_gap_status()
    logger.info(f"Gap status: {status}")
    
    if dry_run:
        logger.info("Dry run mode - skipping backfill")
        return CatchUpResult(
            ready=True,
            days_backfilled=0,
            tickers_updated=0,
            tickers_failed=0,
            rows_added=0,
            rows_dropped=0,
        )
    
    # Perform catch-up
    logger.info("Starting catch-up process...")
    ready, days = service.check_and_backfill()
    
    result = service.get_detailed_result()
    
    # Log summary
    logger.info("=" * 60)
    logger.info("CATCH-UP SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Ready: {result.ready}")
    logger.info(f"  Days backfilled: {result.days_backfilled}")
    logger.info(f"  Tickers updated: {result.tickers_updated}")
    logger.info(f"  Tickers failed: {result.tickers_failed}")
    logger.info(f"  Rows added: {result.rows_added}")
    logger.info(f"  Rows dropped: {result.rows_dropped}")
    logger.info(f"  Confirmed spikes: {len(result.confirmed_spikes)}")
    logger.info(f"  Persistent moves: {len(result.persistent_moves)}")
    if result.errors:
        logger.warning(f"  Errors: {len(result.errors)}")
        for error in result.errors[:5]:  # Show first 5 errors
            logger.warning(f"    - {error}")
    logger.info("=" * 60)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Smart Catch-Up Service for data gap detection and backfill"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check gap status without performing backfill"
    )
    args = parser.parse_args()
    
    try:
        result = run_catch_up(dry_run=args.dry_run)
        sys.exit(0 if result.ready else 1)
    except Exception as e:
        logger.error(f"Catch-up failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
