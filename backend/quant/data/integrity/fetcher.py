"""YFinance data fetcher for SmartCatchUpService.

Provides a simple wrapper around yfinance that conforms to the
DataFetcherProtocol expected by SmartCatchUpService.
"""

import logging
from datetime import date
from typing import List

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class YFinanceFetcher:
    """
    YFinance data fetcher that conforms to DataFetcherProtocol.
    
    Fetches OHLCV data from Yahoo Finance and returns it in the
    format expected by the validation framework.
    """
    
    def __init__(self, threads: bool = True, progress: bool = False):
        """
        Initialize the fetcher.
        
        Args:
            threads: Use multi-threading for faster downloads
            progress: Show download progress bar
        """
        self.threads = threads
        self.progress = progress
    
    def fetch_range(
        self, 
        tickers: List[str], 
        start: date, 
        end: date
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for tickers in date range.
        
        Args:
            tickers: List of ticker symbols
            start: Start date (inclusive)
            end: End date (inclusive)
            
        Returns:
            DataFrame with columns: ticker, date, open, high, low, close, adj_close, volume
        """
        logger.info(f"Fetching {len(tickers)} tickers from {start} to {end}")
        
        # Download from yfinance
        data = yf.download(
            tickers,
            start=start.isoformat(),
            end=end.isoformat(),
            group_by='ticker',
            progress=self.progress,
            threads=self.threads
        )
        
        if data.empty:
            logger.warning("No data returned from yfinance")
            return pd.DataFrame()
        
        # Convert to long format with our expected columns
        records = []
        
        # Handle single ticker vs multi-ticker response
        if len(tickers) == 1:
            ticker = tickers[0]
            df = data.reset_index()
            for _, row in df.iterrows():
                if pd.isna(row.get('Close')):
                    continue
                records.append(self._row_to_record(row, ticker))
        else:
            # Multi-ticker: data has MultiIndex columns
            for ticker in tickers:
                try:
                    if ticker not in data.columns.get_level_values(0):
                        continue
                    ticker_df = data[ticker].reset_index()
                    for _, row in ticker_df.iterrows():
                        if pd.isna(row.get('Close')):
                            continue
                        records.append(self._row_to_record(row, ticker))
                except Exception as e:
                    logger.warning(f"Error processing {ticker}: {e}")
                    continue
        
        if not records:
            return pd.DataFrame()
        
        result = pd.DataFrame(records)
        logger.info(f"Fetched {len(result)} records for {result['ticker'].nunique()} tickers")
        return result
    
    def _row_to_record(self, row: pd.Series, ticker: str) -> dict:
        """Convert a yfinance row to our record format."""
        return {
            'ticker': ticker,
            'date': row['Date'].date() if hasattr(row['Date'], 'date') else row['Date'],
            'open': float(row['Open']) if pd.notna(row['Open']) else None,
            'high': float(row['High']) if pd.notna(row['High']) else None,
            'low': float(row['Low']) if pd.notna(row['Low']) else None,
            'close': float(row['Close']) if pd.notna(row['Close']) else None,
            'adj_close': float(row['Close']) if pd.notna(row['Close']) else None,  # yfinance auto-adjusts
            'volume': int(row['Volume']) if pd.notna(row['Volume']) else 0
        }
