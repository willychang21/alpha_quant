"""YFinance data fetcher for SmartCatchUpService.

Provides a simple wrapper around yfinance that conforms to the
DataFetcherProtocol expected by SmartCatchUpService.

Includes circuit breaker protection for API resilience.
"""

import logging
from datetime import date
from typing import List, Optional

import pandas as pd
import yfinance as yf

from core.circuit_breaker import CircuitBreaker, CircuitOpenError, get_yfinance_circuit_breaker

logger = logging.getLogger(__name__)


class YFinanceFetcher:
    """
    YFinance data fetcher that conforms to DataFetcherProtocol.
    
    Fetches OHLCV data from Yahoo Finance and returns it in the
    format expected by the validation framework.
    """
    
    def __init__(
        self, 
        threads: bool = True, 
        progress: bool = False,
        circuit_breaker: Optional[CircuitBreaker] = None
    ):
        """
        Initialize the fetcher.
        
        Args:
            threads: Use multi-threading for faster downloads
            progress: Show download progress bar
            circuit_breaker: Optional circuit breaker (uses default if None)
        """
        self.threads = threads
        self.progress = progress
        self._circuit_breaker = circuit_breaker
    
    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Get circuit breaker, using default if not set."""
        if self._circuit_breaker is None:
            self._circuit_breaker = get_yfinance_circuit_breaker()
        return self._circuit_breaker
    
    def fetch_range(
        self, 
        tickers: List[str], 
        start: date, 
        end: date
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for tickers in date range.
        
        Uses circuit breaker to handle yfinance API failures gracefully.
        
        Args:
            tickers: List of ticker symbols
            start: Start date (inclusive)
            end: End date (inclusive)
            
        Returns:
            DataFrame with columns: ticker, date, open, high, low, close, adj_close, volume
            
        Raises:
            CircuitOpenError: If circuit breaker is open
        """
        logger.info(f"Fetching {len(tickers)} tickers from {start} to {end}")
        
        # Use circuit breaker to protect yfinance call
        data = self.circuit_breaker.call(
            self._download_from_yfinance,
            tickers, start, end
        )
        
        return self._process_yfinance_response(data, tickers)
    
    def _download_from_yfinance(
        self,
        tickers: List[str],
        start: date,
        end: date
    ) -> pd.DataFrame:
        """Raw yfinance download (wrapped by circuit breaker)."""
        return yf.download(
            tickers,
            start=start.isoformat(),
            end=end.isoformat(),
            group_by='ticker',
            progress=self.progress,
            threads=self.threads
        )
    
    def _process_yfinance_response(
        self,
        data: pd.DataFrame,
        tickers: List[str]
    ) -> pd.DataFrame:
        """Process yfinance response into standard format."""
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
