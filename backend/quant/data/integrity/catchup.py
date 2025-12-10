"""SmartCatchUpService for automatic gap detection and backfill.

This service handles automatic data gap detection and backfill on system startup.
Use case: Development server not running 24/7, needs to catch up on missed 
trading days before becoming operational.

Requirements: 1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 3.1, 4.1, 5.1, 5.2
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Optional, Protocol, Tuple

import pandas as pd

from quant.data.integrity.calendar import MarketCalendar
from quant.data.integrity.enums import ValidationContext
from quant.data.integrity.exceptions import CatchUpError
from quant.data.integrity.models import ValidationReport
from quant.data.integrity.processor import ActionProcessor
from quant.data.integrity.validator import DataValidator

logger = logging.getLogger(__name__)


@dataclass
class CatchUpResult:
    """Result of a catch-up operation.
    
    Provides comprehensive statistics about the backfill operation including
    success metrics, failure counts, and spike classifications.
    
    Requirements: 4.1, 4.2, 4.3, 4.4
    """
    ready: bool
    days_backfilled: int
    tickers_updated: int
    tickers_failed: int
    rows_added: int
    rows_dropped: int
    confirmed_spikes: List[dict] = field(default_factory=list)
    persistent_moves: List[dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "ready": self.ready,
            "days_backfilled": self.days_backfilled,
            "tickers_updated": self.tickers_updated,
            "tickers_failed": self.tickers_failed,
            "rows_added": self.rows_added,
            "rows_dropped": self.rows_dropped,
            "confirmed_spikes": self.confirmed_spikes,
            "persistent_moves": self.persistent_moves,
            "errors": self.errors,
        }


class DataProviderProtocol(Protocol):
    """Protocol for data providers used by SmartCatchUpService."""
    
    def get_universe(self, as_of_date: date) -> List[str]:
        """Get available ticker universe as of given date."""
        ...


class DataFetcherProtocol(Protocol):
    """Protocol for fetching historical data (e.g., yfinance wrapper)."""
    
    def fetch_range(
        self, 
        tickers: List[str], 
        start: date, 
        end: date
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for tickers in date range."""
        ...


class ParquetWriterProtocol(Protocol):
    """Protocol for Parquet writers."""
    
    def write_prices(
        self, 
        df: pd.DataFrame, 
        mode: str = 'append'
    ) -> dict:
        """Write price data to Parquet."""
        ...


class ParquetReaderProtocol(Protocol):
    """Protocol for Parquet readers."""
    
    def read_prices(
        self, 
        tickers: List[str] = None,
        start_date: date = None,
        end_date: date = None,
        columns: List[str] = None
    ) -> pd.DataFrame:
        """Read price data from Parquet."""
        ...


class SmartCatchUpService:
    """
    Handles automatic data gap detection and backfill on system startup.
    
    Use case: Development server not running 24/7, needs to catch up
    on missed trading days before becoming operational.
    
    Key features:
    - Uses NYSE calendar for accurate business day detection
    - Validates backfilled data with BACKFILL context (enables spike verification)
    - Applies actions and writes clean data to Parquet
    - Ticker-level error isolation (partial failures don't block entire process)
    - Exponential backoff retry for network errors
    
    Requirements: 1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 3.1, 4.1, 5.1, 5.2
    """
    
    def __init__(
        self,
        data_provider: DataProviderProtocol,
        data_fetcher: DataFetcherProtocol,
        parquet_reader: ParquetReaderProtocol,
        parquet_writer: ParquetWriterProtocol,
        validator: DataValidator,
        processor: Optional[ActionProcessor] = None,
        market_calendar: Optional[MarketCalendar] = None,
        max_retries: int = 3,
        drop_rate_threshold: float = 0.10,
        initial_lookback_days: int = 365 * 2
    ):
        """Initialize the SmartCatchUpService.
        
        Args:
            data_provider: Provider for getting ticker universe
            data_fetcher: Fetcher for downloading historical data (e.g., yfinance)
            parquet_reader: Reader for checking existing data
            parquet_writer: Writer for persisting backfilled data
            validator: DataValidator for validating fetched data
            processor: ActionProcessor for applying validation actions
            market_calendar: MarketCalendar instance (default: NYSE)
            max_retries: Maximum retry attempts for network errors
            drop_rate_threshold: Alert threshold for drop rate
            initial_lookback_days: Days to fetch if data lake is empty
            
        Requirements: 5.1
        """
        self.data_provider = data_provider
        self.data_fetcher = data_fetcher
        self.reader = parquet_reader
        self.writer = parquet_writer
        self.validator = validator
        self.processor = processor or ActionProcessor()
        self.calendar = market_calendar or MarketCalendar('NYSE')
        self.max_retries = max_retries
        self.drop_rate_threshold = drop_rate_threshold
        self.initial_lookback_days = initial_lookback_days
        
        # Store last result for get_detailed_result()
        self._last_result: Optional[CatchUpResult] = None
    
    def check_and_backfill(
        self, 
        tickers: Optional[List[str]] = None,
        max_gap_days: int = 30
    ) -> Tuple[bool, int]:
        """
        Check for data gaps and backfill if needed.
        
        Args:
            tickers: Optional list of tickers to check. If None, uses full universe.
            max_gap_days: Maximum gap size to attempt backfill (safety limit)
        
        Returns:
            Tuple of (ready: bool, days_backfilled: int)
        
        Raises:
            CatchUpError: If backfill fails critically
            
        Requirements: 5.2
        """
        try:
            # Get last date in data lake
            last_date = self._get_max_date()
            today = date.today()
            
            # Handle empty data lake case (Requirement 5.4)
            if last_date is None:
                logger.warning("No existing data found in data lake - performing initial load")
                backfill_start = today - timedelta(days=self.initial_lookback_days)
                gap_days = self.calendar.count_trading_days(backfill_start, today)
                
                result = self._perform_backfill(
                    start=backfill_start,
                    end=today,
                    tickers=tickers
                )
                self._last_result = result
                return result.ready, gap_days
            
            # Calculate business day gap
            gap_days = self._calculate_business_day_gap(last_date, today)
            
            logger.info(f"Data lake last date: {last_date}, gap: {gap_days} business days")
            
            if gap_days <= 0:
                logger.info("No gap detected, data is up to date")
                self._last_result = CatchUpResult(
                    ready=True,
                    days_backfilled=0,
                    tickers_updated=0,
                    tickers_failed=0,
                    rows_added=0,
                    rows_dropped=0,
                )
                return True, 0
            
            if gap_days > max_gap_days:
                logger.warning(
                    f"Gap of {gap_days} days exceeds max_gap_days ({max_gap_days}). "
                    "Manual intervention may be required."
                )
                # Still attempt backfill but cap at max_gap_days
                backfill_start = today - timedelta(days=max_gap_days * 2)
            else:
                backfill_start = last_date + timedelta(days=1)
            
            # Perform backfill
            result = self._perform_backfill(
                start=backfill_start,
                end=today,
                tickers=tickers
            )
            result.days_backfilled = gap_days
            self._last_result = result
            
            return result.ready, gap_days
            
        except CatchUpError:
            raise
        except Exception as e:
            logger.error(f"Catch-up failed: {e}")
            raise CatchUpError(
                f"Failed to check and backfill data: {e}",
                details={"error": str(e)}
            )
    
    def get_detailed_result(self) -> CatchUpResult:
        """
        Get detailed result of the last catch-up operation.
        
        Returns:
            CatchUpResult with full statistics
            
        Requirements: 4.1, 4.3, 4.4
        """
        if self._last_result is None:
            return CatchUpResult(
                ready=False,
                days_backfilled=0,
                tickers_updated=0,
                tickers_failed=0,
                rows_added=0,
                rows_dropped=0,
                errors=["No catch-up operation has been performed yet"]
            )
        return self._last_result
    
    def _get_max_date(self) -> Optional[date]:
        """
        Get the most recent date in the data lake.
        
        Returns:
            Most recent date, or None if no data exists
            
        Requirements: 1.1
        """
        try:
            # Read a small sample to get the max date
            df = self.reader.read_prices(columns=["date"])
            
            if df is None or df.empty:
                return None
            
            max_date = pd.to_datetime(df["date"]).max()
            return max_date.date() if pd.notna(max_date) else None
            
        except Exception as e:
            logger.warning(f"Could not read max date: {e}")
            return None
    
    def _calculate_business_day_gap(self, last_date: date, target_date: date) -> int:
        """
        Calculate number of missing business days.
        
        Uses NYSE calendar to exclude weekends and holidays.
        
        Args:
            last_date: Last recorded date in data lake
            target_date: Target date (usually today)
            
        Returns:
            Number of missing business days
            
        Requirements: 1.2, 1.3
        """
        if last_date >= target_date:
            return 0
        
        # Get trading days from day after last_date to target_date
        trading_days = self.calendar.valid_days(
            last_date + timedelta(days=1), 
            target_date
        )
        return len(trading_days)
    
    def _fetch_with_retry(
        self, 
        tickers: List[str], 
        start_date: date, 
        end_date: date
    ) -> pd.DataFrame:
        """
        Fetch data from yfinance with exponential backoff retry.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for fetch
            end_date: End date for fetch
            
        Returns:
            DataFrame with fetched data
            
        Raises:
            Exception: If all retries exhausted
            
        Requirements: 3.3
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                df = self.data_fetcher.fetch_range(
                    tickers=tickers,
                    start=start_date,
                    end=end_date
                )
                return df
            except Exception as e:
                last_error = e
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                logger.warning(
                    f"Fetch attempt {attempt + 1}/{self.max_retries} failed: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
        
        # All retries exhausted
        logger.error(f"All {self.max_retries} fetch attempts failed")
        raise last_error
    
    def _perform_backfill(
        self, 
        start: date, 
        end: date,
        tickers: Optional[List[str]] = None
    ) -> CatchUpResult:
        """
        Fetch and validate historical data for the gap period.
        
        Uses BACKFILL context for validation, enabling spike reversion verification.
        Implements ticker-level error isolation.
        
        Args:
            start: Start date of backfill period
            end: End date of backfill period
            tickers: Optional list of tickers (uses universe if None)
        
        Returns:
            CatchUpResult with operation statistics
        
        Requirements: 2.1, 2.2, 2.3, 2.4, 3.1
        """
        errors: List[str] = []
        tickers_updated = 0
        tickers_failed = 0
        total_rows_added = 0
        total_rows_dropped = 0
        all_confirmed_spikes: List[dict] = []
        all_persistent_moves: List[dict] = []
        
        # Get ticker universe if not provided
        if tickers is None:
            try:
                tickers = self.data_provider.get_universe(as_of_date=end)
                logger.info(f"Using universe of {len(tickers)} tickers")
            except Exception as e:
                errors.append(f"Failed to get ticker universe: {e}")
                return CatchUpResult(
                    ready=False,
                    days_backfilled=0,
                    tickers_updated=0,
                    tickers_failed=0,
                    rows_added=0,
                    rows_dropped=0,
                    errors=errors
                )
        
        if not tickers:
            return CatchUpResult(
                ready=True,
                days_backfilled=0,
                tickers_updated=0,
                tickers_failed=0,
                rows_added=0,
                rows_dropped=0,
                errors=["No tickers available for backfill"]
            )
        
        logger.info(f"Backfilling {len(tickers)} tickers from {start} to {end}")
        
        # Fetch historical data with retry (Requirement 3.3)
        try:
            df = self._fetch_with_retry(
                tickers=tickers,
                start_date=start,
                end_date=end
            )
        except Exception as e:
            errors.append(f"Failed to fetch historical data: {e}")
            return CatchUpResult(
                ready=False,
                days_backfilled=0,
                tickers_updated=0,
                tickers_failed=len(tickers),
                rows_added=0,
                rows_dropped=0,
                errors=errors
            )
        
        if df is None or df.empty:
            logger.warning("No data fetched for backfill period")
            return CatchUpResult(
                ready=True,
                days_backfilled=0,
                tickers_updated=0,
                tickers_failed=0,
                rows_added=0,
                rows_dropped=0,
            )
        
        logger.info(f"Fetched {len(df)} rows")
        
        # Process tickers with isolation (Requirement 3.1)
        if 'ticker' in df.columns:
            unique_tickers = df['ticker'].unique()
            
            for ticker in unique_tickers:
                try:
                    ticker_df = df[df['ticker'] == ticker].copy()
                    
                    # Validate with BACKFILL context (Requirement 2.3)
                    report = self.validator.validate(
                        ticker_df,
                        context=ValidationContext.BACKFILL,
                        lazy=True
                    )
                    
                    # Apply actions to get clean data
                    clean_df = self.processor.apply_actions(ticker_df, report)
                    
                    # Write to Parquet
                    self.writer.write_prices(clean_df, mode='append')
                    
                    # Update statistics
                    tickers_updated += 1
                    total_rows_added += len(clean_df)
                    total_rows_dropped += report.rows_dropped
                    
                    # Collect spike classifications
                    all_confirmed_spikes.extend(report.confirmed_spikes)
                    all_persistent_moves.extend(report.persistent_moves)
                    
                except Exception as e:
                    # Ticker isolation: log and continue (Requirement 3.1)
                    tickers_failed += 1
                    error_msg = f"Ticker {ticker} failed: {e}"
                    errors.append(error_msg)
                    logger.warning(error_msg)
        else:
            # No ticker column - process as batch
            try:
                report = self.validator.validate(
                    df,
                    context=ValidationContext.BACKFILL,
                    lazy=True
                )
                
                clean_df = self.processor.apply_actions(df, report)
                self.writer.write_prices(clean_df, mode='append')
                
                tickers_updated = len(tickers)
                total_rows_added = len(clean_df)
                total_rows_dropped = report.rows_dropped
                all_confirmed_spikes = report.confirmed_spikes
                all_persistent_moves = report.persistent_moves
                
            except Exception as e:
                errors.append(f"Batch processing failed: {e}")
                tickers_failed = len(tickers)
        
        # Check drop rate threshold (Requirement 3.2)
        if total_rows_added + total_rows_dropped > 0:
            drop_rate = total_rows_dropped / (total_rows_added + total_rows_dropped)
            if drop_rate > self.drop_rate_threshold:
                logger.warning(
                    f"Drop rate {drop_rate:.1%} exceeds threshold {self.drop_rate_threshold:.1%}"
                )
        
        logger.info(
            f"Backfill complete: {tickers_updated} tickers updated, "
            f"{tickers_failed} failed, {total_rows_added} rows added"
        )
        
        return CatchUpResult(
            ready=tickers_failed == 0 or tickers_updated > 0,
            days_backfilled=0,  # Will be set by caller
            tickers_updated=tickers_updated,
            tickers_failed=tickers_failed,
            rows_added=total_rows_added,
            rows_dropped=total_rows_dropped,
            confirmed_spikes=all_confirmed_spikes,
            persistent_moves=all_persistent_moves,
            errors=errors
        )
    
    def get_gap_status(self) -> dict:
        """
        Get current gap status without performing backfill.
        
        Returns:
            Dict with gap information
        """
        last_date = self._get_max_date()
        today = date.today()
        
        if last_date is None:
            return {
                "has_data": False,
                "last_date": None,
                "gap_days": None,
                "needs_backfill": True,
                "message": "Data lake is empty, initial load required"
            }
        
        gap_days = self._calculate_business_day_gap(last_date, today)
        
        return {
            "has_data": True,
            "last_date": str(last_date),
            "today": str(today),
            "gap_days": gap_days,
            "needs_backfill": gap_days > 0
        }
