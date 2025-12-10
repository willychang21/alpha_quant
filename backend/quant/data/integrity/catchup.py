"""SmartCatchUpService for automatic gap detection and backfill.

This service handles automatic data gap detection and backfill on system startup.
Use case: Development server not running 24/7, needs to catch up on missed 
trading days before becoming operational.

Requirements: 4.3, 4.4, 6.1
"""

import logging
from datetime import date, timedelta
from typing import List, Optional, Protocol, Tuple

import pandas as pd
import pandas_market_calendars as mcal

from quant.data.integrity.enums import ValidationContext
from quant.data.integrity.exceptions import CatchUpError
from quant.data.integrity.models import ValidationReport
from quant.data.integrity.processor import ActionProcessor
from quant.data.integrity.validator import DataValidator

logger = logging.getLogger(__name__)


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
    
    Requirements: 4.3, 4.4, 6.1
    """
    
    def __init__(
        self,
        data_provider: DataProviderProtocol,
        data_fetcher: DataFetcherProtocol,
        parquet_reader: ParquetReaderProtocol,
        parquet_writer: ParquetWriterProtocol,
        validator: DataValidator,
        processor: Optional[ActionProcessor] = None,
        exchange: str = "NYSE"
    ):
        """Initialize the SmartCatchUpService.
        
        Args:
            data_provider: Provider for getting ticker universe
            data_fetcher: Fetcher for downloading historical data (e.g., yfinance)
            parquet_reader: Reader for checking existing data
            parquet_writer: Writer for persisting backfilled data
            validator: DataValidator for validating fetched data
            processor: ActionProcessor for applying validation actions
            exchange: Market calendar to use (default: NYSE)
        """
        self.data_provider = data_provider
        self.data_fetcher = data_fetcher
        self.reader = parquet_reader
        self.writer = parquet_writer
        self.validator = validator
        self.processor = processor or ActionProcessor()
        self.calendar = mcal.get_calendar(exchange)
    
    def check_and_backfill(
        self, 
        tickers: Optional[List[str]] = None,
        max_gap_days: int = 30
    ) -> Tuple[bool, int, Optional[ValidationReport]]:
        """
        Check for data gaps and backfill if needed.
        
        Args:
            tickers: Optional list of tickers to check. If None, uses full universe.
            max_gap_days: Maximum gap size to attempt backfill (safety limit)
        
        Returns:
            Tuple of (ready: bool, days_backfilled: int, report: Optional[ValidationReport])
        
        Raises:
            CatchUpError: If backfill fails
        """
        try:
            # Get last date in data lake
            last_date = self._get_max_date()
            today = date.today()
            
            if last_date is None:
                logger.warning("No existing data found in data lake")
                return True, 0, None
            
            # Calculate business day gap
            gap_days = self._calculate_business_day_gap(last_date, today)
            
            logger.info(f"Data lake last date: {last_date}, gap: {gap_days} business days")
            
            if gap_days <= 0:
                logger.info("No gap detected, data is up to date")
                return True, 0, None
            
            if gap_days > max_gap_days:
                logger.warning(
                    f"Gap of {gap_days} days exceeds max_gap_days ({max_gap_days}). "
                    "Manual intervention may be required."
                )
                # Still attempt backfill but cap at max_gap_days
                backfill_start = today - timedelta(days=max_gap_days * 2)  # Rough estimate
            else:
                backfill_start = last_date + timedelta(days=1)
            
            # Perform backfill
            report = self._perform_backfill(
                start=backfill_start,
                end=today,
                tickers=tickers
            )
            
            return True, gap_days, report
            
        except Exception as e:
            logger.error(f"Catch-up failed: {e}")
            raise CatchUpError(
                f"Failed to check and backfill data: {e}",
                details={"error": str(e)}
            )
    
    def _get_max_date(self) -> Optional[date]:
        """
        Get the most recent date in the data lake.
        
        Returns:
            Most recent date, or None if no data exists
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
    
    def _calculate_business_day_gap(self, start: date, end: date) -> int:
        """
        Calculate number of missing business days between dates.
        
        Uses NYSE calendar to correctly exclude weekends and holidays.
        
        Args:
            start: Start date (last date in data lake)
            end: End date (usually today)
        
        Returns:
            Number of missing business days (0 if no gap)
        """
        if start >= end:
            return 0
        
        # Get NYSE trading days in the range
        schedule = self.calendar.schedule(start_date=start, end_date=end)
        business_days = len(schedule)
        
        # Subtract 1 to exclude the start date (which should already be in data)
        return max(0, business_days - 1)
    
    def _perform_backfill(
        self, 
        start: date, 
        end: date,
        tickers: Optional[List[str]] = None
    ) -> ValidationReport:
        """
        Fetch and validate historical data for the gap period.
        
        Uses BACKFILL context for validation, enabling spike reversion verification.
        
        Args:
            start: Start date of backfill period
            end: End date of backfill period
            tickers: Optional list of tickers (uses universe if None)
        
        Returns:
            ValidationReport with backfill results
        
        Raises:
            CatchUpError: If backfill fails
        """
        # Get ticker universe if not provided
        if tickers is None:
            try:
                tickers = self.data_provider.get_universe(as_of_date=end)
                logger.info(f"Using universe of {len(tickers)} tickers")
            except Exception as e:
                raise CatchUpError(
                    f"Failed to get ticker universe: {e}",
                    details={"error": str(e)}
                )
        
        if not tickers:
            raise CatchUpError(
                "No tickers available for backfill",
                details={"tickers": []}
            )
        
        logger.info(f"Backfilling {len(tickers)} tickers from {start} to {end}")
        
        # Fetch historical data
        try:
            df = self.data_fetcher.fetch_range(
                tickers=tickers,
                start=start,
                end=end
            )
        except Exception as e:
            raise CatchUpError(
                f"Failed to fetch historical data: {e}",
                details={"start": str(start), "end": str(end), "error": str(e)}
            )
        
        if df is None or df.empty:
            logger.warning("No data fetched for backfill period")
            # Return empty report
            from datetime import datetime
            return ValidationReport(
                timestamp=datetime.now(),
                context=ValidationContext.BACKFILL,
                ticker=None,
                rows_input=0,
                rows_output=0,
                structural_issues=[],
                logical_issues=[],
                temporal_issues=[],
                statistical_issues=[],
                rows_dropped=0,
                rows_warned=0,
                rows_interpolated=0,
                rows_ffilled=0,
            )
        
        logger.info(f"Fetched {len(df)} rows")
        
        # Validate with BACKFILL context (enables spike reversion verification)
        report = self.validator.validate(
            df,
            context=ValidationContext.BACKFILL,
            lazy=True
        )
        
        logger.info(
            f"Validation complete: {report.rows_input} input, "
            f"{report.rows_dropped} dropped, {len(report.confirmed_spikes)} confirmed spikes"
        )
        
        # Apply actions to get clean data
        clean_df = self.processor.apply_actions(df, report)
        
        logger.info(f"Writing {len(clean_df)} clean rows to data lake")
        
        # Write to Parquet
        try:
            self.writer.write_prices(clean_df, mode='append')
        except Exception as e:
            raise CatchUpError(
                f"Failed to write backfilled data: {e}",
                details={"rows": len(clean_df), "error": str(e)}
            )
        
        return report
    
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
                "needs_backfill": False
            }
        
        gap_days = self._calculate_business_day_gap(last_date, today)
        
        return {
            "has_data": True,
            "last_date": str(last_date),
            "today": str(today),
            "gap_days": gap_days,
            "needs_backfill": gap_days > 0
        }
