"""Data Freshness Service.

Checks and reports data freshness status for market data.
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple
import logging

from core.structured_logger import get_structured_logger

logger = get_structured_logger("DataFreshness")


class DataFreshnessService:
    """Check and report data freshness status.
    
    Compares the most recent data date against today to determine
    if data is stale based on a configurable threshold.
    """
    
    threshold_hours: float
    _data_lake_path: Optional[Path]
    
    def __init__(
        self, 
        threshold_hours: float = 24.0,
        data_lake_path: Optional[Path] = None
    ) -> None:
        """Initialize freshness service.
        
        Args:
            threshold_hours: Maximum acceptable data lag in hours
            data_lake_path: Path to data lake directory
        """
        self.threshold_hours = threshold_hours
        self._data_lake_path = data_lake_path
    
    @property
    def data_lake_path(self) -> Path:
        """Get data lake path, loading from settings if not set."""
        if self._data_lake_path is None:
            from quant.data.parquet_io import get_data_lake_path
            self._data_lake_path = get_data_lake_path()
        return self._data_lake_path
    
    def get_freshness_status(self) -> Tuple[bool, float, Optional[date]]:
        """Check data freshness.
        
        Returns:
            Tuple of (is_fresh, lag_hours, last_date)
            - is_fresh: True if data lag <= threshold
            - lag_hours: Hours since last data update
            - last_date: Most recent date in data lake
        """
        last_date = self._get_max_date()
        if last_date is None:
            return False, float('inf'), None
        
        lag_hours = self._calculate_lag_hours(last_date)
        is_fresh = lag_hours <= self.threshold_hours
        
        return is_fresh, lag_hours, last_date
    
    def check_and_log(self) -> bool:
        """Check freshness and log appropriate message.
        
        Returns:
            True if data is fresh, False otherwise
        """
        is_fresh, lag_hours, last_date = self.get_freshness_status()
        
        if last_date is None:
            logger.warning("[FRESHNESS] No data found in data lake")
            return False
        
        if is_fresh:
            logger.info(
                f"[FRESHNESS] Data is fresh (lag: {lag_hours:.1f}h, last: {last_date})"
            )
        else:
            logger.warning(
                f"[FRESHNESS] Data is STALE - lag: {lag_hours:.1f}h exceeds "
                f"threshold {self.threshold_hours}h (last: {last_date})"
            )
        
        return is_fresh
    
    def _get_max_date(self) -> Optional[date]:
        """Get the most recent date from price data.
        
        Returns:
            Most recent date or None if no data found
        """
        try:
            prices_path = self.data_lake_path / 'raw' / 'prices'
            
            if not prices_path.exists():
                return None
            
            # Find most recent file modification
            latest_mtime = None
            for parquet_file in prices_path.rglob('*.parquet'):
                mtime = parquet_file.stat().st_mtime
                if latest_mtime is None or mtime > latest_mtime:
                    latest_mtime = mtime
            
            if latest_mtime is None:
                return None
            
            # Convert to date
            return datetime.fromtimestamp(latest_mtime).date()
            
        except Exception as e:
            logger.warning(f"[FRESHNESS] Error reading data lake: {e}")
            return None
    
    def _calculate_lag_hours(self, last_date: date) -> float:
        """Calculate hours since last data update.
        
        Args:
            last_date: Most recent data date
            
        Returns:
            Hours since last update
        """
        now = datetime.now()
        last_datetime = datetime.combine(last_date, datetime.min.time())
        delta = now - last_datetime
        return delta.total_seconds() / 3600


# Singleton instance
_freshness_service: Optional[DataFreshnessService] = None


def get_data_freshness_service(
    threshold_hours: float = 24.0
) -> DataFreshnessService:
    """Get or create DataFreshnessService singleton."""
    global _freshness_service
    if _freshness_service is None:
        from config.settings import get_settings
        settings = get_settings()
        threshold = getattr(settings, 'data_freshness_threshold_hours', threshold_hours)
        _freshness_service = DataFreshnessService(threshold_hours=threshold)
    return _freshness_service
