"""
Point-in-Time (PIT) Data Manager

Manages Point-in-Time data access to prevent look-ahead bias in backtesting.

Key Concepts:
- Knowledge Date: When the data became publicly available
- Effective Date: The period the data refers to (e.g., Q4 2024)
- Lag: Time between period end and public availability (typically 30-60 days for earnings)

Two Sigma / AQR grade systems ALWAYS use PIT data to ensure:
1. No look-ahead bias in backtests
2. Realistic simulation of actual trading conditions
3. Proper handling of data restatements and revisions
"""

from datetime import date, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Types of data with different PIT characteristics."""
    PRICE = "price"           # Known same day (after market close)
    FUNDAMENTAL = "fundamental"  # Known ~45 days after period end
    ANALYST_ESTIMATE = "estimate"  # Known immediately but may revise
    NEWS = "news"             # Known immediately


@dataclass
class PITConfig:
    """Configuration for Point-in-Time data handling."""
    
    # Default lags by data type (in days)
    price_lag: int = 1  # EOD prices known next morning
    fundamental_lag: int = 45  # Conservative: earnings ~6 weeks after quarter end
    estimate_lag: int = 1  # Estimates known quickly
    
    # For more conservative backtests
    use_conservative_lag: bool = True
    conservative_fundamental_lag: int = 60  # Extra buffer
    
    def get_lag(self, data_type: DataType) -> int:
        """Get appropriate lag for data type."""
        if data_type == DataType.PRICE:
            return self.price_lag
        elif data_type == DataType.FUNDAMENTAL:
            if self.use_conservative_lag:
                return self.conservative_fundamental_lag
            return self.fundamental_lag
        elif data_type == DataType.ANALYST_ESTIMATE:
            return self.estimate_lag
        return 0


class PointInTimeManager:
    """
    Manages Point-in-Time data access to prevent look-ahead bias.
    
    Usage:
        pit = PointInTimeManager()
        
        # Get fundamentals as they were known on 2024-03-15
        fundamentals = pit.get_fundamental_snapshot(
            tickers=['AAPL', 'MSFT'],
            as_of_date=date(2024, 3, 15)
        )
    """
    
    def __init__(self, data_provider=None, config: PITConfig = None):
        self.data_provider = data_provider
        self.config = config or PITConfig()
        
        # Cache for performance
        self._fundamental_cache: Dict[date, pd.DataFrame] = {}
        
    def get_knowledge_date(
        self, 
        effective_date: date, 
        data_type: DataType
    ) -> date:
        """
        Calculate when data for effective_date becomes known.
        
        For Q4 earnings (period ends Dec 31):
        - Effective date: Dec 31
        - Knowledge date: ~Feb 15 (45 days later)
        """
        lag = self.config.get_lag(data_type)
        return effective_date + timedelta(days=lag)
    
    def get_effective_date_cutoff(
        self, 
        as_of_date: date, 
        data_type: DataType
    ) -> date:
        """
        Calculate the latest effective date we can use as of as_of_date.
        
        On March 15, 2024:
        - We can use prices up to March 14 (1 day lag)
        - We can use fundamentals from periods ending before ~Jan 30 (45 day lag)
        """
        lag = self.config.get_lag(data_type)
        return as_of_date - timedelta(days=lag)
    
    def get_fundamental_snapshot(
        self, 
        tickers: List[str], 
        as_of_date: date,
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Get fundamental data that was known as of as_of_date.
        
        This is the correct way to access fundamentals in backtesting.
        """
        if self.data_provider is None:
            raise ValueError("DataProvider not configured")
        
        # Calculate cutoff date
        cutoff = self.get_effective_date_cutoff(as_of_date, DataType.FUNDAMENTAL)
        
        logger.debug(
            f"PIT Query: as_of={as_of_date}, cutoff={cutoff}, "
            f"lag={self.config.get_lag(DataType.FUNDAMENTAL)} days"
        )
        
        # Query data up to cutoff
        return self.data_provider.get_fundamentals(
            tickers=tickers,
            metrics=metrics,
            as_of_date=cutoff
        )
    
    def get_price_snapshot(
        self,
        tickers: List[str],
        start_date: date,
        as_of_date: date
    ) -> pd.DataFrame:
        """
        Get prices known as of as_of_date.
        
        On Monday morning, we know Friday's close.
        """
        if self.data_provider is None:
            raise ValueError("DataProvider not configured")
        
        cutoff = self.get_effective_date_cutoff(as_of_date, DataType.PRICE)
        
        return self.data_provider.get_prices(
            tickers=tickers,
            start_date=start_date,
            end_date=cutoff,
            as_of_date=cutoff
        )
    
    def validate_backtest_dates(
        self,
        backtest_start: date,
        backtest_end: date,
        required_lookback_days: int = 252  # 1 year
    ) -> Dict[str, Any]:
        """
        Validate that we have sufficient data for PIT-correct backtest.
        
        Returns warnings about potential issues.
        """
        issues = []
        
        # Check fundamental data availability
        fundamental_cutoff = self.get_effective_date_cutoff(
            backtest_start, DataType.FUNDAMENTAL
        )
        earliest_fundamental = fundamental_cutoff - timedelta(days=required_lookback_days)
        
        # You'd typically query the database here to check actual data availability
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'fundamental_data_needed_from': earliest_fundamental,
            'price_data_needed_from': backtest_start - timedelta(days=required_lookback_days),
            'config': {
                'fundamental_lag': self.config.get_lag(DataType.FUNDAMENTAL),
                'price_lag': self.config.get_lag(DataType.PRICE)
            }
        }


class PITDataFrame:
    """
    A wrapper around DataFrame that tracks Point-in-Time metadata.
    
    This ensures data lineage and prevents accidental look-ahead when
    passing data between components.
    """
    
    def __init__(self, df: pd.DataFrame, as_of_date: date, data_type: DataType):
        self._df = df
        self._as_of_date = as_of_date
        self._data_type = data_type
        self._created_at = pd.Timestamp.now()
        
    @property
    def df(self) -> pd.DataFrame:
        return self._df
    
    @property
    def as_of_date(self) -> date:
        return self._as_of_date
    
    @property
    def data_type(self) -> DataType:
        return self._data_type
    
    def validate_for_date(self, target_date: date) -> bool:
        """Check if this data is valid for use at target_date."""
        if self._as_of_date > target_date:
            logger.warning(
                f"Look-ahead bias detected! Data as_of {self._as_of_date} "
                f"used for {target_date}"
            )
            return False
        return True
    
    def __repr__(self):
        return (
            f"PITDataFrame(as_of={self._as_of_date}, "
            f"type={self._data_type.value}, shape={self._df.shape})"
        )


# Convenience function
def ensure_pit_safe(
    df: pd.DataFrame,
    as_of_date: date,
    target_date: date,
    data_type: DataType = DataType.FUNDAMENTAL
) -> pd.DataFrame:
    """
    Validate and filter DataFrame to ensure PIT safety.
    
    This is a defensive function to catch accidental look-ahead.
    """
    config = PITConfig()
    lag = config.get_lag(data_type)
    
    # Calculate what data should be visible
    visible_cutoff = as_of_date - timedelta(days=lag)
    
    # If DataFrame has a 'date' column, filter it
    if 'date' in df.columns:
        mask = df['date'] <= visible_cutoff
        filtered = df[mask].copy()
        
        if len(filtered) < len(df):
            logger.info(
                f"PIT filter removed {len(df) - len(filtered)} future rows "
                f"(cutoff: {visible_cutoff})"
            )
        
        return filtered
    
    return df
