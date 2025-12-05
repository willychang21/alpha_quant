"""
Data Provider Abstraction Layer

Provides a unified interface for accessing market and fundamental data.
Supports both legacy SQLite and new Parquet/DuckDB backends.

This is the "Two Sigma / AQR" grade data layer that:
1. Abstracts storage from computation
2. Supports Point-in-Time (PIT) queries
3. Enables high-performance batch operations
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union
import pandas as pd
import numpy as np
from datetime import date, datetime
from pathlib import Path
import duckdb
import pyarrow.parquet as pq
import logging

logger = logging.getLogger(__name__)


class DataProvider(ABC):
    """
    Abstract base class for all data providers.
    
    Follows the Repository pattern to decouple data access from business logic.
    """
    
    @abstractmethod
    def get_prices(
        self, 
        tickers: List[str], 
        start_date: date, 
        end_date: date,
        fields: List[str] = None,
        as_of_date: Optional[date] = None  # Point-in-Time
    ) -> pd.DataFrame:
        """
        Retrieve price data for given tickers and date range.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            fields: Columns to return, defaults to ['open', 'high', 'low', 'close', 'adj_close', 'volume']
            as_of_date: For PIT queries, only return data known as of this date
            
        Returns:
            DataFrame with MultiIndex (ticker, date) or (date,) with ticker columns
        """
        pass
    
    @abstractmethod
    def get_returns(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date,
        period: str = 'daily',  # 'daily', 'weekly', 'monthly'
        as_of_date: Optional[date] = None
    ) -> pd.DataFrame:
        """Calculate returns for given tickers."""
        pass
    
    @abstractmethod
    def get_fundamentals(
        self,
        tickers: List[str],
        metrics: List[str] = None,
        as_of_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Get fundamental data in wide format.
        
        Args:
            tickers: List of ticker symbols
            metrics: Specific metrics to retrieve (None = all)
            as_of_date: Point-in-time date for look-ahead bias prevention
            
        Returns:
            Wide-format DataFrame with one row per ticker
        """
        pass
    
    @abstractmethod
    def get_universe(self, as_of_date: date) -> List[str]:
        """Get available ticker universe as of given date."""
        pass
    
    @abstractmethod
    def get_price(self, ticker: str, date: date) -> Optional[float]:
        """Get single price (adjusted close) for backtesting."""
        pass


class ParquetDataProvider(DataProvider):
    """
    High-performance Parquet/DuckDB-based data provider.
    
    Designed for:
    - Fast cross-sectional queries (all tickers on one date)
    - Efficient time-series queries (one ticker across dates)
    - Point-in-Time accurate historical analysis
    """
    
    def __init__(self, data_lake_path: str):
        self.data_lake = Path(data_lake_path)
        self.raw_path = self.data_lake / 'raw'
        self.processed_path = self.data_lake / 'processed'
        self.snapshots_path = self.data_lake / 'snapshots'
        
        # DuckDB for fast SQL queries on Parquet
        self.conn = duckdb.connect(':memory:')
        
        # Price cache for backtest performance
        self._price_cache: Optional[pd.DataFrame] = None
        self._cache_range: Optional[tuple] = None
        
        # Initialize paths
        self._ensure_paths()
        
    def _ensure_paths(self):
        """Create directory structure if it doesn't exist."""
        for path in [
            self.raw_path / 'prices',
            self.raw_path / 'fundamentals',
            self.processed_path / 'factors',
            self.processed_path / 'signals',
            self.snapshots_path
        ]:
            path.mkdir(parents=True, exist_ok=True)
    
    def get_prices(
        self, 
        tickers: List[str], 
        start_date: date, 
        end_date: date,
        fields: List[str] = None,
        as_of_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Retrieve price data using DuckDB for fast queries.
        """
        if fields is None:
            fields = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
        
        prices_path = self.raw_path / 'prices'
        
        if not any(prices_path.glob('**/*.parquet')):
            logger.warning("No parquet files found in prices directory")
            return pd.DataFrame()
        
        # Build field list for SQL
        field_list = ', '.join(['ticker', 'date'] + fields)
        ticker_list = ', '.join([f"'{t}'" for t in tickers])
        
        # Use DuckDB to query Parquet files directly
        query = f"""
            SELECT {field_list}
            FROM read_parquet('{prices_path}/**/*.parquet')
            WHERE ticker IN ({ticker_list})
              AND date >= '{start_date}'
              AND date <= '{end_date}'
        """
        
        if as_of_date:
            # PIT: Only include data that was known as of as_of_date
            query += f" AND date <= '{as_of_date}'"
        
        query += " ORDER BY ticker, date"
        
        try:
            df = self.conn.execute(query).df()
            df['date'] = pd.to_datetime(df['date']).dt.date
            return df
        except Exception as e:
            logger.error(f"Error querying prices: {e}")
            return pd.DataFrame()
    
    def get_returns(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date,
        period: str = 'daily',
        as_of_date: Optional[date] = None
    ) -> pd.DataFrame:
        """Calculate returns efficiently using DuckDB window functions."""
        
        prices_path = self.raw_path / 'prices'
        ticker_list = ', '.join([f"'{t}'" for t in tickers])
        
        # Calculate returns using SQL window function
        query = f"""
            SELECT 
                ticker,
                date,
                adj_close,
                (adj_close / LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date)) - 1 AS ret
            FROM read_parquet('{prices_path}/**/*.parquet')
            WHERE ticker IN ({ticker_list})
              AND date >= '{start_date}'
              AND date <= '{end_date}'
            ORDER BY ticker, date
        """
        
        try:
            df = self.conn.execute(query).df()
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Pivot if needed
            if period == 'daily':
                return df.pivot(index='date', columns='ticker', values='ret')
            else:
                # Resample for weekly/monthly
                df_pivoted = df.pivot(index='date', columns='ticker', values='adj_close')
                df_pivoted.index = pd.to_datetime(df_pivoted.index)
                freq = 'W' if period == 'weekly' else 'ME'
                resampled = df_pivoted.resample(freq).last()
                return resampled.pct_change()
                
        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return pd.DataFrame()
    
    def get_fundamentals(
        self,
        tickers: List[str],
        metrics: List[str] = None,
        as_of_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Get fundamental data in wide format.
        Uses latest available data as of as_of_date.
        """
        fundamentals_path = self.raw_path / 'fundamentals'
        
        if not any(fundamentals_path.glob('**/*.parquet')):
            logger.warning("No fundamental data found")
            return pd.DataFrame()
        
        ticker_list = ', '.join([f"'{t}'" for t in tickers])
        
        # Get latest record per ticker up to as_of_date
        date_filter = f"AND date <= '{as_of_date}'" if as_of_date else ""
        
        query = f"""
            WITH ranked AS (
                SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) as rn
                FROM read_parquet('{fundamentals_path}/**/*.parquet')
                WHERE ticker IN ({ticker_list})
                {date_filter}
            )
            SELECT * FROM ranked WHERE rn = 1
        """
        
        try:
            return self.conn.execute(query).df()
        except Exception as e:
            logger.error(f"Error querying fundamentals: {e}")
            return pd.DataFrame()
    
    def get_universe(self, as_of_date: date) -> List[str]:
        """Get unique tickers available as of given date."""
        prices_path = self.raw_path / 'prices'
        
        query = f"""
            SELECT DISTINCT ticker
            FROM read_parquet('{prices_path}/**/*.parquet')
            WHERE date <= '{as_of_date}'
        """
        
        try:
            result = self.conn.execute(query).df()
            return result['ticker'].tolist()
        except Exception as e:
            logger.error(f"Error getting universe: {e}")
            return []
    
    def get_price(self, ticker: str, target_date: date) -> Optional[float]:
        """
        Get single price for backtesting.
        Uses caching for performance during sequential backtest.
        """
        # Simple cache: if we haven't cached or date is outside range, refresh
        if self._price_cache is None or not self._is_date_in_cache(target_date):
            self._refresh_price_cache(target_date)
        
        if self._price_cache is None:
            return None
        
        try:
            # Get the most recent price on or before target_date
            mask = (self._price_cache['ticker'] == ticker) & \
                   (self._price_cache['date'] <= target_date)
            filtered = self._price_cache[mask]
            
            if filtered.empty:
                return None
            
            return filtered.iloc[-1]['adj_close']
        except Exception:
            return None
    
    def _is_date_in_cache(self, target_date: date) -> bool:
        if self._cache_range is None:
            return False
        return self._cache_range[0] <= target_date <= self._cache_range[1]
    
    def _refresh_price_cache(self, target_date: date):
        """Load price data around target_date into cache."""
        from datetime import timedelta
        
        # Cache 6 months of data for typical backtest performance
        start = target_date - timedelta(days=180)
        end = target_date + timedelta(days=180)
        
        prices_path = self.raw_path / 'prices'
        
        if not any(prices_path.glob('**/*.parquet')):
            self._price_cache = None
            return
            
        query = f"""
            SELECT ticker, date, adj_close
            FROM read_parquet('{prices_path}/**/*.parquet')
            WHERE date >= '{start}' AND date <= '{end}'
            ORDER BY ticker, date
        """
        
        try:
            self._price_cache = self.conn.execute(query).df()
            self._price_cache['date'] = pd.to_datetime(self._price_cache['date']).dt.date
            self._cache_range = (start, end)
        except Exception as e:
            logger.error(f"Error refreshing price cache: {e}")
            self._price_cache = None


class SQLiteDataProvider(DataProvider):
    """
    Legacy SQLite data provider for backwards compatibility.
    Wraps existing SQLAlchemy models.
    """
    
    def __init__(self, db_session):
        from sqlalchemy.orm import Session
        self.db: Session = db_session
        self._price_cache: Dict[tuple, float] = {}
        
    def get_prices(
        self, 
        tickers: List[str], 
        start_date: date, 
        end_date: date,
        fields: List[str] = None,
        as_of_date: Optional[date] = None
    ) -> pd.DataFrame:
        from quant.data.models import Security, MarketDataDaily
        
        if fields is None:
            fields = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
        
        # Get security IDs
        securities = self.db.query(Security).filter(Security.ticker.in_(tickers)).all()
        sid_map = {s.sid: s.ticker for s in securities}
        
        # Query market data
        query = self.db.query(MarketDataDaily).filter(
            MarketDataDaily.sid.in_(sid_map.keys()),
            MarketDataDaily.date >= start_date,
            MarketDataDaily.date <= end_date
        )
        
        if as_of_date:
            query = query.filter(MarketDataDaily.date <= as_of_date)
        
        records = query.order_by(MarketDataDaily.sid, MarketDataDaily.date).all()
        
        data = []
        for r in records:
            row = {
                'ticker': sid_map[r.sid],
                'date': r.date
            }
            for f in fields:
                row[f] = getattr(r, f, None)
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_returns(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date,
        period: str = 'daily',
        as_of_date: Optional[date] = None
    ) -> pd.DataFrame:
        # Fetch more history for return calculation
        from datetime import timedelta
        extended_start = start_date - timedelta(days=30)
        
        prices = self.get_prices(
            tickers, extended_start, end_date, 
            fields=['adj_close'], 
            as_of_date=as_of_date
        )
        
        if prices.empty:
            return pd.DataFrame()
        
        pivoted = prices.pivot(index='date', columns='ticker', values='adj_close')
        returns = pivoted.pct_change()
        
        # Filter to requested date range
        returns = returns[returns.index >= start_date]
        
        return returns
    
    def get_fundamentals(
        self,
        tickers: List[str],
        metrics: List[str] = None,
        as_of_date: Optional[date] = None
    ) -> pd.DataFrame:
        from quant.data.models import Security, Fundamentals
        
        securities = self.db.query(Security).filter(Security.ticker.in_(tickers)).all()
        sid_map = {s.sid: s.ticker for s in securities}
        
        # Build query
        query = self.db.query(Fundamentals).filter(Fundamentals.sid.in_(sid_map.keys()))
        
        if as_of_date:
            query = query.filter(Fundamentals.date <= as_of_date)
        
        if metrics:
            query = query.filter(Fundamentals.metric.in_(metrics))
        
        records = query.all()
        
        # Convert EAV to wide format
        data = {}
        for r in records:
            ticker = sid_map[r.sid]
            if ticker not in data:
                data[ticker] = {'ticker': ticker}
            data[ticker][r.metric] = r.value
        
        return pd.DataFrame(list(data.values()))
    
    def get_universe(self, as_of_date: date) -> List[str]:
        from quant.data.models import Security, MarketDataDaily
        
        # Get tickers with data as of date
        result = self.db.query(Security.ticker).join(MarketDataDaily).filter(
            MarketDataDaily.date <= as_of_date
        ).distinct().all()
        
        return [r[0] for r in result]
    
    def get_price(self, ticker: str, target_date: date) -> Optional[float]:
        """Get single price with caching."""
        cache_key = (ticker, target_date)
        
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]
        
        from quant.data.models import Security, MarketDataDaily
        
        sec = self.db.query(Security).filter(Security.ticker == ticker).first()
        if not sec:
            return None
        
        record = self.db.query(MarketDataDaily)\
            .filter(MarketDataDaily.sid == sec.sid, MarketDataDaily.date <= target_date)\
            .order_by(MarketDataDaily.date.desc())\
            .first()
        
        price = record.adj_close if record else None
        self._price_cache[cache_key] = price
        
        return price


# Factory function for easy instantiation
def create_data_provider(
    provider_type: str = 'auto',
    data_lake_path: str = None,
    db_session = None
) -> DataProvider:
    """
    Factory function to create appropriate data provider.
    
    Args:
        provider_type: 'parquet', 'sqlite', or 'auto'
        data_lake_path: Path to data lake for Parquet provider
        db_session: SQLAlchemy session for SQLite provider
        
    Returns:
        Configured DataProvider instance
    """
    if provider_type == 'parquet' or (provider_type == 'auto' and data_lake_path):
        if data_lake_path is None:
            import os
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_lake_path = os.path.join(base, 'data_lake')
        return ParquetDataProvider(data_lake_path)
    
    elif provider_type == 'sqlite' or (provider_type == 'auto' and db_session):
        if db_session is None:
            from app.core.database import SessionLocal
            db_session = SessionLocal()
        return SQLiteDataProvider(db_session)
    
    else:
        raise ValueError(f"Cannot determine provider type. Provide data_lake_path or db_session.")
