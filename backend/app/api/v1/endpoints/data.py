"""
Data API Endpoints

High-performance data access endpoints using the new Parquet/DuckDB data layer.
Provides fast cross-sectional and time-series queries for quantitative analysis.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import date, timedelta
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================
# Response Models
# ============================================================

class PriceData(BaseModel):
    ticker: str
    date: date
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    adj_close: Optional[float]
    volume: Optional[int]


class ReturnsData(BaseModel):
    ticker: str
    returns: dict  # date -> return


class UniverseData(BaseModel):
    tickers: List[str]
    as_of_date: date


class DataLakeStats(BaseModel):
    total_prices: int
    total_securities: int
    date_range: dict
    storage_size_mb: float


# ============================================================
# Data Provider Factory
# ============================================================

def get_data_provider():
    """Get the appropriate data provider."""
    try:
        from quant.data.data_provider import create_data_provider
        # Try Parquet first, fallback to SQLite
        try:
            return create_data_provider(provider_type='parquet')
        except Exception:
            from app.core.database import SessionLocal
            return create_data_provider(provider_type='sqlite', db_session=SessionLocal())
    except Exception as e:
        logger.error(f"Failed to create data provider: {e}")
        raise HTTPException(status_code=500, detail="Data provider unavailable")


# ============================================================
# Endpoints
# ============================================================

@router.get("/prices", response_model=List[PriceData])
async def get_prices(
    tickers: str = Query(..., description="Comma-separated ticker symbols"),
    start_date: date = Query(default=None, description="Start date (YYYY-MM-DD)"),
    end_date: date = Query(default=None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(default=1000, le=10000, description="Max rows to return")
):
    """
    Get historical price data for specified tickers.
    
    Uses high-performance Parquet/DuckDB backend when available.
    """
    ticker_list = [t.strip().upper() for t in tickers.split(',')]
    
    if not ticker_list:
        raise HTTPException(status_code=400, detail="No tickers specified")
    
    # Default date range
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=365)
    
    try:
        provider = get_data_provider()
        df = provider.get_prices(
            tickers=ticker_list,
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            return []
        
        # Limit results
        df = df.head(limit)
        
        # Convert to response format
        results = []
        for _, row in df.iterrows():
            results.append(PriceData(
                ticker=row['ticker'],
                date=row['date'],
                open=row.get('open'),
                high=row.get('high'),
                low=row.get('low'),
                close=row.get('close'),
                adj_close=row.get('adj_close'),
                volume=int(row['volume']) if row.get('volume') else None
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Error fetching prices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/returns")
async def get_returns(
    tickers: str = Query(..., description="Comma-separated ticker symbols"),
    start_date: date = Query(default=None, description="Start date"),
    end_date: date = Query(default=None, description="End date"),
    period: str = Query(default="daily", description="daily, weekly, or monthly")
):
    """
    Get calculated returns for specified tickers.
    
    Period options:
    - daily: Daily returns
    - weekly: Weekly returns (Friday close to Friday close)
    - monthly: Monthly returns
    """
    ticker_list = [t.strip().upper() for t in tickers.split(',')]
    
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=365)
    
    if period not in ['daily', 'weekly', 'monthly']:
        raise HTTPException(status_code=400, detail="Period must be daily, weekly, or monthly")
    
    try:
        provider = get_data_provider()
        df = provider.get_returns(
            tickers=ticker_list,
            start_date=start_date,
            end_date=end_date,
            period=period
        )
        
        if df.empty:
            return {}
        
        # Convert to dict format
        result = {}
        for ticker in df.columns:
            result[ticker] = {
                str(idx): float(val) if not pd.isna(val) else None
                for idx, val in df[ticker].items()
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating returns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/universe", response_model=UniverseData)
async def get_universe(
    as_of_date: date = Query(default=None, description="Universe as of this date")
):
    """
    Get the stock universe available as of a given date.
    
    Useful for Point-in-Time accurate backtesting.
    """
    if as_of_date is None:
        as_of_date = date.today()
    
    try:
        provider = get_data_provider()
        tickers = provider.get_universe(as_of_date)
        
        return UniverseData(
            tickers=sorted(tickers),
            as_of_date=as_of_date
        )
        
    except Exception as e:
        logger.error(f"Error fetching universe: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=DataLakeStats)
async def get_data_lake_stats():
    """
    Get statistics about the data lake.
    """
    try:
        import duckdb
        from pathlib import Path
        import os
        
        base = Path(__file__).parent.parent.parent.parent.parent
        data_lake = base / 'data_lake'
        
        if not data_lake.exists():
            raise HTTPException(status_code=404, detail="Data lake not found")
        
        conn = duckdb.connect(':memory:')
        
        # Price stats
        prices_path = data_lake / 'raw' / 'prices'
        if prices_path.exists():
            price_count = conn.execute(f"""
                SELECT COUNT(*) FROM read_parquet('{prices_path}/**/*.parquet')
            """).fetchone()[0]
            
            date_range = conn.execute(f"""
                SELECT MIN(date), MAX(date) FROM read_parquet('{prices_path}/**/*.parquet')
            """).fetchone()
        else:
            price_count = 0
            date_range = (None, None)
        
        # Securities count
        securities_path = data_lake / 'raw' / 'securities'
        if securities_path.exists():
            securities_count = conn.execute(f"""
                SELECT COUNT(*) FROM read_parquet('{securities_path}/*.parquet')
            """).fetchone()[0]
        else:
            securities_count = 0
        
        # Calculate storage size
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(data_lake):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        
        return DataLakeStats(
            total_prices=price_count,
            total_securities=securities_count,
            date_range={
                'min': str(date_range[0]) if date_range[0] else None,
                'max': str(date_range[1]) if date_range[1] else None
            },
            storage_size_mb=round(total_size / (1024 * 1024), 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refresh")
async def trigger_refresh(
    days_back: int = Query(default=5, le=30, description="Days of data to refresh")
):
    """
    Trigger a data refresh from yfinance.
    
    Updates both SQLite and Parquet data stores.
    """
    try:
        from scripts.daily_update import DailyUpdateService
        
        service = DailyUpdateService(update_sqlite=True, update_parquet=True)
        service.update_prices(days_back=days_back)
        
        return {
            "status": "success",
            "tickers_updated": service.stats['tickers_updated'],
            "rows_added": service.stats['rows_added'],
            "errors": len(service.stats['errors'])
        }
        
    except Exception as e:
        logger.error(f"Refresh failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Import pandas for returns endpoint
import pandas as pd
