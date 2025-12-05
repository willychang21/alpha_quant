"""
Signals API Endpoints

Trading signal retrieval and filtering.
Now powered by Parquet/DuckDB for high-performance queries.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime, date
from pydantic import BaseModel

from quant.data.signal_store import get_signal_store
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class SignalResponse(BaseModel):
    """Signal response model."""
    id: Optional[int] = None
    ticker: str
    model_name: str
    model_version: str = "v1"
    score: Optional[float] = None
    rank: Optional[int] = None
    metadata_json: Optional[str] = None
    timestamp: Optional[datetime] = None


@router.get("/", response_model=List[SignalResponse])
def get_signals(
    ticker: Optional[str] = Query(default=None, description="Filter by ticker"),
    model_name: Optional[str] = Query(default=None, description="Filter by model"),
    start_date: Optional[date] = Query(default=None, description="Start date"),
    end_date: Optional[date] = Query(default=None, description="End date"),
    limit: int = Query(default=100, le=500, description="Max results")
):
    """
    Retrieve signals with optional filtering.
    
    Now uses Parquet/DuckDB backend for 35x faster queries.
    """
    try:
        store = get_signal_store()
        
        df = store.get_signals(
            ticker=ticker,
            model_name=model_name,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        if df.empty:
            logger.info("No signals found matching criteria")
            return []
        
        logger.info(f"Found {len(df)} signals")
        
        # Convert to response model
        response = []
        for i, row in df.iterrows():
            signal_date = row.get('date')
            if hasattr(signal_date, 'isoformat'):
                timestamp = datetime.combine(signal_date, datetime.min.time())
            else:
                timestamp = None
            
            response.append(SignalResponse(
                id=i,
                ticker=row['ticker'],
                model_name=row.get('model_name', 'unknown'),
                model_version="v1",
                score=float(row['score']) if row.get('score') is not None else None,
                rank=int(row['rank']) if row.get('rank') is not None else None,
                metadata_json=row.get('metadata_json'),
                timestamp=timestamp
            ))
        
        return response
        
    except Exception as e:
        logger.error(f"Error fetching signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest")
def get_latest_signals(
    model_name: str = Query(default="ranking_v3", description="Model name"),
    limit: int = Query(default=50, le=200, description="Max results")
):
    """
    Get the most recent signals for a model.
    """
    try:
        store = get_signal_store()
        df = store.get_latest_signals(model_name=model_name, limit=limit)
        
        if df.empty:
            return {"signals": [], "date": None, "model": model_name}
        
        return {
            "signals": df.to_dict('records'),
            "date": str(df['date'].iloc[0]) if 'date' in df.columns else None,
            "model": model_name,
            "count": len(df)
        }
        
    except Exception as e:
        logger.error(f"Error fetching latest signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dates")
def get_available_dates():
    """
    Get list of dates with available signals.
    """
    try:
        store = get_signal_store()
        dates = store.list_available_dates('signals')
        
        return {
            "dates": [str(d) for d in dates],
            "count": len(dates),
            "latest": str(dates[0]) if dates else None
        }
        
    except Exception as e:
        logger.error(f"Error fetching dates: {e}")
        raise HTTPException(status_code=500, detail=str(e))
