from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime, date

from app.core.database import get_db
from quant.data.models import ModelSignals, Security
from quant.signals.schema import SignalCreate, SignalResponse

router = APIRouter()

# TODO: Update create_signal to use ModelSignals and look up SID
# @router.post("/", response_model=SignalResponse)
# def create_signal(signal: SignalCreate, db: Session = Depends(get_db)):
#     """
#     Create a new trading signal.
#     """
#     # Logic to find SID from ticker, then create ModelSignals
#     pass

@router.get("/", response_model=List[SignalResponse])
def get_signals(
    ticker: Optional[str] = None,
    model_name: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Retrieve signals with optional filtering.
    """
    query = db.query(ModelSignals).join(Security)
    
    # If no date filter provided, default to latest available date
    if not start_date and not end_date:
        latest_date = db.query(func.max(ModelSignals.date)).scalar()
        print(f"DEBUG: Latest Date found: {latest_date}")
        if latest_date:
            print(f"DEBUG: Filtering by latest date: {latest_date}")
            query = query.filter(ModelSignals.date == latest_date)
        else:
            print("DEBUG: No latest date found.")
    else:
        print(f"DEBUG: Date filters provided: start={start_date}, end={end_date}")
            
    if ticker:
        query = query.filter(Security.ticker == ticker)
    if model_name:
        query = query.filter(ModelSignals.model_name == model_name)
    if start_date:
        query = query.filter(ModelSignals.date >= start_date)
    if end_date:
        query = query.filter(ModelSignals.date <= end_date)
        
    results = query.order_by(ModelSignals.score.desc()).limit(limit).all()
    print(f"DEBUG: Returned {len(results)} results.")
    
    # Map to response model
    response = []
    for r in results:
        response.append({
            "id": r.id,
            "ticker": r.security.ticker,
            "model_name": r.model_name,
            "model_version": "v1", # Placeholder or add to model
            "score": r.score,
            "metadata_json": r.metadata_json,
            "timestamp": datetime.combine(r.date, datetime.min.time())
        })
        
    return response
