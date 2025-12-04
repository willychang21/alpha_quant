from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from typing import List, Dict
from quant.data.models import Security, MarketDataDaily
import logging

logger = logging.getLogger(__name__)

def get_or_create_security(db: Session, ticker: str, name: str = None, sector: str = None) -> Security:
    """
    Get a security by ticker, or create it if it doesn't exist.
    """
    security = db.query(Security).filter(Security.ticker == ticker).first()
    if not security:
        security = Security(ticker=ticker, name=name, type="EQUITY", active=True)
        db.add(security)
        db.commit()
        db.refresh(security)
    return security

def bulk_upsert_market_data(db: Session, data: List[Dict]):
    """
    Bulk upsert market data.
    Since SQLite doesn't support ON CONFLICT DO UPDATE fully in older versions via SQLAlchemy core easily without specific dialect handling,
    we'll use a check-and-insert approach or native SQLite upsert if available.
    
    For simplicity and robustness in this MVP:
    1. Check existing records for (sid, date).
    2. Update if exists, Insert if not.
    
    For high performance in production (Postgres), we'd use `postgres_insert.on_conflict_do_update`.
    """
    if not data:
        return

    # Naive approach for MVP SQLite (can be slow for millions of rows, but fine for daily updates)
    # Optimization: Load all existing dates for these SIDs into memory to minimize queries.
    
    sids = {d['sid'] for d in data}
    existing_records = db.query(MarketDataDaily).filter(MarketDataDaily.sid.in_(sids)).all()
    existing_map = {(r.sid, r.date): r for r in existing_records}
    
    new_records = []
    
    for row in data:
        key = (row['sid'], row['date'])
        if key in existing_map:
            # Update existing
            record = existing_map[key]
            record.open = row['open']
            record.high = row['high']
            record.low = row['low']
            record.close = row['close']
            record.adj_close = row['adj_close']
            record.volume = row['volume']
        else:
            # Insert new
            new_records.append(MarketDataDaily(**row))
            
    if new_records:
        db.bulk_save_objects(new_records)
        
    db.commit()
    logger.info(f"Upserted {len(data)} rows ({len(new_records)} new).")

def bulk_upsert_fundamentals(db: Session, data: List[Dict]):
    """
    Bulk upsert fundamental data.
    """
    if not data:
        return

    # Naive check-and-insert for MVP
    # Key is (sid, date, metric)
    from quant.data.models import Fundamentals
    
    sids = {d['sid'] for d in data}
    existing_records = db.query(Fundamentals).filter(Fundamentals.sid.in_(sids)).all()
    existing_map = {(r.sid, r.date, r.metric): r for r in existing_records}
    
    new_records = []
    
    for row in data:
        key = (row['sid'], row['date'], row['metric'])
        if key in existing_map:
            # Update
            record = existing_map[key]
            record.value = row['value']
            record.period = row.get('period')
        else:
            # Insert
            new_records.append(Fundamentals(**row))
            
    if new_records:
        db.bulk_save_objects(new_records)
        
    db.commit()
    logger.info(f"Upserted {len(data)} fundamental records ({len(new_records)} new).")
