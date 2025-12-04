from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from quant.data.models import ModelSignals, PortfolioTargets, Security
from quant.backtest.engine import BacktestEngine
from datetime import date, timedelta
import logging
import sys
from app.core.database import SessionLocal

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/rankings")
def get_rankings():
    """
    Get latest stock rankings.
    """
    db = SessionLocal()
    try:
        # Find latest date
        sys.stderr.write("DEBUG: Entering get_rankings\n")
        sys.stderr.flush()
    
        total_count = db.query(ModelSignals).count()
        print(f"DEBUG: Total ModelSignals in DB: {total_count}")
        
        latest = db.query(ModelSignals).order_by(ModelSignals.date.desc()).first()
        if not latest:
            print("DEBUG: No latest signal found in DB")
            return []
        
        print(f"DEBUG: Latest signal date: {latest.date}")
            
        signals = db.query(ModelSignals).filter(ModelSignals.date == latest.date, ModelSignals.model_name == 'ranking_v1').order_by(ModelSignals.rank.asc()).all()
        print(f"DEBUG: Found {len(signals)} ranking signals")
        
        results = []
        for s in signals:
            try:
                if not s.security:
                    print(f"DEBUG: Signal {s.id} has no security attached")
                    continue
                    
                results.append({
                    "rank": s.rank,
                    "ticker": s.security.ticker,
                    "score": s.score,
                    "date": s.date,
                    "metadata": s.metadata_json
                })
            except Exception as e:
                print(f"DEBUG: Error processing signal {s.id}: {e}")
                
        return results
    finally:
        db.close()

@router.get("/portfolio")
def get_portfolio():
    """
    Get latest portfolio targets.
    """
    db = SessionLocal()
    try:
        sys.stderr.write("DEBUG: Entering get_portfolio\n")
        sys.stderr.flush()
        
        latest = db.query(PortfolioTargets).order_by(PortfolioTargets.date.desc()).first()
        if not latest:
            print("DEBUG: No latest portfolio target found")
            return []
            
        print(f"DEBUG: Latest portfolio date: {latest.date}")
        
        targets = db.query(PortfolioTargets).filter(PortfolioTargets.date == latest.date, PortfolioTargets.model_name == 'mvo_sharpe').all()
        print(f"DEBUG: Found {len(targets)} portfolio targets")
        
        return [
            {
                "ticker": t.security.ticker,
                "weight": t.weight,
                "date": t.date
            }
            for t in targets
        ]
    except Exception as e:
        print(f"DEBUG: Error in get_portfolio: {e}")
        return []
    finally:
        db.close()

@router.post("/backtest")
def run_backtest(db: Session = Depends(get_db)):
    """
    Run a backtest simulation (1 Year).
    """
    try:
        engine = BacktestEngine(db)
        end_date = date.today()
        start_date = end_date - timedelta(days=365)
        
        results = engine.run_backtest(start_date, end_date)
        return results
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
