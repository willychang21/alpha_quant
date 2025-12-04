import sys
import os
import logging
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from quant.data.models import ModelSignals, PortfolioTargets, Security

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_rankings(db: Session):
    logger.info("--- Debugging Rankings ---")
    latest = db.query(ModelSignals).order_by(ModelSignals.date.desc()).first()
    if not latest:
        logger.error("No latest signal found!")
        return

    logger.info(f"Latest Date: {latest.date}")
    
    signals = db.query(ModelSignals).filter(ModelSignals.date == latest.date, ModelSignals.model_name == 'ranking_v1').order_by(ModelSignals.rank.asc()).all()
    logger.info(f"Found {len(signals)} signals for ranking_v1 on {latest.date}")
    
    for s in signals[:5]:
        try:
            ticker = s.security.ticker
            logger.info(f"Rank {s.rank}: {ticker} (Score: {s.score})")
        except Exception as e:
            logger.error(f"Error accessing security for signal {s.id}: {e}")

def debug_portfolio(db: Session):
    logger.info("\n--- Debugging Portfolio ---")
    latest = db.query(PortfolioTargets).order_by(PortfolioTargets.date.desc()).first()
    if not latest:
        logger.error("No latest portfolio target found!")
        return

    logger.info(f"Latest Date: {latest.date}")
    
    targets = db.query(PortfolioTargets).filter(PortfolioTargets.date == latest.date, PortfolioTargets.model_name == 'mvo_sharpe').all()
    logger.info(f"Found {len(targets)} targets for mvo_sharpe on {latest.date}")
    
    for t in targets[:5]:
        try:
            ticker = t.security.ticker
            logger.info(f"Ticker: {ticker} (Weight: {t.weight})")
        except Exception as e:
            logger.error(f"Error accessing security for target {t.id}: {e}")

if __name__ == "__main__":
    db = SessionLocal()
    try:
        debug_rankings(db)
        debug_portfolio(db)
    finally:
        db.close()
