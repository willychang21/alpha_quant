import asyncio
import logging
import sys
import os
from datetime import date, datetime

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.core.database import SessionLocal
from quant.selection.ranking import RankingEngine
from quant.model_registry.registry import ModelRegistry
from core.monitoring import MonitoringService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting Daily Job...")
    db = SessionLocal()
    monitor = MonitoringService(db)
    
    try:
        # 1. Ranking & Signal Generation
        logger.info("Running Ranking Engine...")
        engine = RankingEngine(db)
        top_picks = await engine.run_ranking(date.today())
        
        if top_picks is not None:
            logger.info(f"Top Picks:\n{top_picks[['ticker', 'score', 'rank']]}")
            
        # 2. Portfolio Optimization
        logger.info("Running Portfolio Optimization (MVO)...")
        from quant.portfolio.optimizer import PortfolioOptimizer
        optimizer = PortfolioOptimizer(db)
        allocations = optimizer.run_optimization(date.today())
        
        if allocations:
            logger.info("Optimization Complete.")
        else:
            logger.warning("Optimization returned no allocations.")
            
        # 3. Register Signals (Optional, if we want to push to the new Signals table as well)
        # The RankingEngine currently writes to ModelSignals (legacy table?).
        # We should probably migrate that to the new Signal table eventually.
        # For now, let's log success.
        
        monitor.record_success("daily_job")
        logger.info("Daily Job Completed Successfully.")
        
    except Exception as e:
        logger.error(f"Daily Job Failed: {e}")
        monitor.record_failure("daily_job", str(e))
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(main())
