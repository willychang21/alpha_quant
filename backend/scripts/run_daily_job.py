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
            
        # 2. Portfolio Optimization (Tier-2: Kelly + Vol Targeting)
        logger.info("Running Portfolio Optimization (Tier-2: Kelly + Vol Targeting)...")
        from quant.portfolio.optimizer import PortfolioOptimizer
        optimizer = PortfolioOptimizer(db)
        
        # Switch to Kelly Optimization with 15% Volatility Target
        allocations = optimizer.run_optimization(
            date.today(), 
            optimizer='kelly', 
            target_vol=0.15
        )
        
        if allocations:
            logger.info("Optimization Complete.")
            
            # 3. Execution Schedule (Tier-3: VWAP)
            logger.info("Generating Execution Schedule (Tier-3: VWAP)...")
            from quant.execution.algo import VWAPExecution
            
            # Generate schedule for the top allocation
            top_ticker, top_weight = allocations[0]
            # Assume $1M portfolio, price $100 (simplified for log)
            target_shares = int((1_000_000 * top_weight) / 100) 
            
            vwap_algo = VWAPExecution()
            schedule = vwap_algo.generate_schedule(total_shares=target_shares)
            logger.info(f"VWAP Schedule for {top_ticker} ({target_shares} shares):\n{schedule.head()}")
            
        else:
            logger.warning("Optimization returned no allocations.")
            
        # 4. Register Signals / Monitoring
        monitor.record_success("daily_job")
        logger.info("Daily Job Completed Successfully (Tier-1 -> Tier-2 -> Tier-3).")
        
    except Exception as e:
        logger.error(f"Daily Job Failed: {e}")
        monitor.record_failure("daily_job", str(e))
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(main())
