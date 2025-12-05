import asyncio
import logging
import sys
import os
from datetime import date
import pandas as pd

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import SessionLocal
from quant.selection.ranking import RankingEngine
from quant.portfolio.optimizer import PortfolioOptimizer
from quant.reporting.trade_list import TradeListGenerator

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_weekly_pipeline():
    logger.info("🚀 Starting Weekly Quant Job (Low-Frequency System)...")
    
    db = SessionLocal()
    today = date.today()
    
    try:
        # 1. Tier-1: Alpha Generation (Ranking)
        # Includes: VSM, BAB, QMJ, Revisions, ValuationComposite
        logger.info("--- Phase 1: Alpha Generation ---")
        ranking_engine = RankingEngine(db)
        await ranking_engine.run_ranking(today)
        
        # 2. Tier-2: Portfolio Construction (Constrained Kelly)
        # Includes: Sector Constraints, Beta Constraints, Vol Targeting
        logger.info("--- Phase 2: Portfolio Construction ---")
        optimizer = PortfolioOptimizer(db)
        optimizer.run_optimization(
            today, 
            optimizer='kelly', 
            target_vol=0.15,
            sector_constraints=True,
            beta_constraints=True
        )
        
        # 3. Tier-3: Reporting (Trade List)
        logger.info("--- Phase 3: Trade List Generation ---")
        generator = TradeListGenerator(db)
        trade_list = generator.generate_trade_list(today, model_name='kelly_v1')
        
        if not trade_list.empty:
            filename = f"weekly_trade_list_{today}.csv"
            path = os.path.join("data", "exports", filename)
            trade_list.to_csv(path, index=False)
            logger.info(f"✅ Trade list saved to {path}")
            print("\n--- Weekly Trade List ---")
            print(trade_list[['Ticker', 'Action', 'Shares', 'Weight', 'Sector']].to_string())
        else:
            logger.warning("⚠️ No trades generated.")
            
    except Exception as e:
        logger.error(f"Weekly Job Failed: {e}")
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(run_weekly_pipeline())
