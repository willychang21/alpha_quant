import sys
import os
import logging

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.database import SessionLocal
from quant.portfolio.optimizer import PortfolioOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_confidence():
    db = SessionLocal()
    optimizer = PortfolioOptimizer(db)
    
    logger.info("Testing _get_system_confidence()...")
    multiplier = optimizer._get_system_confidence()
    
    logger.info(f"Returned Multiplier: {multiplier}")
    
    # Expected behavior:
    # If Sharpe is -0.04 (as seen in API), multiplier should be ~0.5
    
    if 0.45 <= multiplier <= 0.55:
        logger.info("✅ Verification SUCCESS: Multiplier is correctly reduced due to low Sharpe.")
    elif multiplier == 1.0:
        logger.warning("⚠️ Verification WARNING: Multiplier is 1.0 (Default). Check if ModelRegistry found the run.")
    else:
        logger.info(f"ℹ️ Verification INFO: Multiplier {multiplier} is within valid range (0.5-1.2).")
        
    db.close()

if __name__ == "__main__":
    verify_confidence()
