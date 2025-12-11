"""
DCA Scheduler / Worker

This script runs the background maintenance tasks that were previously part of the
API server startup:
1. Seed securities (S&P 500, Nasdaq 100)
2. Smart Catch-up (Data Lake backfill)
3. Signal Generation (Rankings, Targets)

Run this separately from the API server to perform data ingestion and analysis.
"""
import asyncio
import logging
from app.core.logging_config import setup_logging
from app.core.startup import startup_service

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

from app.core.database import init_db

async def main():
    logger.info("🔧 Starting DCA Scheduler/Worker...")
    
    try:
        # Ensure database tables exist
        init_db()
        # Run the full FAANG-style startup pipeline
        await startup_service.run_startup_tasks()
        
    except Exception as e:
        logger.error(f"❌ Scheduler failed: {e}")
        raise
    
    logger.info("🏁 Scheduler finished successfully.")

if __name__ == "__main__":
    asyncio.run(main())
