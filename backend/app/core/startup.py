"""Application startup hooks.

Handles initialization tasks that should run when the server starts,
including the Smart Catch-Up Service for automatic data backfill.
"""

import logging
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)


class StartupService:
    """
    Manages application startup tasks.
    
    Primary responsibility: Run SmartCatchUpService to ensure
    data lake is up-to-date before accepting requests.
    """
    
    def __init__(self, enable_catchup: bool = True, max_gap_days: int = 30):
        """
        Initialize startup service.
        
        Args:
            enable_catchup: Whether to run catch-up on startup
            max_gap_days: Maximum gap size to attempt backfill
        """
        self.enable_catchup = enable_catchup
        self.max_gap_days = max_gap_days
        self._catchup_result = None
    
    async def run_startup_tasks(self):
        """
        Run all startup tasks.
        
        Called by FastAPI lifespan context manager.
        """
        logger.info("=" * 50)
        logger.info("Running startup tasks...")
        logger.info("=" * 50)
        
        if self.enable_catchup:
            await self._run_catchup()
        
        logger.info("Startup tasks complete")
    
    async def _run_catchup(self):
        """
        Run Smart Catch-Up Service to backfill missing data.
        
        This ensures the data lake is up-to-date before the
        application starts accepting trading/analysis requests.
        """
        try:
            # Import here to avoid circular imports
            from quant.data.integrity import (
                SmartCatchUpService,
                OHLCVValidator,
                ActionProcessor,
                MarketCalendar,
            )
            from quant.data.integrity.fetcher import YFinanceFetcher
            from quant.data.parquet_io import ParquetReader, ParquetWriter, get_data_lake_path
            from quant.data.data_provider import ParquetDataProvider
            
            data_lake_path = get_data_lake_path()
            
            # Initialize components
            data_provider = ParquetDataProvider(str(data_lake_path))
            fetcher = YFinanceFetcher()
            reader = ParquetReader(str(data_lake_path))
            writer = ParquetWriter(str(data_lake_path))
            validator = OHLCVValidator()
            processor = ActionProcessor()
            calendar = MarketCalendar('NYSE')
            
            # Create catch-up service
            service = SmartCatchUpService(
                data_provider=data_provider,
                data_fetcher=fetcher,
                parquet_reader=reader,
                parquet_writer=writer,
                validator=validator,
                processor=processor,
                market_calendar=calendar,
                max_retries=3,
                drop_rate_threshold=0.10,
                initial_lookback_days=365 * 2,
            )
            
            # Check gap status first
            gap_status = service.get_gap_status()
            logger.info(f"Data lake status: {gap_status}")
            
            if not gap_status.get('needs_backfill', False):
                logger.info("✅ Data is up-to-date, no backfill needed")
                self._catchup_result = {
                    'ready': True,
                    'days_backfilled': 0,
                    'message': 'Data already up-to-date'
                }
                return
            
            # Run catch-up
            logger.info(f"🔄 Starting catch-up (gap: {gap_status.get('gap_days', 'unknown')} days)...")
            ready, days = service.check_and_backfill(max_gap_days=self.max_gap_days)
            
            # Get detailed result
            result = service.get_detailed_result()
            self._catchup_result = result.to_dict()
            
            if ready:
                logger.info(f"✅ Catch-up complete: {days} days backfilled")
                logger.info(f"   Tickers updated: {result.tickers_updated}")
                logger.info(f"   Rows added: {result.rows_added}")
                if result.tickers_failed > 0:
                    logger.warning(f"   Tickers failed: {result.tickers_failed}")
                if result.confirmed_spikes:
                    logger.warning(f"   Confirmed spikes: {len(result.confirmed_spikes)}")
            else:
                logger.error(f"❌ Catch-up failed: {result.errors}")
                
        except ImportError as e:
            logger.warning(f"Catch-up service not available: {e}")
            self._catchup_result = {
                'ready': True,
                'days_backfilled': 0,
                'message': f'Catch-up skipped: {e}'
            }
        except Exception as e:
            logger.error(f"Catch-up failed with error: {e}")
            self._catchup_result = {
                'ready': False,
                'days_backfilled': 0,
                'error': str(e)
            }
    
    def get_catchup_result(self) -> Optional[dict]:
        """Get the result of the last catch-up operation."""
        return self._catchup_result


# Global instance
startup_service = StartupService(enable_catchup=True)
