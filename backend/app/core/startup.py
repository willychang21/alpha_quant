"""Application startup hooks.

Handles initialization tasks that should run when the server starts,
including the Smart Catch-Up Service for automatic data backfill.

FAANG-style startup: All initialization happens automatically on launch.
"""

import logging
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)


class StartupService:
    """
    Manages application startup tasks.
    
    FAANG-style initialization pipeline:
    1. Seed securities (S&P 500 + Nasdaq 100)
    2. Run SmartCatchUpService to ensure data lake is up-to-date
    3. Generate signals if stale (rankings, portfolio targets)
    """
    
    def __init__(
        self, 
        enable_catchup: bool = True, 
        enable_seeding: bool = True,
        enable_signals: bool = True,
        max_gap_days: int = 30,
        signal_staleness_days: int = 7
    ):
        """
        Initialize startup service.
        
        Args:
            enable_catchup: Whether to run catch-up on startup
            enable_seeding: Whether to seed securities on startup
            enable_signals: Whether to generate signals if stale
            max_gap_days: Maximum gap size to attempt backfill
            signal_staleness_days: Days before signals are considered stale
        """
        self.enable_catchup = enable_catchup
        self.enable_seeding = enable_seeding
        self.enable_signals = enable_signals
        self.max_gap_days = max_gap_days
        self.signal_staleness_days = signal_staleness_days
        self._catchup_result = None
        self._signals_result = None
    
    async def run_startup_tasks(self):
        """
        Run all startup tasks.
        
        Called by FastAPI lifespan context manager.
        """
        logger.info("=" * 50)
        logger.info("🚀 Startup Pipeline")
        logger.info("=" * 50)
        
        # Phase 1: Seed securities
        if self.enable_seeding:
            await self._seed_securities()
        
        # Phase 2: Data catch-up
        if self.enable_catchup:
            await self._run_catchup()
        
        # Phase 3: Generate signals if stale
        if self.enable_signals:
            await self._generate_signals_if_stale()
        else:
            logger.info("⏸️  Phase 3: Signal generation disabled (skipping)")
        
        logger.info("=" * 50)
        logger.info("✅ Startup tasks complete")
        logger.info("=" * 50)
    
    async def _seed_securities(self):
        """
        Seed securities from S&P 500 and Nasdaq 100.
        Only adds new securities, doesn't duplicate.
        """
        try:
            import pandas as pd
            import requests
            from io import StringIO
            from app.core.database import SessionLocal
            from quant.data.models import Security
            
            logger.info("📊 Phase 1: Checking securities...")
            
            db = SessionLocal()
            existing_count = db.query(Security).count()
            
            if existing_count >= 400:  # Already seeded
                logger.info(f"   Securities already seeded ({existing_count} found)")
                db.close()
                return
            
            logger.info("   Fetching S&P 500 and Nasdaq 100 tickers...")
            
            # Fetch S&P 500
            sp500_tickers = []
            try:
                url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, timeout=10)
                tables = pd.read_html(StringIO(response.text))
                sp500_tickers = [t.replace('.', '-') for t in tables[0]['Symbol'].tolist()]
            except Exception as e:
                logger.warning(f"   Failed to fetch S&P 500: {e}")
            
            # Fetch Nasdaq 100
            nasdaq_tickers = []
            try:
                url = "https://en.wikipedia.org/wiki/Nasdaq-100"
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, timeout=10)
                tables = pd.read_html(StringIO(response.text))
                for table in tables:
                    if 'Ticker' in table.columns:
                        nasdaq_tickers = table['Ticker'].tolist()
                        break
                    if 'Symbol' in table.columns:
                        nasdaq_tickers = table['Symbol'].tolist()
                        break
            except Exception as e:
                logger.warning(f"   Failed to fetch Nasdaq 100: {e}")
            
            # Combine and deduplicate
            all_tickers = sorted(list(set(sp500_tickers + nasdaq_tickers)))
            
            if not all_tickers:
                logger.warning("   No tickers fetched, using fallback list")
                all_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
            
            # Add new securities
            added_count = 0
            for ticker in all_tickers:
                exists = db.query(Security).filter(Security.ticker == ticker).first()
                if not exists:
                    sec = Security(ticker=ticker, name=ticker, type="Equity")
                    db.add(sec)
                    added_count += 1
            
            db.commit()
            db.close()
            
            logger.info(f"   ✅ Added {added_count} new securities (total: {existing_count + added_count})")
            
        except Exception as e:
            logger.error(f"   ❌ Securities seeding failed: {e}")
    
    async def _generate_signals_if_stale(self):
        """
        Generate ranking signals and portfolio targets if not already computed today.
        
        Logic: Only regenerate if latest signals are NOT from today.
        This ensures fresh signals on first startup of the day, but skips
        on subsequent restarts within the same day.
        """
        try:
            from quant.data.signal_store import get_signal_store
            from app.core.database import SessionLocal
            
            logger.info("📈 Phase 3: Checking signal freshness (Parquet)...")
            
            today = date.today()
            
            # Check latest signal date from Parquet SignalStore
            store = get_signal_store()
            signal_dates = store.list_available_dates('signals')
            target_dates = store.list_available_dates('targets')
            
            latest_signal_date = signal_dates[0] if signal_dates else None
            latest_target_date = target_dates[0] if target_dates else None
            
            # Only regenerate if NOT computed today
            signals_stale = latest_signal_date != today
            targets_stale = latest_target_date != today
            
            if not signals_stale and not targets_stale:
                logger.info(f"   ✅ Signals already computed today ({latest_signal_date})")
                logger.info(f"   ✅ Targets already computed today ({latest_target_date})")
                return
            
            logger.info(f"   Signals need update: {signals_stale} (last: {latest_signal_date})")
            logger.info(f"   Targets need update: {targets_stale} (last: {latest_target_date})")
            logger.info("   🔄 Generating fresh signals for today (ENABLE_SIGNALS=True)...")
            
            # Need db session for RankingEngine and PortfolioOptimizer (for Security lookups)
            db = SessionLocal()
            
            # Run ranking engine
            if signals_stale:
                try:
                    from quant.selection.ranking import RankingEngine
                    ranking_engine = RankingEngine(db)
                    await ranking_engine.run_ranking(today)
                    logger.info("   ✅ Rankings generated (Parquet)")
                except Exception as e:
                    logger.warning(f"   ⚠️ Ranking generation failed: {e}")
            
            # Run portfolio optimizer
            if targets_stale:
                try:
                    from quant.portfolio.optimizer import PortfolioOptimizer
                    optimizer = PortfolioOptimizer(db)
                    optimizer.run_optimization(
                        today,
                        optimizer='kelly',
                        target_vol=0.15,
                        sector_constraints=True,
                        beta_constraints=True
                    )
                    logger.info("   ✅ Portfolio targets generated (Parquet)")
                except Exception as e:
                    logger.warning(f"   ⚠️ Portfolio optimization failed: {e}")
            
            db.close()
            
            self._signals_result = {
                'signals_generated': signals_stale,
                'targets_generated': targets_stale,
                'date': str(today)
            }
            
        except Exception as e:
            logger.error(f"   ❌ Signal generation failed: {e}")
    
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

            logger.info("📊 Phase 2: Data catch-up...")
            
            data_lake_path = get_data_lake_path()
            
            # Initialize components
            data_provider = ParquetDataProvider(str(data_lake_path))
            fetcher = YFinanceFetcher(progress=True)
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


# Global instance - FAANG-style: all initialization enabled by default
import os
enable_signals_env = os.getenv("ENABLE_SIGNALS", "false").lower() == "true"

startup_service = StartupService(
    enable_catchup=True,
    enable_seeding=True,
    enable_signals=enable_signals_env,
    max_gap_days=30,
    signal_staleness_days=0
)
