"""
Application Startup Hooks
=========================

Manages the initialization pipeline of the DCA Quant application.
This module implements the "On-Startup" pattern to ensure the application 
state is consistent and ready for business logic.

Key Responsibilities:
---------------------
1.  **Security Seeding**: Ensuring the reference universe (S&P 500, Nasdaq 100) is populated.
2.  **Smart Catch-Up**: Syncing the data lake with the latest market data.
3.  **Signal Generation**: Updating quantitative signals if they are stale.

FAANG Design Principles:
-   **Idempotency**: All operations can be run multiple times without side effects.
-   **Observability**: Clear logging of progress and failures.
-   **Resilience**: Failures in non-critical paths (like seeding) do not block startup.

"""

import logging
import asyncio
from datetime import date
from typing import List, Optional, Dict, Any, Union
from io import StringIO

import requests
import pandas as pd
from sqlalchemy.orm import Session

# Local application imports
# Using lazy imports inside methods where appropriate to avoid circular dependency issues during early startup
# but standard imports for utilities.
from app.core.database import SessionLocal
from quant.data.models import Security

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_NASDAQ_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'

DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']

logger = logging.getLogger(__name__)


class StartupService:
    """
    Orchestrates the application startup startup lifecycle.
    """

    def __init__(
        self,
        enable_catchup: bool = True,
        enable_seeding: bool = True,
        enable_signals: bool = True,
        max_gap_days: int = 30,
    ):
        self.enable_catchup = enable_catchup
        self.enable_seeding = enable_seeding
        self.enable_signals = enable_signals
        self.max_gap_days = max_gap_days

        # State tracking for health checks or debugging
        self._catchup_result: Optional[Dict[str, Any]] = None
        self._signals_result: Optional[Dict[str, Any]] = None

    async def run_startup_tasks(self) -> None:
        """
        Execute the full startup pipeline.
        This is the main entry point called by the FastAPI lifespan handler.
        """
        logger.info("=" * 60)
        logger.info("🚀  DCA Quant Engine - Startup Pipeline Initiated")
        logger.info("=" * 60)

        # Phase 1: Seed Securities
        # We run this first so subsequent steps (catch-up) have a universe to work with.
        if self.enable_seeding:
            await self._seed_securities()
        else:
            logger.info("⏩  Phase 1: Seeding skipped (config)")

        # Phase 2: Data Lake Catch-up
        if self.enable_catchup:
            await self._run_catchup()
        else:
            logger.info("⏩  Phase 2: Data catch-up skipped (config)")

        # Phase 3: Signal Generation
        if self.enable_signals:
            await self._generate_signals_if_stale()
        else:
            logger.info("⏩  Phase 3: Signal generation skipped (config)")

        logger.info("=" * 60)
        logger.info("✅  Startup Pipeline Completed")
        logger.info("=" * 60)

    # --------------------------------------------------------------------------
    # Phase 1: Security Seeding
    # --------------------------------------------------------------------------

    async def _seed_securities(self) -> None:
        """
        Populate the database with the initial universe of securities.
        Fetches fresh lists from Wikipedia (S&P 500 & Nasdaq 100).
        """
        logger.info("🔎  Phase 1: Verifying Security Universe...")

        try:
            # Run in thread pool to avoid blocking async loop with synchronous requests/db
            await asyncio.to_thread(self._seed_securities_sync)
        except Exception as e:
             logger.error(f"❌  Phase 1 Failed: {e}", exc_info=True)

    def _seed_securities_sync(self) -> None:
        """Synchronous implementation of security seeding."""
        with SessionLocal() as db:
            existing_count = db.query(Security).count()
            
            # Optimization: If we have a healthy number of stocks, assume seeded.
            # In production, you might want to run this weekly regardless.
            if existing_count >= 400:
                logger.info(f"    ✔ Universe populated ({existing_count} securities detected).")
                return

            logger.info("    Fetching ticker lists from external sources...")
            tickers = self._fetch_all_tickers()
            
            self._upsert_securities(db, tickers)

    def _fetch_all_tickers(self) -> List[str]:
        """Fetch and aggregate tickers from all configured sources."""
        sp500 = self._fetch_tickers_from_wiki(WIKI_SP500_URL, "S&P 500")
        nasdaq = self._fetch_tickers_from_wiki(WIKI_NASDAQ_URL, "Nasdaq 100")
        
        all_tickers = sorted(list(set(sp500 + nasdaq)))
        
        if not all_tickers:
            logger.warning("    ⚠️ Failed to fetch external tickers. Using fallback list.")
            return DEFAULT_TICKERS
            
        return all_tickers

    def _fetch_tickers_from_wiki(self, url: str, name: str) -> List[str]:
        """Helper to scrape tickers from Wikipedia."""
        try:
            response = requests.get(url, headers={'User-Agent': USER_AGENT}, timeout=10)
            response.raise_for_status()
            
            tables = pd.read_html(StringIO(response.text))
            
            # Heuristic: Find the table with 'Symbol' or 'Ticker'
            for table in tables:
                if 'Symbol' in table.columns:
                    return [t.replace('.', '-') for t in table['Symbol'].tolist()]
                if 'Ticker' in table.columns:
                    return [t.replace('.', '-') for t in table['Ticker'].tolist()]
                    
            logger.warning(f"    ⚠️ Beppie could not find symbol column for {name}")
            return []
        except Exception as e:
            logger.warning(f"    ⚠️ Failed to fetch {name}: {str(e)}")
            return []

    def _upsert_securities(self, db: Session, tickers: List[str]) -> None:
        """Add missing securities to the database."""
        added_count = 0
        # Bulk query for efficiency could be better, but loop is fine for startup < 1000 items
        # A set check is faster than DB queries in loop
        existing_tickers = {s.ticker for s in db.query(Security.ticker).all()}
        
        for ticker in tickers:
            if ticker not in existing_tickers:
                sec = Security(ticker=ticker, name=ticker, type="Equity")
                db.add(sec)
                added_count += 1
        
        db.commit()
        if added_count > 0:
            logger.info(f"    ✔ Added {added_count} new securities.")
        else:
            logger.info("    ✔ No new securities found.")

    # --------------------------------------------------------------------------
    # Phase 2: Data Catch-up
    # --------------------------------------------------------------------------

    async def _run_catchup(self) -> None:
        """
        Run the Smart Catch-Up Service to backfill missing OHLCV data.
        """
        logger.info("📥  Phase 2: Synchronizing Data Lake...")

        try:
            # Lazy imports to optimize startup time and avoid circular refs
            from quant.data.integrity import (
                SmartCatchUpService, OHLCVValidator, ActionProcessor, MarketCalendar
            )
            from quant.data.integrity.fetcher import YFinanceFetcher
            from quant.data.parquet_io import ParquetReader, ParquetWriter, get_data_lake_path
            from quant.data.data_provider import ParquetDataProvider

            data_lake_path = get_data_lake_path()
            
            # Dependency Injection for the Service
            service = SmartCatchUpService(
                data_provider=ParquetDataProvider(str(data_lake_path)),
                data_fetcher=YFinanceFetcher(progress=True),
                parquet_reader=ParquetReader(str(data_lake_path)),
                parquet_writer=ParquetWriter(str(data_lake_path)),
                validator=OHLCVValidator(),
                processor=ActionProcessor(),
                market_calendar=MarketCalendar('NYSE'),
                max_retries=3,
                drop_rate_threshold=0.10,
                initial_lookback_days=365 * 2,
            )
            
            # Check status
            gap_status = await asyncio.to_thread(service.get_gap_status)
            
            if not gap_status.get('needs_backfill', False):
                logger.info("    ✔ Data Lake is up-to-date.")
                self._catchup_result = {'ready': True, 'message': 'Up to date'}
                return

            logger.info(f"    🔄 Backfill needed (Gap: {gap_status.get('gap_days')} days). executing...")
            
            ready, days = await asyncio.to_thread(service.check_and_backfill, max_gap_days=self.max_gap_days)
            result = service.get_detailed_result()
            self._catchup_result = result.to_dict()

            if ready:
                logger.info(f"    ✔ Upload complete. {result.rows_added} rows added across {result.tickers_updated} tickers.")
            else:
                logger.error(f"    ❌ Partial or failed catchup: {result.errors}")

        except Exception as e:
            logger.error(f"    ❌ Phase 2 Failed: {e}")
            self._catchup_result = {'ready': False, 'error': str(e)}

    # --------------------------------------------------------------------------
    # Phase 3: Signal Generation
    # --------------------------------------------------------------------------

    async def _generate_signals_if_stale(self) -> None:
        """
        Trigger Ranking and Portfolio Optimization if data for today is missing.
        """
        logger.info("🧠  Phase 3: Determining Signal Freshness...")
        
        try:
            from quant.data.signal_store import get_signal_store
            from quant.selection.ranking import RankingEngine
            from quant.portfolio.optimizer import PortfolioOptimizer
            
            store = get_signal_store()
            today = date.today()
            
            latest_ranking = self._get_latest_date(store, 'signals')
            latest_target = self._get_latest_date(store, 'targets')
            
            # Logic: Update if stale
            if latest_ranking == today and latest_target == today:
                 logger.info("    ✔ Signals are fresh for today.")
                 return

            logger.info("    🔄 Computation required.")
            
            # Using SessionLocal context logic
            # Database access needs to correspond to the execution context
            
            if latest_ranking != today:
                # Assuming execute_ranking might be async based on original code `await`
                logger.info("    Running Ranking Engine...")
                with SessionLocal() as db:
                     engine = RankingEngine(db)
                     if asyncio.iscoroutinefunction(engine.run_ranking):
                         await engine.run_ranking(today)
                     else:
                         engine.run_ranking(today)
            
            if latest_target != today:
                logger.info("    Running Portfolio Optimizer...")
                with SessionLocal() as db:
                    optimizer = PortfolioOptimizer(db)
                    opts = dict(
                        target_date=today,
                        optimizer='kelly',
                        target_vol=0.15,
                        sector_constraints=True,
                        beta_constraints=True
                    )
                    if asyncio.iscoroutinefunction(optimizer.run_optimization):
                        await optimizer.run_optimization(**opts)
                    else:
                        optimizer.run_optimization(**opts)

            logger.info("    ✔ Signals updated successfully.")

        except Exception as e:
            logger.error(f"    ❌ Phase 3 Failed: {e}", exc_info=True)

    def _get_latest_date(self, store: Any, key: str) -> Optional[date]:
        dates = store.list_available_dates(key)
        return dates[0] if dates else None


# Global instance
startup_service = StartupService(
    enable_catchup=True,
    enable_seeding=True,
    enable_signals=True,
    max_gap_days=30,
)
