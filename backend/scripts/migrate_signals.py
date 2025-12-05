#!/usr/bin/env python3
"""
Migrate Signals and Targets from SQLite to Parquet

One-time migration script to move model_signals and portfolio_targets
from SQLite to the new Parquet-based SignalStore.

Usage:
    cd backend
    python scripts/migrate_signals.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime, date
import argparse
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sqlalchemy import create_engine, text

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_sqlite_path() -> Path:
    """Get path to SQLite database."""
    base = Path(__file__).parent.parent
    return base / 'data' / 'database.sqlite'


def migrate_signals():
    """Migrate model_signals table to Parquet."""
    logger.info("=" * 50)
    logger.info("Migrating model_signals to Parquet...")
    
    from quant.data.signal_store import SignalStore
    
    store = SignalStore()
    sqlite_path = get_sqlite_path()
    
    if not sqlite_path.exists():
        logger.warning(f"SQLite database not found: {sqlite_path}")
        return 0
    
    engine = create_engine(f'sqlite:///{sqlite_path}')
    
    # Query all signals with ticker
    query = text("""
        SELECT 
            s.ticker,
            ms.date,
            ms.model_name,
            ms.score,
            ms.rank,
            ms.metadata_json
        FROM model_signals ms
        JOIN securities s ON ms.sid = s.sid
        ORDER BY ms.date, ms.model_name, ms.rank
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    if df.empty:
        logger.warning("No signals found in SQLite")
        return 0
    
    logger.info(f"Found {len(df)} signals to migrate")
    
    # Group by date and model, write each group
    total_written = 0
    grouped = df.groupby(['date', 'model_name'])
    
    for (signal_date, model_name), group in grouped:
        # Convert date string to date object if needed
        if isinstance(signal_date, str):
            signal_date = datetime.strptime(signal_date, '%Y-%m-%d').date()
        
        # Prepare DataFrame for SignalStore
        signals_df = group[['ticker', 'score', 'rank', 'metadata_json']].copy()
        
        result = store.write_signals(signal_date, model_name, signals_df)
        total_written += result['rows_written']
    
    logger.info(f"Migrated {total_written} signals across {len(grouped)} date/model combinations")
    return total_written


def migrate_targets():
    """Migrate portfolio_targets table to Parquet."""
    logger.info("=" * 50)
    logger.info("Migrating portfolio_targets to Parquet...")
    
    from quant.data.signal_store import SignalStore
    
    store = SignalStore()
    sqlite_path = get_sqlite_path()
    
    if not sqlite_path.exists():
        logger.warning(f"SQLite database not found: {sqlite_path}")
        return 0
    
    engine = create_engine(f'sqlite:///{sqlite_path}')
    
    query = text("""
        SELECT 
            s.ticker,
            pt.date,
            pt.model_name,
            pt.weight
        FROM portfolio_targets pt
        JOIN securities s ON pt.sid = s.sid
        ORDER BY pt.date, pt.model_name, pt.weight DESC
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    if df.empty:
        logger.warning("No targets found in SQLite")
        return 0
    
    logger.info(f"Found {len(df)} targets to migrate")
    
    # Group by date and model
    total_written = 0
    grouped = df.groupby(['date', 'model_name'])
    
    for (target_date, model_name), group in grouped:
        if isinstance(target_date, str):
            target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        
        targets_df = group[['ticker', 'weight']].copy()
        
        result = store.write_targets(target_date, model_name, targets_df)
        total_written += result['rows_written']
    
    logger.info(f"Migrated {total_written} targets across {len(grouped)} date/model combinations")
    return total_written


def verify_migration():
    """Verify migration success."""
    logger.info("=" * 50)
    logger.info("Verifying migration...")
    
    from quant.data.signal_store import SignalStore
    
    store = SignalStore()
    stats = store.get_stats()
    
    logger.info(f"Signal files: {stats['signal_files']}")
    logger.info(f"Target files: {stats['target_files']}")
    logger.info(f"Total size: {stats['total_size_mb']} MB")
    
    # Test reading
    latest_signals = store.get_latest_signals()
    latest_targets = store.get_latest_targets()
    
    logger.info(f"Latest signals: {len(latest_signals)} rows")
    logger.info(f"Latest targets: {len(latest_targets)} rows")
    
    if not latest_signals.empty:
        logger.info(f"Signal date range: {store.list_available_dates('signals')[:3]}...")
    
    if not latest_targets.empty:
        logger.info(f"Target date range: {store.list_available_dates('targets')[:3]}...")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Migrate signals to Parquet')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify existing migration')
    parser.add_argument('--signals-only', action='store_true',
                        help='Only migrate signals')
    parser.add_argument('--targets-only', action='store_true',
                        help='Only migrate targets')
    args = parser.parse_args()
    
    if args.verify_only:
        verify_migration()
        return 0
    
    signals_count = 0
    targets_count = 0
    
    if not args.targets_only:
        signals_count = migrate_signals()
    
    if not args.signals_only:
        targets_count = migrate_targets()
    
    logger.info("=" * 50)
    logger.info("Migration Complete")
    logger.info("=" * 50)
    logger.info(f"Signals migrated: {signals_count}")
    logger.info(f"Targets migrated: {targets_count}")
    
    # Verify
    verify_migration()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
