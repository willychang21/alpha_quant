#!/usr/bin/env python3
"""
SQLite to Parquet Migration Script

Migrates data from the legacy SQLite database to the new Parquet-based data lake.

Usage:
    cd backend
    python scripts/migrate_sqlite_to_parquet.py
    
Options:
    --verify-only    Only verify existing migration, don't write
    --force          Overwrite existing Parquet files
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import argparse
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MigrationStats:
    """Track migration statistics."""
    def __init__(self):
        self.prices_migrated = 0
        self.fundamentals_migrated = 0
        self.securities_migrated = 0
        self.errors = []
        self.start_time = datetime.now()
    
    def report(self):
        duration = (datetime.now() - self.start_time).total_seconds()
        logger.info("=" * 60)
        logger.info("Migration Complete")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration:.1f} seconds")
        logger.info(f"Prices migrated: {self.prices_migrated:,}")
        logger.info(f"Fundamentals migrated: {self.fundamentals_migrated:,}")
        logger.info(f"Securities migrated: {self.securities_migrated:,}")
        if self.errors:
            logger.warning(f"Errors: {len(self.errors)}")
            for e in self.errors[:5]:
                logger.warning(f"  - {e}")


def get_sqlite_path() -> Path:
    """Get path to SQLite database."""
    base = Path(__file__).parent.parent
    return base / 'data' / 'database.sqlite'


def get_data_lake_path() -> Path:
    """Get path to data lake."""
    base = Path(__file__).parent.parent
    return base / 'data_lake'


def migrate_securities(engine, writer, stats: MigrationStats):
    """Migrate securities master data."""
    logger.info("Migrating securities...")
    
    query = text("""
        SELECT sid, ticker, name, exchange, type, active
        FROM securities
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    if df.empty:
        logger.warning("No securities found")
        return {}
    
    # Create SID to ticker mapping for later use
    sid_to_ticker = dict(zip(df['sid'], df['ticker']))
    
    # Write to data lake
    data_lake = get_data_lake_path()
    securities_path = data_lake / 'raw' / 'securities'
    securities_path.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(securities_path / 'securities.parquet', index=False)
    
    stats.securities_migrated = len(df)
    logger.info(f"Migrated {len(df)} securities")
    
    return sid_to_ticker


def migrate_prices(engine, sid_to_ticker: dict, stats: MigrationStats):
    """Migrate price data with year partitioning."""
    logger.info("Migrating prices...")
    
    from quant.data.parquet_io import ParquetWriter
    from datetime import datetime as dt
    
    writer = ParquetWriter(str(get_data_lake_path()))
    
    # Get date range
    with engine.connect() as conn:
        range_query = text("SELECT MIN(date) as min_date, MAX(date) as max_date FROM market_data_daily")
        result = conn.execute(range_query).fetchone()
        min_date_str, max_date_str = result
        
        # Parse dates from strings if needed
        if isinstance(min_date_str, str):
            min_date = dt.strptime(min_date_str, '%Y-%m-%d').date()
            max_date = dt.strptime(max_date_str, '%Y-%m-%d').date()
        else:
            min_date = min_date_str
            max_date = max_date_str
            
        logger.info(f"Price date range: {min_date} to {max_date}")
    
    # Migrate in yearly batches for memory efficiency
    for year in range(min_date.year, max_date.year + 1):
        logger.info(f"  Processing year {year}...")
        
        query = text(f"""
            SELECT 
                md.sid,
                md.date,
                md.open,
                md.high,
                md.low,
                md.close,
                md.adj_close,
                md.volume
            FROM market_data_daily md
            WHERE strftime('%Y', md.date) = '{year}'
            ORDER BY md.sid, md.date
        """)
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        if df.empty:
            continue
        
        # Map SID to ticker
        df['ticker'] = df['sid'].map(sid_to_ticker)
        df = df.dropna(subset=['ticker'])
        
        # Drop SID column (replaced by ticker)
        df = df.drop(columns=['sid'])
        
        # Write to Parquet
        writer.write_prices(df, mode='append')
        
        stats.prices_migrated += len(df)
        logger.info(f"    Year {year}: {len(df):,} rows")
    
    logger.info(f"Total prices migrated: {stats.prices_migrated:,}")


def migrate_fundamentals_to_wide(engine, sid_to_ticker: dict, stats: MigrationStats):
    """
    Migrate fundamentals from EAV to wide format.
    
    EAV (Entity-Attribute-Value) -> Wide Table transformation.
    """
    logger.info("Migrating fundamentals (EAV → Wide format)...")
    
    from quant.data.parquet_io import ParquetWriter
    
    writer = ParquetWriter(str(get_data_lake_path()))
    
    # Query all fundamentals
    query = text("""
        SELECT 
            f.sid,
            f.date,
            f.metric,
            f.value,
            f.period
        FROM fundamentals f
        ORDER BY f.sid, f.date, f.metric
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    if df.empty:
        logger.warning("No fundamentals found")
        return
    
    # Map SID to ticker
    df['ticker'] = df['sid'].map(sid_to_ticker)
    df = df.dropna(subset=['ticker'])
    
    # Pivot from EAV to wide format
    logger.info("  Pivoting to wide format...")
    wide_df = df.pivot_table(
        index=['ticker', 'date'],
        columns='metric',
        values='value',
        aggfunc='first'  # Take first value if duplicates
    ).reset_index()
    
    # Flatten column names
    wide_df.columns.name = None
    
    # Standardize column names (lowercase, underscores)
    wide_df.columns = [
        c.lower().replace(' ', '_').replace('-', '_') 
        for c in wide_df.columns
    ]
    
    # Add year for partitioning
    wide_df['year'] = pd.to_datetime(wide_df['date']).dt.year
    
    # Calculate derived ratios if base data available
    wide_df = calculate_derived_ratios(wide_df)
    
    # Write to Parquet
    writer.write_fundamentals(wide_df, mode='overwrite')
    
    stats.fundamentals_migrated = len(wide_df)
    logger.info(f"Migrated {len(wide_df):,} fundamental records ({len(wide_df.columns)} columns)")


def calculate_derived_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived financial ratios from base metrics."""
    
    # ROE
    if 'net_income' in df.columns and 'total_stockholder_equity' in df.columns:
        df['roe'] = df['net_income'] / df['total_stockholder_equity'].replace(0, np.nan)
    elif 'net_income' in df.columns and 'total_equity' in df.columns:
        df['roe'] = df['net_income'] / df['total_equity'].replace(0, np.nan)
    
    # ROA
    if 'net_income' in df.columns and 'total_assets' in df.columns:
        df['roa'] = df['net_income'] / df['total_assets'].replace(0, np.nan)
    
    # Debt to Equity
    if 'total_debt' in df.columns and 'total_stockholder_equity' in df.columns:
        df['debt_to_equity'] = df['total_debt'] / df['total_stockholder_equity'].replace(0, np.nan)
    
    # Gross Margin
    if 'gross_profit' in df.columns and 'total_revenue' in df.columns:
        df['gross_margin'] = df['gross_profit'] / df['total_revenue'].replace(0, np.nan)
    
    # Operating Margin
    if 'operating_income' in df.columns and 'total_revenue' in df.columns:
        df['operating_margin'] = df['operating_income'] / df['total_revenue'].replace(0, np.nan)
    
    # Net Margin
    if 'net_income' in df.columns and 'total_revenue' in df.columns:
        df['net_margin'] = df['net_income'] / df['total_revenue'].replace(0, np.nan)
    
    return df


def verify_migration(stats: MigrationStats) -> bool:
    """Verify migration integrity."""
    logger.info("Verifying migration...")
    
    import duckdb
    
    data_lake = get_data_lake_path()
    sqlite_path = get_sqlite_path()
    
    conn = duckdb.connect(':memory:')
    
    # Verify price count
    prices_path = data_lake / 'raw' / 'prices'
    if prices_path.exists():
        parquet_count = conn.execute(f"""
            SELECT COUNT(*) FROM read_parquet('{prices_path}/**/*.parquet')
        """).fetchone()[0]
        
        sqlite_engine = create_engine(f'sqlite:///{sqlite_path}')
        with sqlite_engine.connect() as c:
            sqlite_count = c.execute(text("SELECT COUNT(*) FROM market_data_daily")).fetchone()[0]
        
        if parquet_count == sqlite_count:
            logger.info(f"✓ Prices: {parquet_count:,} rows match")
        else:
            logger.error(f"✗ Prices mismatch: Parquet={parquet_count:,}, SQLite={sqlite_count:,}")
            stats.errors.append(f"Price count mismatch: {parquet_count} vs {sqlite_count}")
            return False
    else:
        logger.warning("Prices not migrated yet")
    
    # Verify fundamentals
    fundamentals_path = data_lake / 'raw' / 'fundamentals'
    if fundamentals_path.exists():
        parquet_count = conn.execute(f"""
            SELECT COUNT(*) FROM read_parquet('{fundamentals_path}/**/*.parquet')
        """).fetchone()[0]
        logger.info(f"✓ Fundamentals: {parquet_count:,} rows (wide format)")
    
    # Verify securities
    securities_path = data_lake / 'raw' / 'securities'
    if securities_path.exists():
        parquet_count = conn.execute(f"""
            SELECT COUNT(*) FROM read_parquet('{securities_path}/*.parquet')
        """).fetchone()[0]
        logger.info(f"✓ Securities: {parquet_count:,} rows")
    
    return len(stats.errors) == 0


def run_benchmark():
    """Benchmark query performance: SQLite vs Parquet."""
    logger.info("Running performance benchmark...")
    
    import duckdb
    import time
    
    data_lake = get_data_lake_path()
    sqlite_path = get_sqlite_path()
    
    prices_path = data_lake / 'raw' / 'prices'
    
    if not prices_path.exists():
        logger.warning("Parquet data not available for benchmark")
        return
    
    # Test 1: Time series for single ticker
    logger.info("\nBenchmark 1: 5 years of AAPL prices")
    
    # SQLite
    sqlite_engine = create_engine(f'sqlite:///{sqlite_path}')
    start = time.time()
    with sqlite_engine.connect() as c:
        result = c.execute(text("""
            SELECT md.date, md.adj_close 
            FROM market_data_daily md
            JOIN securities s ON md.sid = s.sid
            WHERE s.ticker = 'AAPL'
            ORDER BY md.date
        """)).fetchall()
    sqlite_time = time.time() - start
    logger.info(f"  SQLite: {sqlite_time*1000:.1f}ms ({len(result)} rows)")
    
    # Parquet/DuckDB
    conn = duckdb.connect(':memory:')
    start = time.time()
    result = conn.execute(f"""
        SELECT date, adj_close 
        FROM read_parquet('{prices_path}/**/*.parquet')
        WHERE ticker = 'AAPL'
        ORDER BY date
    """).fetchall()
    parquet_time = time.time() - start
    logger.info(f"  Parquet: {parquet_time*1000:.1f}ms ({len(result)} rows)")
    logger.info(f"  Speedup: {sqlite_time/parquet_time:.1f}x")
    
    # Test 2: Cross-sectional (all tickers on one date)
    logger.info("\nBenchmark 2: All tickers on 2024-06-01")
    
    start = time.time()
    with sqlite_engine.connect() as c:
        result = c.execute(text("""
            SELECT s.ticker, md.adj_close 
            FROM market_data_daily md
            JOIN securities s ON md.sid = s.sid
            WHERE md.date = '2024-06-01'
        """)).fetchall()
    sqlite_time = time.time() - start
    logger.info(f"  SQLite: {sqlite_time*1000:.1f}ms ({len(result)} rows)")
    
    start = time.time()
    result = conn.execute(f"""
        SELECT ticker, adj_close 
        FROM read_parquet('{prices_path}/**/*.parquet')
        WHERE date = '2024-06-01'
    """).fetchall()
    parquet_time = time.time() - start
    logger.info(f"  Parquet: {parquet_time*1000:.1f}ms ({len(result)} rows)")
    logger.info(f"  Speedup: {sqlite_time/parquet_time:.1f}x")


def main():
    parser = argparse.ArgumentParser(description='Migrate SQLite to Parquet')
    parser.add_argument('--verify-only', action='store_true', 
                        help='Only verify existing migration')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing Parquet files')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run performance benchmark after migration')
    args = parser.parse_args()
    
    stats = MigrationStats()
    
    # Check if SQLite exists
    sqlite_path = get_sqlite_path()
    if not sqlite_path.exists():
        logger.error(f"SQLite database not found: {sqlite_path}")
        return 1
    
    # Create data lake directory
    data_lake = get_data_lake_path()
    data_lake.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Source: {sqlite_path}")
    logger.info(f"Target: {data_lake}")
    
    if args.verify_only:
        success = verify_migration(stats)
        return 0 if success else 1
    
    # Connect to SQLite
    engine = create_engine(f'sqlite:///{sqlite_path}')
    
    try:
        # Import writer
        from quant.data.parquet_io import ParquetWriter
        writer = ParquetWriter(str(data_lake))
        
        # Step 1: Migrate securities
        sid_to_ticker = migrate_securities(engine, writer, stats)
        
        # Step 2: Migrate prices
        migrate_prices(engine, sid_to_ticker, stats)
        
        # Step 3: Migrate fundamentals (EAV → Wide)
        migrate_fundamentals_to_wide(engine, sid_to_ticker, stats)
        
        # Verify
        verify_migration(stats)
        
        # Benchmark
        if args.benchmark:
            run_benchmark()
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        stats.errors.append(str(e))
        raise
    finally:
        stats.report()
    
    return 0 if len(stats.errors) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
