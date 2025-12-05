"""
Parquet Data Writer

High-performance Parquet file writer with:
- Year-based partitioning for efficient queries
- Snappy compression for speed/size balance
- Append and overwrite modes
- Schema validation
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
from datetime import date, datetime
from typing import List, Optional, Dict, Any
import logging
import json

logger = logging.getLogger(__name__)


# Standard schemas for data lake tables
PRICE_SCHEMA = pa.schema([
    ('ticker', pa.string()),
    ('date', pa.date32()),
    ('open', pa.float64()),
    ('high', pa.float64()),
    ('low', pa.float64()),
    ('close', pa.float64()),
    ('adj_close', pa.float64()),
    ('volume', pa.int64()),
    ('year', pa.int32()),  # Partition column
])

FUNDAMENTAL_SCHEMA = pa.schema([
    ('ticker', pa.string()),
    ('date', pa.date32()),
    ('period_end', pa.date32()),
    ('filing_date', pa.date32()),  # For PIT accuracy
    # Wide format columns - main metrics
    ('total_revenue', pa.float64()),
    ('gross_profit', pa.float64()),
    ('ebitda', pa.float64()),
    ('operating_income', pa.float64()),
    ('net_income', pa.float64()),
    ('total_assets', pa.float64()),
    ('total_liabilities', pa.float64()),
    ('total_equity', pa.float64()),
    ('cash_and_equivalents', pa.float64()),
    ('total_debt', pa.float64()),
    ('operating_cash_flow', pa.float64()),
    ('free_cash_flow', pa.float64()),
    ('capex', pa.float64()),
    # Derived ratios
    ('roe', pa.float64()),
    ('roa', pa.float64()),
    ('roic', pa.float64()),
    ('debt_to_equity', pa.float64()),
    ('current_ratio', pa.float64()),
    ('gross_margin', pa.float64()),
    ('operating_margin', pa.float64()),
    ('net_margin', pa.float64()),
    ('year', pa.int32()),  # Partition column
])

SIGNAL_SCHEMA = pa.schema([
    ('ticker', pa.string()),
    ('date', pa.date32()),
    ('model_name', pa.string()),
    ('score', pa.float64()),
    ('rank', pa.int32()),
    ('factor_breakdown', pa.string()),  # JSON string
])


class ParquetWriter:
    """
    Efficient Parquet file writer for data lake.
    
    Features:
    - Automatic partitioning by year
    - Schema enforcement
    - Incremental updates
    - Metadata tracking
    """
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def write_prices(
        self, 
        df: pd.DataFrame, 
        mode: str = 'append'  # 'append' or 'overwrite'
    ) -> Dict[str, Any]:
        """
        Write price data with year partitioning.
        
        Args:
            df: DataFrame with columns matching PRICE_SCHEMA
            mode: 'append' to add to existing, 'overwrite' to replace
            
        Returns:
            Write statistics
        """
        target_path = self.base_path / 'raw' / 'prices'
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Ensure date column is proper type
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date
            
        # Add year partition column
        df['year'] = pd.to_datetime(df['date']).dt.year
        
        # Convert to PyArrow table
        table = pa.Table.from_pandas(df, preserve_index=False)
        
        # Write with partitioning
        pq.write_to_dataset(
            table,
            root_path=str(target_path),
            partition_cols=['year'],
            compression='snappy',
            existing_data_behavior='overwrite_or_ignore' if mode == 'append' else 'delete_matching'
        )
        
        stats = {
            'rows_written': len(df),
            'tickers': df['ticker'].nunique(),
            'date_range': (df['date'].min(), df['date'].max()),
            'path': str(target_path)
        }
        
        logger.info(f"Wrote {stats['rows_written']} price records to {target_path}")
        return stats
    
    def write_fundamentals(
        self, 
        df: pd.DataFrame, 
        mode: str = 'append'
    ) -> Dict[str, Any]:
        """
        Write fundamental data in wide format.
        """
        target_path = self.base_path / 'raw' / 'fundamentals'
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Ensure date columns
        for col in ['date', 'period_end', 'filing_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.date
        
        # Add year partition
        df['year'] = pd.to_datetime(df['date']).dt.year
        
        table = pa.Table.from_pandas(df, preserve_index=False)
        
        pq.write_to_dataset(
            table,
            root_path=str(target_path),
            partition_cols=['year'],
            compression='snappy',
            existing_data_behavior='overwrite_or_ignore' if mode == 'append' else 'delete_matching'
        )
        
        stats = {
            'rows_written': len(df),
            'tickers': df['ticker'].nunique(),
            'path': str(target_path)
        }
        
        logger.info(f"Wrote {stats['rows_written']} fundamental records")
        return stats
    
    def write_signals(
        self, 
        df: pd.DataFrame,
        model_name: str,
        signal_date: date
    ) -> Dict[str, Any]:
        """
        Write model signals for a specific date.
        """
        target_path = self.base_path / 'processed' / 'signals'
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        df['model_name'] = model_name
        df['date'] = signal_date
        
        # Create filename with date for easy access
        filename = f"{model_name}_{signal_date.isoformat()}.parquet"
        
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, target_path / filename, compression='snappy')
        
        return {
            'rows_written': len(df),
            'path': str(target_path / filename)
        }
    
    def write_snapshot(
        self,
        universe: List[str],
        features: pd.DataFrame,
        snapshot_date: date,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create a Point-in-Time snapshot for reproducibility.
        
        Snapshots capture:
        - Universe membership
        - All features as of that date
        - Metadata for lineage
        """
        snapshot_path = self.base_path / 'snapshots' / snapshot_date.isoformat()
        snapshot_path.mkdir(parents=True, exist_ok=True)
        
        # Write universe
        universe_df = pd.DataFrame({'ticker': universe})
        pq.write_table(
            pa.Table.from_pandas(universe_df),
            snapshot_path / 'universe.parquet'
        )
        
        # Write features
        pq.write_table(
            pa.Table.from_pandas(features),
            snapshot_path / 'features.parquet',
            compression='snappy'
        )
        
        # Write metadata
        meta = {
            'snapshot_date': snapshot_date.isoformat(),
            'created_at': datetime.now().isoformat(),
            'universe_size': len(universe),
            'feature_count': len(features.columns),
            **(metadata or {})
        }
        
        with open(snapshot_path / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"Created PIT snapshot for {snapshot_date}")
        return {'path': str(snapshot_path), 'metadata': meta}


class ParquetReader:
    """
    Efficient Parquet file reader for data lake.
    """
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        
    def read_prices(
        self,
        tickers: List[str] = None,
        start_date: date = None,
        end_date: date = None,
        columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Read price data with optional filtering.
        
        Uses predicate pushdown for efficient reads.
        """
        price_path = self.base_path / 'raw' / 'prices'
        
        if not price_path.exists():
            return pd.DataFrame()
        
        # Build filters
        filters = []
        if start_date:
            filters.append(('date', '>=', start_date))
        if end_date:
            filters.append(('date', '<=', end_date))
        if tickers:
            filters.append(('ticker', 'in', tickers))
        
        try:
            table = pq.read_table(
                price_path,
                columns=columns,
                filters=filters if filters else None
            )
            return table.to_pandas()
        except Exception as e:
            logger.error(f"Error reading prices: {e}")
            return pd.DataFrame()
    
    def read_snapshot(self, snapshot_date: date) -> Dict[str, Any]:
        """
        Read a PIT snapshot.
        
        Returns:
            Dict with 'universe', 'features', and 'metadata'
        """
        snapshot_path = self.base_path / 'snapshots' / snapshot_date.isoformat()
        
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot not found for {snapshot_date}")
        
        universe = pq.read_table(snapshot_path / 'universe.parquet').to_pandas()
        features = pq.read_table(snapshot_path / 'features.parquet').to_pandas()
        
        with open(snapshot_path / 'metadata.json') as f:
            metadata = json.load(f)
        
        return {
            'universe': universe['ticker'].tolist(),
            'features': features,
            'metadata': metadata
        }
    
    def list_snapshots(self) -> List[date]:
        """List available snapshot dates."""
        snapshot_path = self.base_path / 'snapshots'
        
        if not snapshot_path.exists():
            return []
        
        dates = []
        for d in snapshot_path.iterdir():
            if d.is_dir():
                try:
                    dates.append(date.fromisoformat(d.name))
                except ValueError:
                    pass
        
        return sorted(dates)


# Convenience functions
def get_data_lake_path() -> Path:
    """Get default data lake path."""
    import os
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return Path(base) / 'data_lake'


def get_writer() -> ParquetWriter:
    """Get default ParquetWriter."""
    return ParquetWriter(str(get_data_lake_path()))


def get_reader() -> ParquetReader:
    """Get default ParquetReader."""
    return ParquetReader(str(get_data_lake_path()))
