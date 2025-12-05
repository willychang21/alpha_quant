"""
Signal Store - Parquet-based Signal and Target Storage

Replaces SQLite-based ModelSignals and PortfolioTargets tables.
Provides efficient storage and retrieval using DuckDB for queries.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
from pathlib import Path
from datetime import date, datetime
from typing import List, Optional, Dict, Any
import logging
import json

logger = logging.getLogger(__name__)


def get_data_lake_path() -> Path:
    """Get default data lake path."""
    import os
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return Path(base) / 'data_lake'


class SignalStore:
    """
    Parquet-based storage for model signals and portfolio targets.
    
    Replaces the SQLite ModelSignals and PortfolioTargets tables.
    Uses date-partitioned Parquet files for efficient querying.
    
    Directory structure:
        data_lake/
        └── processed/
            ├── signals/
            │   └── {model_name}_{date}.parquet
            └── targets/
                └── {model_name}_{date}.parquet
    """
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else get_data_lake_path()
        self.signals_path = self.base_path / 'processed' / 'signals'
        self.targets_path = self.base_path / 'processed' / 'targets'
        
        # Create directories
        self.signals_path.mkdir(parents=True, exist_ok=True)
        self.targets_path.mkdir(parents=True, exist_ok=True)
        
        # DuckDB connection for queries
        self._conn = duckdb.connect(':memory:')
    
    # =========================================================================
    # Signal Operations
    # =========================================================================
    
    def write_signals(
        self, 
        signal_date: date, 
        model_name: str, 
        signals: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Write ranking signals to Parquet.
        
        Args:
            signal_date: Date of the signals
            model_name: Model identifier (e.g., 'ranking_v3')
            signals: DataFrame with columns: ticker, score, rank, metadata (optional)
            
        Returns:
            Write statistics
        """
        if signals.empty:
            logger.warning(f"No signals to write for {model_name} on {signal_date}")
            return {'rows_written': 0}
        
        # Ensure required columns
        required_cols = ['ticker', 'score', 'rank']
        for col in required_cols:
            if col not in signals.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Add metadata
        signals = signals.copy()
        signals['date'] = signal_date
        signals['model_name'] = model_name
        signals['created_at'] = datetime.now()
        
        # Serialize metadata if present
        if 'metadata' in signals.columns:
            signals['metadata_json'] = signals['metadata'].apply(
                lambda x: json.dumps(x) if isinstance(x, dict) else str(x)
            )
            signals = signals.drop(columns=['metadata'])
        
        # Write to Parquet
        filename = f"{model_name}_{signal_date.isoformat()}.parquet"
        filepath = self.signals_path / filename
        
        table = pa.Table.from_pandas(signals, preserve_index=False)
        pq.write_table(table, filepath, compression='snappy')
        
        logger.info(f"Wrote {len(signals)} signals to {filepath}")
        
        return {
            'rows_written': len(signals),
            'path': str(filepath),
            'date': signal_date.isoformat(),
            'model_name': model_name
        }
    
    def get_latest_signals(
        self, 
        model_name: str = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get most recent signals.
        
        Args:
            model_name: Filter by model (default: any)
            limit: Max rows to return
            
        Returns:
            DataFrame with signal data
        """
        if not any(self.signals_path.iterdir()):
            return pd.DataFrame()
        
        # Find latest file
        files = list(self.signals_path.glob('*.parquet'))
        if not files:
            return pd.DataFrame()
        
        if model_name:
            files = [f for f in files if f.stem.startswith(model_name)]
            if not files:
                return pd.DataFrame()
        
        # Sort by date in filename (descending)
        files.sort(key=lambda f: f.stem.split('_')[-1], reverse=True)
        latest_file = files[0]
        
        # Read with DuckDB
        df = self._conn.execute(f"""
            SELECT * FROM read_parquet('{latest_file}')
            ORDER BY rank ASC
            LIMIT {limit}
        """).df()
        
        return df
    
    def get_signals(
        self,
        ticker: str = None,
        model_name: str = None,
        start_date: date = None,
        end_date: date = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get signals with optional filtering.
        
        Args:
            ticker: Filter by ticker
            model_name: Filter by model name
            start_date: Start date filter
            end_date: End date filter
            limit: Max rows
            
        Returns:
            Filtered DataFrame
        """
        files = list(self.signals_path.glob('*.parquet'))
        if not files:
            return pd.DataFrame()
        
        # Filter files by model name and date range
        filtered_files = []
        for f in files:
            parts = f.stem.rsplit('_', 1)
            if len(parts) == 2:
                file_model, file_date_str = parts
                try:
                    file_date = date.fromisoformat(file_date_str)
                    
                    if model_name and file_model != model_name:
                        continue
                    if start_date and file_date < start_date:
                        continue
                    if end_date and file_date > end_date:
                        continue
                    
                    filtered_files.append(f)
                except ValueError:
                    continue
        
        if not filtered_files:
            return pd.DataFrame()
        
        # Build query
        file_list = ", ".join([f"'{f}'" for f in filtered_files])
        
        where_clauses = []
        if ticker:
            where_clauses.append(f"ticker = '{ticker}'")
        
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        
        query = f"""
            SELECT * FROM read_parquet([{file_list}])
            {where_sql}
            ORDER BY date DESC, rank ASC
            LIMIT {limit}
        """
        
        return self._conn.execute(query).df()
    
    # =========================================================================
    # Portfolio Target Operations
    # =========================================================================
    
    def write_targets(
        self, 
        target_date: date, 
        model_name: str, 
        targets: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Write portfolio targets to Parquet.
        
        Args:
            target_date: Date of the targets
            model_name: Optimizer identifier (e.g., 'mvo_sharpe')
            targets: DataFrame with columns: ticker, weight
            
        Returns:
            Write statistics
        """
        if targets.empty:
            logger.warning(f"No targets to write for {model_name} on {target_date}")
            return {'rows_written': 0}
        
        required_cols = ['ticker', 'weight']
        for col in required_cols:
            if col not in targets.columns:
                raise ValueError(f"Missing required column: {col}")
        
        targets = targets.copy()
        targets['date'] = target_date
        targets['model_name'] = model_name
        targets['created_at'] = datetime.now()
        
        filename = f"{model_name}_{target_date.isoformat()}.parquet"
        filepath = self.targets_path / filename
        
        table = pa.Table.from_pandas(targets, preserve_index=False)
        pq.write_table(table, filepath, compression='snappy')
        
        logger.info(f"Wrote {len(targets)} targets to {filepath}")
        
        return {
            'rows_written': len(targets),
            'path': str(filepath)
        }
    
    def get_latest_targets(
        self, 
        model_name: str = "mvo_sharpe"
    ) -> pd.DataFrame:
        """
        Get most recent portfolio targets.
        
        Args:
            model_name: Optimizer name to filter by
            
        Returns:
            DataFrame with ticker and weight
        """
        files = list(self.targets_path.glob(f'{model_name}_*.parquet'))
        if not files:
            return pd.DataFrame()
        
        # Sort by date descending
        files.sort(key=lambda f: f.stem.split('_')[-1], reverse=True)
        latest_file = files[0]
        
        return self._conn.execute(f"""
            SELECT ticker, weight, date 
            FROM read_parquet('{latest_file}')
            ORDER BY weight DESC
        """).df()
    
    def get_targets_history(
        self,
        model_name: str = "mvo_sharpe",
        start_date: date = None,
        end_date: date = None
    ) -> pd.DataFrame:
        """
        Get historical portfolio targets.
        """
        files = list(self.targets_path.glob(f'{model_name}_*.parquet'))
        if not files:
            return pd.DataFrame()
        
        # Filter by date range
        filtered_files = []
        for f in files:
            try:
                file_date = date.fromisoformat(f.stem.split('_')[-1])
                if start_date and file_date < start_date:
                    continue
                if end_date and file_date > end_date:
                    continue
                filtered_files.append(f)
            except ValueError:
                continue
        
        if not filtered_files:
            return pd.DataFrame()
        
        file_list = ", ".join([f"'{f}'" for f in filtered_files])
        
        return self._conn.execute(f"""
            SELECT * FROM read_parquet([{file_list}])
            ORDER BY date DESC, weight DESC
        """).df()
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def list_available_dates(self, data_type: str = 'signals') -> List[date]:
        """List available dates for signals or targets."""
        path = self.signals_path if data_type == 'signals' else self.targets_path
        files = list(path.glob('*.parquet'))
        
        dates = []
        for f in files:
            try:
                date_str = f.stem.split('_')[-1]
                dates.append(date.fromisoformat(date_str))
            except ValueError:
                continue
        
        return sorted(set(dates), reverse=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        signal_files = list(self.signals_path.glob('*.parquet'))
        target_files = list(self.targets_path.glob('*.parquet'))
        
        total_size = sum(f.stat().st_size for f in signal_files + target_files)
        
        return {
            'signal_files': len(signal_files),
            'target_files': len(target_files),
            'signal_dates': len(self.list_available_dates('signals')),
            'target_dates': len(self.list_available_dates('targets')),
            'total_size_mb': round(total_size / (1024 * 1024), 2)
        }


# Singleton instance
_store_instance = None

def get_signal_store() -> SignalStore:
    """Get singleton SignalStore instance."""
    global _store_instance
    if _store_instance is None:
        _store_instance = SignalStore()
    return _store_instance
