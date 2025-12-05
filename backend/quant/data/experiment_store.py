"""
Experiment Store - Track backtest runs with full lineage

Professional-grade experiment tracking for quantitative research.
Stores all backtest parameters, results, and artifacts for reproducibility.
"""

import json
import hashlib
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging

logger = logging.getLogger(__name__)


def get_experiments_path() -> Path:
    """Get experiments directory path."""
    import os
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return Path(base) / 'data_lake' / 'experiments'


class ExperimentStore:
    """
    Store and retrieve backtest experiments with full lineage.
    
    Each experiment includes:
    - Config: All parameters used
    - Results: Equity curve, trades, metrics
    - Metadata: Timestamp, git SHA, duration
    
    Directory structure:
        experiments/
        └── {run_id}/
            ├── config.json
            ├── equity_curve.parquet
            ├── trades.parquet
            ├── metrics.json
            └── metadata.json
    """
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else get_experiments_path()
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def create_run_id(self, config: Dict[str, Any]) -> str:
        """
        Create unique run ID from config hash + timestamp.
        
        Format: {date}_{short_hash}
        Example: 2024-12-05_a1b2c3
        """
        # Create hash from config
        config_str = json.dumps(config, sort_keys=True, default=str)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]
        
        # Combine with date
        run_id = f"{date.today().isoformat()}_{config_hash}"
        
        return run_id
    
    def save_experiment(
        self,
        config: Dict[str, Any],
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame = None,
        metrics: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Save a complete experiment.
        
        Args:
            config: All backtest parameters
            equity_curve: DataFrame with date, portfolio_value, benchmark, etc.
            trades: DataFrame of executed trades (optional)
            metrics: Performance metrics dict (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            run_id: Unique identifier for this run
        """
        run_id = self.create_run_id(config)
        run_path = self.base_path / run_id
        
        # Handle duplicate run IDs (add suffix)
        counter = 1
        while run_path.exists():
            run_id = f"{run_id}_{counter}"
            run_path = self.base_path / run_id
            counter += 1
        
        run_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Save config
        with open(run_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        # 2. Save equity curve
        if equity_curve is not None and not equity_curve.empty:
            table = pa.Table.from_pandas(equity_curve, preserve_index=True)
            pq.write_table(table, run_path / 'equity_curve.parquet', compression='snappy')
        
        # 3. Save trades
        if trades is not None and not trades.empty:
            table = pa.Table.from_pandas(trades, preserve_index=False)
            pq.write_table(table, run_path / 'trades.parquet', compression='snappy')
        
        # 4. Save metrics
        if metrics:
            with open(run_path / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
        
        # 5. Save metadata
        meta = {
            'run_id': run_id,
            'created_at': datetime.now().isoformat(),
            'config_hash': hashlib.md5(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest(),
            **(metadata or {})
        }
        
        # Try to get git SHA
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, cwd=str(self.base_path.parent.parent)
            )
            if result.returncode == 0:
                meta['git_sha'] = result.stdout.strip()
        except:
            pass
        
        with open(run_path / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"Saved experiment: {run_id}")
        return run_id
    
    def load_experiment(self, run_id: str) -> Dict[str, Any]:
        """Load a complete experiment."""
        run_path = self.base_path / run_id
        
        if not run_path.exists():
            raise FileNotFoundError(f"Experiment not found: {run_id}")
        
        result = {'run_id': run_id}
        
        # Load config
        config_file = run_path / 'config.json'
        if config_file.exists():
            with open(config_file) as f:
                result['config'] = json.load(f)
        
        # Load equity curve
        equity_file = run_path / 'equity_curve.parquet'
        if equity_file.exists():
            result['equity_curve'] = pq.read_table(equity_file).to_pandas()
        
        # Load trades
        trades_file = run_path / 'trades.parquet'
        if trades_file.exists():
            result['trades'] = pq.read_table(trades_file).to_pandas()
        
        # Load metrics
        metrics_file = run_path / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                result['metrics'] = json.load(f)
        
        # Load metadata
        meta_file = run_path / 'metadata.json'
        if meta_file.exists():
            with open(meta_file) as f:
                result['metadata'] = json.load(f)
        
        return result
    
    def list_experiments(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent experiments."""
        experiments = []
        
        for run_dir in sorted(self.base_path.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            
            meta_file = run_dir / 'metadata.json'
            config_file = run_dir / 'config.json'
            
            exp = {'run_id': run_dir.name}
            
            if meta_file.exists():
                with open(meta_file) as f:
                    exp['metadata'] = json.load(f)
            
            if config_file.exists():
                with open(config_file) as f:
                    exp['config'] = json.load(f)
            
            experiments.append(exp)
            
            if len(experiments) >= limit:
                break
        
        return experiments
    
    def delete_experiment(self, run_id: str):
        """Delete an experiment."""
        import shutil
        run_path = self.base_path / run_id
        
        if run_path.exists():
            shutil.rmtree(run_path)
            logger.info(f"Deleted experiment: {run_id}")
        else:
            raise FileNotFoundError(f"Experiment not found: {run_id}")


# Singleton instance
_store_instance = None

def get_experiment_store() -> ExperimentStore:
    """Get singleton ExperimentStore instance."""
    global _store_instance
    if _store_instance is None:
        _store_instance = ExperimentStore()
    return _store_instance
