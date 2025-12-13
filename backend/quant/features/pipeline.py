"""Feature Pipeline with Hash-Based Caching.

Processes raw factor values into clean, tradable signals.
Supports Parquet-based caching for repeated computations.

Integration with Registry Pattern:
    Use FactorPipeline.from_config() to create a configuration-driven pipeline.
"""

import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, TYPE_CHECKING
import logging

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependencies
if TYPE_CHECKING:
    from quant.features.dynamic_pipeline import DynamicFactorPipeline


class FactorPipeline:
    """Processes raw factor values into clean, tradable signals.
    
    For configuration-driven factor computation using the Registry Pattern,
    use the `from_config()` class method or `DynamicFactorPipeline` directly.
    """
    
    @classmethod
    def from_config(cls, config_path: str) -> "DynamicFactorPipeline":
        """Create a dynamic pipeline from configuration file.
        
        This method provides integration with the new Registry Pattern.
        Factors are loaded dynamically based on YAML/JSON configuration.
        
        Args:
            config_path: Path to YAML or JSON strategy configuration.
            
        Returns:
            DynamicFactorPipeline instance configured from file.
            
        Example:
            pipeline = FactorPipeline.from_config("config/strategies.yaml")
            factor_results = pipeline.compute_all(market_data)
        """
        from quant.features.dynamic_pipeline import DynamicFactorPipeline
        return DynamicFactorPipeline(config_path=config_path)
    
    @staticmethod
    def _get_cache_key(df: pd.DataFrame) -> str:
        """Compute hash of DataFrame for cache key.
        
        Uses MD5 hash of the pandas object hash for speed.
        """
        return hashlib.md5(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()
    
    @staticmethod
    def winsorize(series: pd.Series, limits: List[float] = [0.01, 0.01]) -> pd.Series:
        """Clip outliers at the given percentiles (e.g., 1% and 99%)."""
        if series.empty: 
            return series
        return series.clip(
            lower=series.quantile(limits[0]), 
            upper=series.quantile(1 - limits[1])
        )
        
    @staticmethod
    def z_score(series: pd.Series) -> pd.Series:
        """Normalize to Mean=0, Std=1."""
        if series.empty: 
            return series
        std = series.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=series.index)
        return (series - series.mean()) / std
        
    @staticmethod
    def neutralize(series: pd.Series, groups: pd.Series) -> pd.Series:
        """Subtract group (sector) means from the scores."""
        if series.empty or groups.empty: 
            return series
        
        df = pd.DataFrame({'score': series, 'group': groups})
        group_means = df.groupby('group')['score'].transform('mean')
        return df['score'] - group_means

    @classmethod
    def process_factors(
        cls, 
        df: pd.DataFrame, 
        sector_col: str = 'sector',
        cache_dir: str = None
    ) -> pd.DataFrame:
        """Main pipeline execution with optional caching.
        
        Args:
            df: DataFrame where columns are raw factors and rows are tickers
            sector_col: Column name for sector grouping
            cache_dir: Optional directory for Parquet cache
            
        Returns:
            Processed DataFrame with z-scores and neutralized factors
        """
        # Check cache if enabled
        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            
            key = cls._get_cache_key(df)
            cache_file = cache_path / f"{key}.parquet"
            
            if cache_file.exists():
                try:
                    logger.debug(f"Cache hit: {cache_file.name}")
                    return pd.read_parquet(cache_file)
                except Exception as e:
                    logger.warning(f"Cache read failed: {e}, recomputing...")
        
        # Compute factors
        processed = cls._compute_factors(df, sector_col)
        
        # Store in cache if enabled
        if cache_dir:
            try:
                processed.to_parquet(cache_file)
                logger.debug(f"Cached result: {cache_file.name}")
            except Exception as e:
                logger.warning(f"Cache write failed: {e}")
        
        return processed
    
    @classmethod
    def _compute_factors(cls, df: pd.DataFrame, sector_col: str) -> pd.DataFrame:
        """Internal factor computation logic."""
        processed = df.copy()
        
        # 1. Handle QMJ Composite if components exist
        if 'roe' in processed.columns and 'debt_to_equity' in processed.columns:
            z_roe = cls.z_score(cls.winsorize(processed['roe']))
            z_gm = cls.z_score(cls.winsorize(processed.get('gross_margin', pd.Series(dtype=float))))
            z_de = cls.z_score(cls.winsorize(processed['debt_to_equity']))
            
            # Combine: Profitability + Safety (Low Debt)
            processed['quality'] = (z_roe + z_gm - z_de) / 3.0
            
        # 2. Process Top-Level Factors
        factors_to_process = [
            'momentum', 'volatility_scaled_momentum', 
            'betting_against_beta', 'quality', 'value', 'upside'
        ]
        
        for factor in factors_to_process:
            if factor in processed.columns:
                # A. Winsorize
                processed[factor] = cls.winsorize(processed[factor])
                
                # C. Z-Score (Global)
                processed[f'z_{factor}'] = cls.z_score(processed[factor])
                
                # D. Neutralize (Sector)
                if sector_col in processed.columns:
                    processed[f'z_{factor}_neutral'] = cls.neutralize(
                        processed[f'z_{factor}'], 
                        processed[sector_col]
                    )
                    
        return processed

