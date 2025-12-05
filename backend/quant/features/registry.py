"""
Feature Registry

Central registry for all computed features used in factor models.

Inspired by Two Sigma / AQR feature engineering practices:
1. Every feature has a clear definition and metadata
2. Features are versioned for reproducibility
3. Proper handling of lookback requirements
4. Caching and incremental computation support
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional
from datetime import date, timedelta
from enum import Enum
import pandas as pd
import numpy as np
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


class FeatureCategory(Enum):
    """Standard factor categories."""
    VALUE = "value"
    QUALITY = "quality"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    SIZE = "size"
    GROWTH = "growth"
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    COMPOSITE = "composite"


class FeatureFrequency(Enum):
    """Feature update frequency."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class FeatureDefinition:
    """
    Definition of a computed feature.
    
    Attributes:
        name: Unique identifier (e.g., 'momentum_12m')
        category: Factor category
        description: Human-readable description
        compute_fn: Function that computes the feature
        lookback_days: Days of historical data required
        frequency: How often the feature updates
        version: Feature version for tracking changes
        dependencies: Other features this depends on
        params: Configuration parameters
    """
    name: str
    category: FeatureCategory
    description: str
    compute_fn: Callable
    lookback_days: int
    frequency: FeatureFrequency = FeatureFrequency.DAILY
    version: str = "1.0"
    dependencies: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Generate hash for cache invalidation
        param_str = json.dumps(self.params, sort_keys=True)
        self.hash = hashlib.md5(
            f"{self.name}:{self.version}:{param_str}".encode()
        ).hexdigest()[:8]
    
    def compute(self, data: pd.DataFrame, as_of_date: date) -> pd.Series:
        """Execute the compute function with proper error handling."""
        try:
            return self.compute_fn(data, as_of_date, **self.params)
        except Exception as e:
            logger.error(f"Error computing feature {self.name}: {e}")
            # Return NaN series on error
            return pd.Series(dtype=float)


class FeatureRegistry:
    """
    Central registry for all computed features.
    
    Usage:
        registry = FeatureRegistry()
        
        @registry.register(
            name='momentum_12m',
            category=FeatureCategory.MOMENTUM,
            description='12-month price momentum',
            lookback_days=252
        )
        def compute_momentum_12m(data: pd.DataFrame, as_of_date: date) -> pd.Series:
            # ... implementation
            pass
        
        # Compute all features
        features = registry.compute_all(data, as_of_date)
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._features: Dict[str, FeatureDefinition] = {}
            cls._instance._cache: Dict[str, pd.DataFrame] = {}
        return cls._instance
    
    def register(
        self,
        name: str,
        category: FeatureCategory,
        description: str,
        lookback_days: int,
        frequency: FeatureFrequency = FeatureFrequency.DAILY,
        version: str = "1.0",
        dependencies: List[str] = None,
        params: Dict[str, Any] = None
    ):
        """
        Decorator to register a feature compute function.
        """
        def decorator(fn: Callable):
            definition = FeatureDefinition(
                name=name,
                category=category,
                description=description,
                compute_fn=fn,
                lookback_days=lookback_days,
                frequency=frequency,
                version=version,
                dependencies=dependencies or [],
                params=params or {}
            )
            self._features[name] = definition
            logger.debug(f"Registered feature: {name} (v{version})")
            return fn
        return decorator
    
    def register_definition(self, definition: FeatureDefinition):
        """Register a pre-built feature definition."""
        self._features[definition.name] = definition
    
    def get(self, name: str) -> Optional[FeatureDefinition]:
        """Get a feature definition by name."""
        return self._features.get(name)
    
    def list_features(
        self, 
        category: FeatureCategory = None
    ) -> List[FeatureDefinition]:
        """List all registered features, optionally filtered by category."""
        features = list(self._features.values())
        if category:
            features = [f for f in features if f.category == category]
        return features
    
    def compute_feature(
        self,
        name: str,
        data: pd.DataFrame,
        as_of_date: date,
        use_cache: bool = True
    ) -> pd.Series:
        """Compute a single feature."""
        if name not in self._features:
            raise ValueError(f"Unknown feature: {name}")
        
        definition = self._features[name]
        
        # Check cache
        cache_key = f"{name}:{definition.hash}:{as_of_date}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Compute dependencies first
        for dep_name in definition.dependencies:
            if dep_name not in data.columns:
                data[dep_name] = self.compute_feature(dep_name, data, as_of_date)
        
        # Compute feature
        result = definition.compute(data, as_of_date)
        
        # Cache result
        if use_cache:
            self._cache[cache_key] = result
        
        return result
    
    def compute_all(
        self,
        data: pd.DataFrame,
        as_of_date: date,
        categories: List[FeatureCategory] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Compute all registered features.
        
        Args:
            data: Input data (prices, fundamentals)
            as_of_date: Date for PIT computation
            categories: Optional filter by category
            use_cache: Whether to use cached results
            
        Returns:
            DataFrame with one column per feature
        """
        features_to_compute = self.list_features()
        if categories:
            features_to_compute = [
                f for f in features_to_compute 
                if f.category in categories
            ]
        
        # Topological sort by dependencies
        features_to_compute = self._topological_sort(features_to_compute)
        
        result = pd.DataFrame(index=data.index if 'ticker' in data.columns 
                              else data['ticker'].unique() if 'ticker' in data.columns 
                              else range(len(data)))
        
        for definition in features_to_compute:
            try:
                result[definition.name] = self.compute_feature(
                    definition.name, data, as_of_date, use_cache
                )
            except Exception as e:
                logger.error(f"Failed to compute {definition.name}: {e}")
                result[definition.name] = np.nan
        
        return result
    
    def _topological_sort(
        self, 
        features: List[FeatureDefinition]
    ) -> List[FeatureDefinition]:
        """Sort features by dependency order."""
        # Simple implementation - assumes no cycles
        by_name = {f.name: f for f in features}
        visited = set()
        result = []
        
        def visit(f: FeatureDefinition):
            if f.name in visited:
                return
            for dep in f.dependencies:
                if dep in by_name:
                    visit(by_name[dep])
            visited.add(f.name)
            result.append(f)
        
        for f in features:
            visit(f)
        
        return result
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get registry metadata for documentation."""
        return {
            'total_features': len(self._features),
            'by_category': {
                cat.value: len([f for f in self._features.values() if f.category == cat])
                for cat in FeatureCategory
            },
            'features': [
                {
                    'name': f.name,
                    'category': f.category.value,
                    'description': f.description,
                    'lookback_days': f.lookback_days,
                    'version': f.version
                }
                for f in self._features.values()
            ]
        }
    
    def clear_cache(self):
        """Clear the feature cache."""
        self._cache.clear()
        logger.info("Feature cache cleared")


# Global registry instance
registry = FeatureRegistry()


# ============================================================
# Standard Feature Definitions
# ============================================================

@registry.register(
    name='momentum_12m',
    category=FeatureCategory.MOMENTUM,
    description='12-month price momentum (excluding most recent month)',
    lookback_days=252
)
def compute_momentum_12m(data: pd.DataFrame, as_of_date: date, **kwargs) -> pd.Series:
    """
    12-month momentum, excluding the most recent month to avoid reversal effect.
    Standard academic definition.
    """
    if 'adj_close' not in data.columns:
        return pd.Series(dtype=float)
    
    # Group by ticker, calculate 12m return excluding last month
    def calc_mom(group):
        if len(group) < 252:
            return np.nan
        sorted_group = group.sort_values('date')
        price_12m_ago = sorted_group.iloc[-252]['adj_close']
        price_1m_ago = sorted_group.iloc[-21]['adj_close']  # Skip last month
        return (price_1m_ago / price_12m_ago) - 1
    
    return data.groupby('ticker').apply(calc_mom)


@registry.register(
    name='momentum_6m',
    category=FeatureCategory.MOMENTUM,
    description='6-month price momentum',
    lookback_days=126
)
def compute_momentum_6m(data: pd.DataFrame, as_of_date: date, **kwargs) -> pd.Series:
    """6-month momentum."""
    if 'adj_close' not in data.columns:
        return pd.Series(dtype=float)
    
    def calc_mom(group):
        if len(group) < 126:
            return np.nan
        sorted_group = group.sort_values('date')
        price_6m_ago = sorted_group.iloc[-126]['adj_close']
        current_price = sorted_group.iloc[-1]['adj_close']
        return (current_price / price_6m_ago) - 1
    
    return data.groupby('ticker').apply(calc_mom)


@registry.register(
    name='volatility_60d',
    category=FeatureCategory.VOLATILITY,
    description='60-day realized volatility (annualized)',
    lookback_days=60
)
def compute_volatility_60d(data: pd.DataFrame, as_of_date: date, **kwargs) -> pd.Series:
    """Annualized 60-day volatility."""
    if 'adj_close' not in data.columns:
        return pd.Series(dtype=float)
    
    def calc_vol(group):
        if len(group) < 60:
            return np.nan
        sorted_group = group.sort_values('date').tail(60)
        returns = sorted_group['adj_close'].pct_change().dropna()
        return returns.std() * np.sqrt(252)
    
    return data.groupby('ticker').apply(calc_vol)


@registry.register(
    name='price_to_earnings',
    category=FeatureCategory.VALUE,
    description='Price to trailing 12-month earnings',
    lookback_days=30
)
def compute_pe_ratio(data: pd.DataFrame, as_of_date: date, **kwargs) -> pd.Series:
    """P/E ratio from fundamentals."""
    # This would typically use fundamental data
    # Placeholder - would be computed from merged price + fundamental data
    if 'market_cap' in data.columns and 'net_income' in data.columns:
        return data['market_cap'] / data['net_income']
    return pd.Series(dtype=float)


@registry.register(
    name='roe',
    category=FeatureCategory.QUALITY,
    description='Return on Equity',
    lookback_days=0,
    frequency=FeatureFrequency.QUARTERLY
)
def compute_roe(data: pd.DataFrame, as_of_date: date, **kwargs) -> pd.Series:
    """ROE from fundamentals - already in wide format."""
    if 'roe' in data.columns:
        return data['roe']
    if 'net_income' in data.columns and 'total_equity' in data.columns:
        return data['net_income'] / data['total_equity']
    return pd.Series(dtype=float)


@registry.register(
    name='revenue_growth_yoy',
    category=FeatureCategory.GROWTH,
    description='Year-over-year revenue growth',
    lookback_days=365,
    frequency=FeatureFrequency.QUARTERLY
)
def compute_revenue_growth(data: pd.DataFrame, as_of_date: date, **kwargs) -> pd.Series:
    """YoY revenue growth."""
    if 'total_revenue' not in data.columns:
        return pd.Series(dtype=float)
    
    # Would require multiple periods of fundamental data
    # Placeholder
    return pd.Series(dtype=float)


# ============================================================
# Composite Features
# ============================================================

@registry.register(
    name='quality_score',
    category=FeatureCategory.COMPOSITE,
    description='Composite quality score combining ROE, margins, and earnings stability',
    lookback_days=0,
    dependencies=['roe']
)
def compute_quality_score(data: pd.DataFrame, as_of_date: date, **kwargs) -> pd.Series:
    """
    Composite quality score.
    Combines multiple quality metrics into a single z-score.
    """
    metrics = []
    
    if 'roe' in data.columns:
        roe_zscore = (data['roe'] - data['roe'].mean()) / data['roe'].std()
        metrics.append(roe_zscore)
    
    if 'gross_margin' in data.columns:
        margin_zscore = (data['gross_margin'] - data['gross_margin'].mean()) / data['gross_margin'].std()
        metrics.append(margin_zscore)
    
    if not metrics:
        return pd.Series(dtype=float)
    
    return pd.concat(metrics, axis=1).mean(axis=1)
