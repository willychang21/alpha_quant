"""
Sector Rotation Analyzer Module.

Implements Relative Rotation Graph (RRG) methodology for sector rotation analysis.
Tracks the 11 GICS sectors relative to SPY benchmark to identify capital flows.

RRG Quadrants:
- Leading (top-right): RS Ratio > 100 AND RS Momentum > 0
- Weakening (bottom-right): RS Ratio > 100 AND RS Momentum < 0
- Lagging (bottom-left): RS Ratio < 100 AND RS Momentum < 0
- Improving (top-left): RS Ratio < 100 AND RS Momentum > 0

Reference: Julius de Kempenaer's Relative Rotation Graphs
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime
import logging

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from quant.features.capital_flow.models import SectorRotationResult

logger = logging.getLogger(__name__)


class SectorRotationAnalyzer:
    """Analyzes sector rotation using Relative Rotation Graph methodology."""
    
    # 11 GICS Sector ETFs
    SECTOR_ETFS = {
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLE': 'Energy',
        'XLU': 'Utilities',
        'XLI': 'Industrials',
        'XLC': 'Communication Services',
        'XLB': 'Materials',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLV': 'Healthcare',
        'XLRE': 'Real Estate'
    }
    
    # Mapping from sector names to ETF symbols
    SECTOR_TO_ETF = {v: k for k, v in SECTOR_ETFS.items()}
    
    def __init__(
        self,
        benchmark: str = 'SPY',
        rs_period: int = 14,
        momentum_period: int = 14
    ):
        """Initialize SectorRotationAnalyzer.
        
        Args:
            benchmark: Benchmark ETF symbol (default: SPY)
            rs_period: Period for RS Ratio SMA calculation (default: 14)
            momentum_period: Period for RS Momentum calculation (default: 14)
        """
        self.benchmark = benchmark
        self.rs_period = rs_period
        self.momentum_period = momentum_period
        
        # Cache for previous quadrant states (for transition detection)
        self._previous_quadrants: Dict[str, str] = {}
    
    def calculate_rs_ratio(
        self,
        sector_prices: pd.Series,
        benchmark_prices: pd.Series
    ) -> pd.Series:
        """Calculate Relative Strength Ratio.
        
        RS Ratio = (Sector Price / Benchmark Price) * 100, smoothed by SMA.
        Normalized around 100 (above 100 = outperforming, below = underperforming).
        
        Args:
            sector_prices: Sector ETF price series
            benchmark_prices: Benchmark (SPY) price series
            
        Returns:
            Series with RS Ratio values (normalized around 100)
        """
        if len(sector_prices) < self.rs_period:
            logger.warning("Insufficient data for RS Ratio calculation")
            return pd.Series([100.0] * len(sector_prices), index=sector_prices.index)
        
        try:
            # Calculate raw relative strength
            rs_raw = sector_prices / benchmark_prices
            
            # Normalize to base 100 using first value as reference
            base_rs = rs_raw.iloc[:self.rs_period].mean()
            if base_rs > 0:
                rs_normalized = (rs_raw / base_rs) * 100
            else:
                rs_normalized = rs_raw * 100
            
            # Apply SMA smoothing
            rs_ratio = rs_normalized.rolling(window=self.rs_period).mean()
            
            # Fill NaN with neutral value
            rs_ratio = rs_ratio.fillna(100.0)
            
            return rs_ratio
            
        except Exception as e:
            logger.error(f"RS Ratio calculation failed: {e}")
            return pd.Series([100.0] * len(sector_prices), index=sector_prices.index)
    
    def calculate_rs_momentum(
        self,
        rs_ratio: pd.Series
    ) -> pd.Series:
        """Calculate RS Momentum (rate of change of RS Ratio).
        
        RS Momentum = (Current RS Ratio / RS Ratio N periods ago - 1) * 100
        
        Args:
            rs_ratio: RS Ratio series
            
        Returns:
            Series with RS Momentum values
        """
        if len(rs_ratio) < self.momentum_period + 1:
            logger.warning("Insufficient data for RS Momentum calculation")
            return pd.Series([0.0] * len(rs_ratio), index=rs_ratio.index)
        
        try:
            # Rate of change (percentage change over momentum period)
            rs_momentum = rs_ratio.pct_change(periods=self.momentum_period) * 100
            
            # Fill NaN with neutral value
            rs_momentum = rs_momentum.fillna(0.0)
            
            return rs_momentum
            
        except Exception as e:
            logger.error(f"RS Momentum calculation failed: {e}")
            return pd.Series([0.0] * len(rs_ratio), index=rs_ratio.index)
    
    def classify_quadrant(
        self,
        rs_ratio: float,
        rs_momentum: float
    ) -> str:
        """Classify sector into RRG quadrant.
        
        Quadrant classification is deterministic:
        - Leading: RS Ratio > 100 AND RS Momentum > 0
        - Weakening: RS Ratio > 100 AND RS Momentum < 0
        - Lagging: RS Ratio < 100 AND RS Momentum < 0
        - Improving: RS Ratio < 100 AND RS Momentum > 0
        
        Args:
            rs_ratio: Current RS Ratio value
            rs_momentum: Current RS Momentum value
            
        Returns:
            Quadrant name: 'Leading', 'Weakening', 'Lagging', or 'Improving'
        """
        if rs_ratio >= 100 and rs_momentum >= 0:
            return 'Leading'
        elif rs_ratio >= 100 and rs_momentum < 0:
            return 'Weakening'
        elif rs_ratio < 100 and rs_momentum < 0:
            return 'Lagging'
        else:  # rs_ratio < 100 and rs_momentum >= 0
            return 'Improving'
    
    def analyze_sector(
        self,
        sector_symbol: str,
        sector_prices: pd.Series,
        benchmark_prices: pd.Series,
        timestamp: Optional[datetime] = None
    ) -> SectorRotationResult:
        """Analyze a single sector's rotation status.
        
        Args:
            sector_symbol: Sector ETF symbol
            sector_prices: Sector price series
            benchmark_prices: Benchmark price series
            timestamp: Analysis timestamp (default: now)
            
        Returns:
            SectorRotationResult with analysis
        """
        timestamp = timestamp or datetime.now()
        sector_name = self.SECTOR_ETFS.get(sector_symbol, sector_symbol)
        
        try:
            # Calculate RS Ratio and Momentum
            rs_ratio_series = self.calculate_rs_ratio(sector_prices, benchmark_prices)
            rs_momentum_series = self.calculate_rs_momentum(rs_ratio_series)
            
            # Get current values
            current_rs_ratio = float(rs_ratio_series.iloc[-1])
            current_rs_momentum = float(rs_momentum_series.iloc[-1])
            
            # Classify quadrant
            quadrant = self.classify_quadrant(current_rs_ratio, current_rs_momentum)
            
            # Check for transition
            previous_quadrant = self._previous_quadrants.get(sector_symbol)
            transition_signal = False
            
            if previous_quadrant is not None and previous_quadrant != quadrant:
                transition_signal = True
                # Special signal for Lagging -> Improving (early capital inflow)
                if previous_quadrant == 'Lagging' and quadrant == 'Improving':
                    logger.info(f"🚀 {sector_symbol} ({sector_name}): Lagging -> Improving transition detected!")
            
            # Update cache
            self._previous_quadrants[sector_symbol] = quadrant
            
            return SectorRotationResult(
                symbol=sector_symbol,
                sector_name=sector_name,
                rs_ratio=current_rs_ratio,
                rs_momentum=current_rs_momentum,
                quadrant=quadrant,
                previous_quadrant=previous_quadrant,
                transition_signal=transition_signal,
                timestamp=timestamp
            )
            
        except Exception as e:
            logger.error(f"Sector analysis failed for {sector_symbol}: {e}")
            return SectorRotationResult(
                symbol=sector_symbol,
                sector_name=sector_name,
                rs_ratio=100.0,
                rs_momentum=0.0,
                quadrant='Lagging',
                previous_quadrant=None,
                transition_signal=False,
                timestamp=timestamp
            )
    
    def analyze_all_sectors(
        self,
        period: str = '6mo'
    ) -> Dict[str, SectorRotationResult]:
        """Analyze all sector ETFs and return rotation data.
        
        Args:
            period: Historical data period for yfinance (default: '6mo')
            
        Returns:
            Dictionary mapping sector symbol to SectorRotationResult
        """
        if not YFINANCE_AVAILABLE:
            logger.error("yfinance not available. Cannot fetch sector data.")
            return {}
        
        results = {}
        timestamp = datetime.now()
        
        try:
            # Fetch all sector ETFs and benchmark in one call
            all_symbols = list(self.SECTOR_ETFS.keys()) + [self.benchmark]
            
            logger.info(f"Fetching sector rotation data for {len(all_symbols)} symbols...")
            
            # Download data
            data = yf.download(
                all_symbols,
                period=period,
                progress=False,
                group_by='ticker'
            )
            
            if data.empty:
                logger.warning("No sector data received from yfinance")
                return {}
            
            # Extract benchmark prices
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    benchmark_prices = data[self.benchmark]['Close']
                else:
                    benchmark_prices = data['Close']
            except KeyError:
                logger.error(f"Could not extract benchmark ({self.benchmark}) prices")
                return {}
            
            # Analyze each sector
            for symbol, sector_name in self.SECTOR_ETFS.items():
                try:
                    # Extract sector prices
                    if isinstance(data.columns, pd.MultiIndex):
                        sector_prices = data[symbol]['Close']
                    else:
                        continue  # Single ticker mode, shouldn't happen
                    
                    # Clean data
                    valid_mask = sector_prices.notna() & benchmark_prices.notna()
                    sector_prices_clean = sector_prices[valid_mask]
                    benchmark_prices_clean = benchmark_prices[valid_mask]
                    
                    if len(sector_prices_clean) < self.rs_period + self.momentum_period:
                        logger.warning(f"Insufficient data for {symbol}")
                        continue
                    
                    result = self.analyze_sector(
                        sector_symbol=symbol,
                        sector_prices=sector_prices_clean,
                        benchmark_prices=benchmark_prices_clean,
                        timestamp=timestamp
                    )
                    
                    results[symbol] = result
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze {symbol}: {e}")
                    continue
            
            logger.info(f"Analyzed {len(results)} sectors successfully")
            
            return results
            
        except Exception as e:
            logger.error(f"Sector analysis failed: {e}")
            return {}
    
    def get_sector_quadrant(self, sector_name: str) -> str:
        """Get the current quadrant for a sector by name.
        
        Args:
            sector_name: Sector name (e.g., 'Technology')
            
        Returns:
            Quadrant name or 'Unknown' if not available
        """
        symbol = self.SECTOR_TO_ETF.get(sector_name)
        if symbol and symbol in self._previous_quadrants:
            return self._previous_quadrants[symbol]
        return 'Unknown'
    
    def get_quadrant_score(self, quadrant: str) -> float:
        """Convert quadrant to a numeric score.
        
        Args:
            quadrant: Quadrant name
            
        Returns:
            Score: positive for favorable, negative for unfavorable
        """
        quadrant_scores = {
            'Leading': 1.0,
            'Improving': 0.5,
            'Weakening': -0.5,
            'Lagging': -1.0,
            'Unknown': 0.0
        }
        return quadrant_scores.get(quadrant, 0.0)
