"""Advanced Rotation Factor.

FeatureGenerator that integrates Advanced Market Rotation analysis
into the RankingEngine factor pipeline.

Combines:
- Levy Relative Strength (RSL)
- Mansfield Relative Strength (MRS)
- Volume Structure Analysis
- Scorecard System

Produces a composite score for multi-factor stock selection.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from quant.features.base import FeatureGenerator
from quant.rotation.levy_rs import LevyRSCalculator
from quant.rotation.mansfield_rs import MansfieldRSCalculator
from quant.rotation.volume_structure import VolumeStructureAnalyzer
from quant.rotation.scorecard import ScorecardSystem, ScorecardWeights

logger = logging.getLogger(__name__)


class AdvancedRotationFactor(FeatureGenerator):
    """Advanced Rotation Factor combining relative strength and volume analysis.
    
    Integrates with existing RankingEngine factor pipeline to provide
    a composite score based on multiple technical indicators.
    
    Components:
        - Levy RS: Price momentum relative to moving average
        - Mansfield RS: Relative strength vs benchmark
        - Volume Structure: Institutional activity detection
        - Scorecard: Multi-factor signal generation
    
    Attributes:
        levy_calc: LevyRSCalculator instance
        mansfield_calc: MansfieldRSCalculator instance
        volume_analyzer: VolumeStructureAnalyzer instance
        scorecard: ScorecardSystem instance
    """
    
    def __init__(
        self,
        weights: Optional[ScorecardWeights] = None,
        benchmark_prices: Optional[pd.Series] = None
    ):
        """Initialize Advanced Rotation Factor.
        
        Args:
            weights: Scorecard weights configuration
            benchmark_prices: Optional pre-loaded benchmark prices (SPY)
        """
        self.levy_calc = LevyRSCalculator()
        self.mansfield_calc = MansfieldRSCalculator()
        self.volume_analyzer = VolumeStructureAnalyzer()
        self.scorecard = ScorecardSystem(weights)
        
        self._benchmark_prices: Optional[pd.Series] = benchmark_prices
        self._benchmark_cache_date: Optional[datetime] = None
    
    @property
    def name(self) -> str:
        """Unique name of the factor."""
        return "AdvancedRotation"
    
    @property
    def description(self) -> str:
        """Description of what this factor measures."""
        return "Composite score from Levy RS, Mansfield RS, and volume structure analysis."
    
    def set_benchmark_prices(self, prices: pd.Series) -> None:
        """Set benchmark prices for Mansfield RS calculation.
        
        Args:
            prices: Series of benchmark (SPY) closing prices
        """
        self._benchmark_prices = prices
        self._benchmark_cache_date = datetime.now()
    
    def _get_rrg_quadrant(self, sector: Optional[str]) -> Optional[str]:
        """Get RRG quadrant from Capital Flow Detection system.
        
        Args:
            sector: Stock sector name
            
        Returns:
            RRG quadrant or None if unavailable
        """
        if not sector:
            return None
        
        try:
            from quant.features.capital_flow import CapitalFlowFactor
            cf_factor = CapitalFlowFactor()
            
            # Check if sector data is cached
            if not cf_factor._sector_cache:
                cf_factor.refresh_sector_data()
            
            sector_symbol = cf_factor.sector_analyzer.SECTOR_TO_ETF.get(sector)
            if sector_symbol:
                cached = cf_factor._sector_cache.get(sector_symbol, {})
                return cached.get('quadrant', 'Unknown')
            
            return None
        except Exception as e:
            logger.debug(f"Could not get RRG quadrant: {e}")
            return None
    
    def compute(
        self,
        history: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
        ticker: str = None,
        sector: str = None
    ) -> pd.Series:
        """Compute advanced rotation score for a stock.
        
        Args:
            history: DataFrame with OHLCV data (columns: Open, High, Low, Close, Volume)
            fundamentals: Optional fundamental data (for future use)
            ticker: Stock ticker symbol
            sector: Stock sector for RRG lookup
            
        Returns:
            Series with composite advanced rotation score
        """
        if history.empty:
            return pd.Series([0.0], index=[pd.Timestamp.now()])
        
        try:
            # Extract OHLCV columns (handle both cases)
            close_col = 'Close' if 'Close' in history.columns else 'close'
            high_col = 'High' if 'High' in history.columns else 'high'
            low_col = 'Low' if 'Low' in history.columns else 'low'
            volume_col = 'Volume' if 'Volume' in history.columns else 'volume'
            
            close = history[close_col]
            high = history[high_col]
            low = history[low_col]
            volume = history[volume_col]
            
            # 1. Calculate Levy RS
            rsl_series = self.levy_calc.calculate_rsl(close)
            current_rsl = float(rsl_series.iloc[-1])
            rsl_signal = self.levy_calc.get_momentum_signal(current_rsl)
            
            # Approximate percentile (assume neutral if no universe)
            # RSL of 1.0 = 50th percentile, 1.1 = 75th, 0.9 = 25th (rough estimate)
            rsl_percentile = max(0, min(100, (current_rsl - 0.8) / 0.4 * 100))
            
            # 2. Calculate Mansfield RS (if benchmark available)
            mrs_signal = None
            if self._benchmark_prices is not None and len(self._benchmark_prices) > 0:
                mrs_result = self.mansfield_calc.compute_result(
                    ticker or 'UNKNOWN',
                    close,
                    self._benchmark_prices
                )
                mrs_signal = mrs_result.signal
            
            # 3. Analyze volume structure
            vol_result = self.volume_analyzer.compute_result(
                ticker or 'UNKNOWN',
                high, low, close, volume
            )
            volume_pattern = vol_result.pattern
            
            # 4. Get RRG quadrant from Capital Flow Detection
            rrg_quadrant = self._get_rrg_quadrant(sector)
            
            # 5. Compute scorecard
            scorecard_result = self.scorecard.evaluate(
                ticker=ticker or 'UNKNOWN',
                rrg_quadrant=rrg_quadrant,
                rsl_percentile=rsl_percentile,
                mrs_signal=mrs_signal,
                volume_pattern=volume_pattern,
                fundamental_momentum=None  # Reserved for future
            )
            
            # Return composite score as Series
            return pd.Series(
                [scorecard_result.total_score],
                index=[history.index[-1]]
            )
            
        except Exception as e:
            logger.warning(f"Advanced rotation computation failed for {ticker}: {e}")
            return pd.Series([0.0], index=[history.index[-1]] if len(history) > 0 else [pd.Timestamp.now()])
    
    def get_detailed_analysis(
        self,
        history: pd.DataFrame,
        ticker: str,
        sector: str,
        benchmark_prices: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Get detailed breakdown of advanced rotation analysis.
        
        Args:
            history: OHLCV DataFrame
            ticker: Stock ticker
            sector: Stock sector
            benchmark_prices: Optional benchmark prices for MRS
            
        Returns:
            Dictionary with all component scores and signals
        """
        if benchmark_prices is not None:
            self.set_benchmark_prices(benchmark_prices)
        
        try:
            # Extract columns
            close_col = 'Close' if 'Close' in history.columns else 'close'
            high_col = 'High' if 'High' in history.columns else 'high'
            low_col = 'Low' if 'Low' in history.columns else 'low'
            volume_col = 'Volume' if 'Volume' in history.columns else 'volume'
            
            close = history[close_col]
            high = history[high_col]
            low = history[low_col]
            volume = history[volume_col]
            
            # Levy RS
            rsl_result = self.levy_calc.compute_result(ticker, close)
            
            # Mansfield RS
            mrs_result = None
            if self._benchmark_prices is not None:
                mrs_result = self.mansfield_calc.compute_result(
                    ticker, close, self._benchmark_prices
                )
            
            # Volume Structure
            vol_result = self.volume_analyzer.compute_result(
                ticker, high, low, close, volume
            )
            
            # RRG Quadrant
            rrg_quadrant = self._get_rrg_quadrant(sector)
            
            # Scorecard
            rsl_percentile = max(0, min(100, (rsl_result.rsl - 0.8) / 0.4 * 100))
            scorecard_result = self.scorecard.evaluate(
                ticker=ticker,
                rrg_quadrant=rrg_quadrant,
                rsl_percentile=rsl_percentile,
                mrs_signal=mrs_result.signal if mrs_result else None,
                volume_pattern=vol_result.pattern
            )
            
            return {
                'ticker': ticker,
                'sector': sector,
                'levy_rs': rsl_result.to_dict(),
                'mansfield_rs': mrs_result.to_dict() if mrs_result else None,
                'volume_analysis': vol_result.to_dict(),
                'rrg_quadrant': rrg_quadrant,
                'scorecard': scorecard_result.to_dict(),
                'composite_score': scorecard_result.total_score,
                'signal': scorecard_result.signal,
                'confidence': scorecard_result.confidence
            }
            
        except Exception as e:
            logger.error(f"Detailed analysis failed for {ticker}: {e}")
            return {
                'ticker': ticker,
                'sector': sector,
                'error': str(e)
            }
