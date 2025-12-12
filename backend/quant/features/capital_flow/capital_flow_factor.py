"""
Capital Flow Factor Module.

Integrates sector rotation and money flow signals into the RankingEngine
factor pipeline as a FeatureGenerator.

Combines:
- Sector Flow Score: Based on sector's RRG quadrant
- Money Flow Score: Based on MFI, OBV, and divergence signals
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

from quant.features.base import FeatureGenerator
from quant.features.capital_flow.models import MoneyFlowResult
from quant.features.capital_flow.money_flow import MoneyFlowCalculator
from quant.features.capital_flow.divergence import DivergenceDetector
from quant.features.capital_flow.sector_rotation import SectorRotationAnalyzer

logger = logging.getLogger(__name__)


class CapitalFlowFactor(FeatureGenerator):
    """Capital Flow Factor combining sector rotation and money flow signals.
    
    Integrates with existing RankingEngine factor pipeline.
    
    Components:
    - sector_flow_score: Based on sector's RRG quadrant position
    - money_flow_score: Based on MFI, OBV, and divergence signals
    """
    
    def __init__(
        self,
        sector_weight: float = 0.4,
        money_flow_weight: float = 0.6,
        mfi_period: int = 14,
        divergence_lookback: int = 20
    ):
        """Initialize CapitalFlowFactor.
        
        Args:
            sector_weight: Weight for sector rotation score (default: 0.4)
            money_flow_weight: Weight for money flow score (default: 0.6)
            mfi_period: MFI calculation period
            divergence_lookback: Divergence detection lookback
        """
        self.sector_weight = sector_weight
        self.money_flow_weight = money_flow_weight
        
        # Initialize components
        self.money_flow_calc = MoneyFlowCalculator(mfi_period=mfi_period)
        self.divergence_detector = DivergenceDetector(lookback=divergence_lookback)
        self.sector_analyzer = SectorRotationAnalyzer()
        
        # Cache for sector rotation data
        self._sector_cache: Dict[str, Any] = {}
        self._cache_timestamp: Optional[pd.Timestamp] = None
    
    @property
    def name(self) -> str:
        """Unique name of the factor."""
        return "CapitalFlow"
    
    @property
    def description(self) -> str:
        """Description of what this factor measures."""
        return "Composite capital flow score from sector rotation and money flow indicators."
    
    def refresh_sector_data(self, force: bool = False) -> None:
        """Refresh sector rotation data cache.
        
        Args:
            force: Force refresh even if cache is recent
        """
        import datetime
        
        now = pd.Timestamp.now()
        
        # Refresh if cache is empty, forced, or older than 1 hour
        should_refresh = (
            force or
            self._cache_timestamp is None or
            (now - self._cache_timestamp).total_seconds() > 3600
        )
        
        if should_refresh:
            logger.info("Refreshing sector rotation data...")
            results = self.sector_analyzer.analyze_all_sectors()
            self._sector_cache = {
                symbol: result.to_dict() for symbol, result in results.items()
            }
            self._cache_timestamp = now
    
    def get_sector_flow_score(self, sector: str) -> float:
        """Get sector rotation score based on RRG quadrant.
        
        Args:
            sector: Sector name (e.g., 'Technology')
            
        Returns:
            Sector flow score: positive for favorable, negative for unfavorable
        """
        if not sector or sector == 'Unknown':
            return 0.0
        
        # Map sector name to ETF symbol
        sector_symbol = self.sector_analyzer.SECTOR_TO_ETF.get(sector)
        
        if not sector_symbol:
            # Try to find partial match
            for name, symbol in self.sector_analyzer.SECTOR_TO_ETF.items():
                if name.lower() in sector.lower() or sector.lower() in name.lower():
                    sector_symbol = symbol
                    break
        
        if not sector_symbol:
            return 0.0
        
        # Get quadrant from cache
        cached_result = self._sector_cache.get(sector_symbol)
        
        if cached_result:
            quadrant = cached_result.get('quadrant', 'Unknown')
            return self.sector_analyzer.get_quadrant_score(quadrant)
        
        # Fallback: get from analyzer state
        quadrant = self.sector_analyzer.get_sector_quadrant(sector)
        return self.sector_analyzer.get_quadrant_score(quadrant)
    
    def get_money_flow_score(self, history: pd.DataFrame) -> float:
        """Get money flow score from MFI, OBV, and divergence signals.
        
        Args:
            history: OHLCV DataFrame with columns: Open, High, Low, Close, Volume
            
        Returns:
            Money flow score (normalized)
        """
        if history.empty or len(history) < 20:
            return 0.0
        
        try:
            # Handle column name variations
            close_col = 'Close' if 'Close' in history.columns else 'close'
            high_col = 'High' if 'High' in history.columns else 'high'
            low_col = 'Low' if 'Low' in history.columns else 'low'
            volume_col = 'Volume' if 'Volume' in history.columns else 'volume'
            
            # Extract series
            close = history[close_col]
            high = history[high_col]
            low = history[low_col]
            volume = history[volume_col]
            
            # Calculate MFI
            mfi = self.money_flow_calc.calculate_mfi(high, low, close, volume)
            current_mfi = float(mfi.iloc[-1])
            mfi_signal = self.money_flow_calc.classify_mfi(current_mfi)
            
            # Calculate OBV
            obv = self.money_flow_calc.calculate_obv(close, volume)
            obv_zscore = self.money_flow_calc.normalize_obv(obv)
            current_obv_zscore = float(obv_zscore.iloc[-1])
            obv_trend = self.money_flow_calc.classify_obv_trend(obv)
            
            # Calculate divergence score
            divergence_score = self.divergence_detector.calculate_divergence_score(
                close, mfi, obv
            )
            
            # Combine into composite score
            # MFI contribution: oversold = positive, overbought = negative
            mfi_contribution = 0.0
            if mfi_signal == 'oversold':
                mfi_contribution = 0.5  # Bullish signal
            elif mfi_signal == 'overbought':
                mfi_contribution = -0.3  # Mild bearish
            else:
                # Linear scale: 50 is neutral
                mfi_contribution = (50 - current_mfi) / 100  # -0.3 to +0.5
            
            # OBV contribution: accumulation = positive
            obv_contribution = 0.0
            if obv_trend == 'accumulation':
                obv_contribution = 0.3
            elif obv_trend == 'distribution':
                obv_contribution = -0.3
            
            # Combine with weights
            composite = (
                0.25 * mfi_contribution +
                0.25 * obv_contribution +
                0.50 * divergence_score  # Divergence is the key signal
            )
            
            # Clamp to reasonable range
            return max(-1.0, min(1.0, composite))
            
        except Exception as e:
            logger.debug(f"Money flow score calculation failed: {e}")
            return 0.0
    
    def compute(
        self,
        history: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
        ticker: str = None,
        sector: str = None
    ) -> pd.Series:
        """Compute capital flow score for a stock.
        
        Args:
            history: OHLCV DataFrame
            fundamentals: Optional fundamental data (unused)
            ticker: Stock ticker symbol
            sector: Stock sector name
            
        Returns:
            Series with composite capital flow score
        """
        if history.empty:
            return pd.Series([0.0], index=[0])
        
        try:
            # Ensure sector data is fresh (lazy refresh)
            if not self._sector_cache:
                self.refresh_sector_data()
            
            # Get component scores
            sector_score = self.get_sector_flow_score(sector) if sector else 0.0
            money_flow_score = self.get_money_flow_score(history)
            
            # Compute weighted composite
            composite_score = (
                self.sector_weight * sector_score +
                self.money_flow_weight * money_flow_score
            )
            
            # Return as Series (for compatibility with other factors)
            result = pd.Series([composite_score], index=[history.index[-1]])
            
            return result
            
        except Exception as e:
            logger.debug(f"Capital flow score computation failed: {e}")
            return pd.Series([0.0], index=[history.index[-1]] if len(history) > 0 else [0])
    
    def get_detailed_analysis(
        self,
        history: pd.DataFrame,
        ticker: str,
        sector: str
    ) -> Dict[str, Any]:
        """Get detailed capital flow analysis for a stock.
        
        Args:
            history: OHLCV DataFrame
            ticker: Stock ticker
            sector: Stock sector
            
        Returns:
            Dictionary with detailed breakdown
        """
        try:
            # Handle column name variations
            close_col = 'Close' if 'Close' in history.columns else 'close'
            high_col = 'High' if 'High' in history.columns else 'high'
            low_col = 'Low' if 'Low' in history.columns else 'low'
            volume_col = 'Volume' if 'Volume' in history.columns else 'volume'
            
            close = history[close_col]
            high = history[high_col]
            low = history[low_col]
            volume = history[volume_col]
            
            # Calculate indicators
            mfi = self.money_flow_calc.calculate_mfi(high, low, close, volume)
            obv = self.money_flow_calc.calculate_obv(close, volume)
            obv_zscore = self.money_flow_calc.normalize_obv(obv)
            divergence_score = self.divergence_detector.calculate_divergence_score(
                close, mfi, obv
            )
            
            current_mfi = float(mfi.iloc[-1])
            current_obv_zscore = float(obv_zscore.iloc[-1])
            
            # Get sector info
            sector_symbol = self.sector_analyzer.SECTOR_TO_ETF.get(sector, None)
            sector_data = self._sector_cache.get(sector_symbol, {})
            
            return {
                'ticker': ticker,
                'sector': sector,
                'sector_symbol': sector_symbol,
                'sector_quadrant': sector_data.get('quadrant', 'Unknown'),
                'sector_rs_ratio': sector_data.get('rs_ratio', 100.0),
                'sector_rs_momentum': sector_data.get('rs_momentum', 0.0),
                'sector_flow_score': self.get_sector_flow_score(sector),
                'mfi': current_mfi,
                'mfi_signal': self.money_flow_calc.classify_mfi(current_mfi),
                'obv_zscore': current_obv_zscore,
                'obv_trend': self.money_flow_calc.classify_obv_trend(obv),
                'divergence_score': divergence_score,
                'money_flow_score': self.get_money_flow_score(history),
                'composite_score': float(self.compute(history, None, ticker, sector).iloc[-1])
            }
            
        except Exception as e:
            logger.error(f"Detailed analysis failed for {ticker}: {e}")
            return {
                'ticker': ticker,
                'sector': sector,
                'error': str(e)
            }
