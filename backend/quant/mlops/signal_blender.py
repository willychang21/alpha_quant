"""ML Signal Blender Module.

Aggregates multiple ML alpha signals into a unified ranking score with:
- Cross-sectional Z-score normalization for unit-free combination
- Regime-aware dynamic weight adjustment
- Graceful degradation for missing/null signals

Example:
    >>> blender = MLSignalBlender()
    >>> scores = blender.blend(
    ...     tickers=['AAPL', 'GOOGL', 'MSFT'],
    ...     shap_attributions=shap_results,
    ...     gbm_predictions=gbm_preds,
    ...     residual_alphas=residuals,
    ...     regime_probs=regime_output,
    ...     supply_chain_scores=sc_scores
    ... )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class RegimeWeights:
    """Regime-specific weight configurations.
    
    Defines how different market regimes affect alpha weights:
    - Bull/Low Vol: Favor momentum, residual alpha
    - Bear/High Vol: Favor quality, reduce mean-reversion signals
    """
    # Base weights (regime-neutral)
    shap: float = 0.10      # SHAP attribution influence
    gbm: float = 0.30       # Constrained GBM prediction
    residual: float = 0.25  # Residual alpha
    supply_chain: float = 0.15  # Supply chain momentum
    # Remaining weight goes to existing factors
    
    # Regime adjustment multipliers
    REGIME_ADJUSTMENTS: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'Bull': {
            'shap': 1.0,
            'gbm': 1.0,
            'residual': 1.2,      # Favor mean reversion in bull
            'supply_chain': 1.1,
        },
        'Bear': {
            'shap': 1.0,
            'gbm': 1.3,           # Favor quality/trend in bear
            'residual': 0.6,      # Reduce mean reversion
            'supply_chain': 0.8,
        },
        'HighVol': {
            'shap': 0.8,
            'gbm': 1.4,           # Strong favor for constrained model
            'residual': 0.4,      # Significantly reduce mean reversion
            'supply_chain': 0.7,
        },
        'LowVol': {
            'shap': 1.1,
            'gbm': 0.9,
            'residual': 1.3,      # Mean reversion works in low vol
            'supply_chain': 1.2,
        },
        'Unknown': {
            'shap': 1.0,
            'gbm': 1.0,
            'residual': 1.0,
            'supply_chain': 1.0,
        },
    })
    
    def get_adjusted_weights(self, regime: str) -> Dict[str, float]:
        """Get regime-adjusted weights.
        
        Args:
            regime: Current market regime ('Bull', 'Bear', 'HighVol', 'LowVol', 'Unknown').
            
        Returns:
            Dict of signal name to adjusted weight.
        """
        adjustments = self.REGIME_ADJUSTMENTS.get(regime, self.REGIME_ADJUSTMENTS['Unknown'])
        
        raw_weights = {
            'shap': self.shap * adjustments['shap'],
            'gbm': self.gbm * adjustments['gbm'],
            'residual': self.residual * adjustments['residual'],
            'supply_chain': self.supply_chain * adjustments['supply_chain'],
        }
        
        # Normalize to sum to 1.0
        total = sum(raw_weights.values())
        if total > 0:
            return {k: v / total for k, v in raw_weights.items()}
        return raw_weights


class SignalNormalizer:
    """Cross-sectional signal normalization.
    
    Provides multiple normalization methods:
    - Z-score: (x - mean) / std
    - Robust Z-score: (x - median) / IQR
    - Rank percentile: rank / n
    """
    
    @staticmethod
    def zscore(series: pd.Series) -> pd.Series:
        """Standard cross-sectional Z-score normalization.
        
        Args:
            series: Raw signal values indexed by ticker.
            
        Returns:
            Z-scored values (mean=0, std=1).
        """
        if series.empty or series.std() < 1e-10:
            return pd.Series(0.0, index=series.index)
        
        return (series - series.mean()) / series.std()
    
    @staticmethod
    def robust_zscore(series: pd.Series) -> pd.Series:
        """Robust Z-score using median and IQR.
        
        More resistant to outliers than standard Z-score.
        
        Args:
            series: Raw signal values indexed by ticker.
            
        Returns:
            Robust Z-scored values.
        """
        if series.empty:
            return series
        
        median = series.median()
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        
        if iqr < 1e-10:
            return pd.Series(0.0, index=series.index)
        
        # Use IQR as scale (1.35 makes it comparable to std for normal data)
        return (series - median) / (iqr / 1.35)
    
    @staticmethod
    def rank_percentile(series: pd.Series) -> pd.Series:
        """Rank-based percentile normalization.
        
        Maps values to [0, 1] based on cross-sectional rank.
        
        Args:
            series: Raw signal values indexed by ticker.
            
        Returns:
            Percentile ranks in [0, 1].
        """
        if series.empty:
            return series
        
        return series.rank(pct=True)
    
    @staticmethod
    def winsorize(series: pd.Series, limits: Tuple[float, float] = (0.01, 0.99)) -> pd.Series:
        """Winsorize extreme values.
        
        Args:
            series: Raw signal values.
            limits: (lower_percentile, upper_percentile) for clipping.
            
        Returns:
            Winsorized series.
        """
        if series.empty:
            return series
        
        lower = series.quantile(limits[0])
        upper = series.quantile(limits[1])
        
        return series.clip(lower=lower, upper=upper)


class MLSignalBlender:
    """Blends multiple ML alpha signals into unified ranking scores.
    
    Aggregates signals from:
    - SHAP attributions (factor importance)
    - Constrained GBM predictions (quality/trend)
    - Residual alpha (mean reversion)
    - Online HMM (regime detection)
    - Supply chain momentum (spillover effects)
    
    Attributes:
        weights: RegimeWeights configuration.
        normalizer: SignalNormalizer instance.
        normalization_method: Method to use ('zscore', 'robust', 'rank').
    """
    
    def __init__(
        self,
        base_weights: Optional[RegimeWeights] = None,
        normalization_method: str = 'robust'
    ):
        """Initialize MLSignalBlender.
        
        Args:
            base_weights: Custom RegimeWeights, or None for defaults.
            normalization_method: 'zscore', 'robust', or 'rank'.
        """
        self.weights = base_weights or RegimeWeights()
        self.normalizer = SignalNormalizer()
        self.normalization_method = normalization_method
        
        # Track which signals were available in last blend
        self._last_active_signals: List[str] = []
        self._last_regime: str = 'Unknown'
        self._last_weights_used: Dict[str, float] = {}
    
    def _normalize(self, series: pd.Series) -> pd.Series:
        """Apply configured normalization method.
        
        Args:
            series: Raw signal values.
            
        Returns:
            Normalized signal values.
        """
        if self.normalization_method == 'zscore':
            return self.normalizer.zscore(series)
        elif self.normalization_method == 'robust':
            return self.normalizer.robust_zscore(series)
        elif self.normalization_method == 'rank':
            return self.normalizer.rank_percentile(series)
        else:
            logger.warning(f"Unknown normalization method: {self.normalization_method}")
            return self.normalizer.robust_zscore(series)
    
    def _extract_shap_signal(
        self,
        shap_attributions: Optional[List[Any]],
        tickers: List[str]
    ) -> pd.Series:
        """Extract composite signal from SHAP attributions.
        
        Uses sum of positive contributions as signal (higher = more bullish factors).
        
        Args:
            shap_attributions: List of SHAPAttribution objects.
            tickers: List of ticker symbols.
            
        Returns:
            Series of SHAP-derived signals indexed by ticker.
        """
        if shap_attributions is None or len(shap_attributions) == 0:
            return pd.Series(dtype=float)
        
        signals = {}
        for attr in shap_attributions:
            if hasattr(attr, 'ticker') and hasattr(attr, 'factor_contributions'):
                # Sum positive contributions (bullish factors)
                positive_sum = sum(
                    v for v in attr.factor_contributions.values() if v > 0
                )
                signals[attr.ticker] = positive_sum
        
        return pd.Series(signals)
    
    def _extract_regime(
        self,
        regime_probs: Optional[Tuple[str, np.ndarray]]
    ) -> Tuple[str, float]:
        """Extract regime label and confidence.
        
        Args:
            regime_probs: Tuple of (regime_label, probability_array) from OnlineHMM.
            
        Returns:
            Tuple of (regime_label, confidence_score).
        """
        if regime_probs is None:
            return 'Unknown', 0.5
        
        try:
            regime_label, probs = regime_probs
            # Confidence is the max probability
            confidence = float(np.max(probs)) if len(probs) > 0 else 0.5
            return regime_label, confidence
        except Exception as e:
            logger.warning(f"Failed to extract regime: {e}")
            return 'Unknown', 0.5
    
    def blend(
        self,
        tickers: List[str],
        shap_attributions: Optional[List[Any]] = None,
        gbm_predictions: Optional[pd.Series] = None,
        residual_alphas: Optional[pd.Series] = None,
        regime_probs: Optional[Tuple[str, np.ndarray]] = None,
        supply_chain_scores: Optional[pd.Series] = None,
        existing_scores: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Blend all ML signals into unified ranking scores.
        
        Args:
            tickers: List of ticker symbols to score.
            shap_attributions: List of SHAPAttribution objects.
            gbm_predictions: Series of GBM predictions indexed by ticker.
            residual_alphas: Series of residual alpha values indexed by ticker.
            regime_probs: Tuple of (regime_label, probabilities) from OnlineHMM.
            supply_chain_scores: Series of supply chain momentum indexed by ticker.
            existing_scores: Optional existing factor scores to blend with.
            
        Returns:
            DataFrame with columns: ticker, blended_score, component scores, regime.
        """
        # Get regime and adjust weights
        regime, confidence = self._extract_regime(regime_probs)
        adjusted_weights = self.weights.get_adjusted_weights(regime)
        
        self._last_regime = regime
        logger.info(f"Blending signals with regime={regime} (confidence={confidence:.2f})")
        
        # Initialize result DataFrame
        result = pd.DataFrame({'ticker': tickers})
        result.set_index('ticker', inplace=True)
        
        # Collect available signals
        available_signals: Dict[str, pd.Series] = {}
        
        # 1. SHAP signal
        shap_signal = self._extract_shap_signal(shap_attributions, tickers)
        if not shap_signal.empty:
            available_signals['shap'] = shap_signal
            result['shap_raw'] = shap_signal.reindex(tickers)
        
        # 2. GBM predictions
        if gbm_predictions is not None and not gbm_predictions.empty:
            available_signals['gbm'] = gbm_predictions
            result['gbm_raw'] = gbm_predictions.reindex(tickers)
        
        # 3. Residual alpha
        if residual_alphas is not None and not residual_alphas.empty:
            available_signals['residual'] = residual_alphas
            result['residual_raw'] = residual_alphas.reindex(tickers)
        
        # 4. Supply chain momentum (graceful degradation)
        if supply_chain_scores is not None and not supply_chain_scores.empty:
            # Check for sufficient coverage
            coverage = supply_chain_scores.reindex(tickers).notna().mean()
            if coverage > 0.1:  # At least 10% tickers have data
                available_signals['supply_chain'] = supply_chain_scores
                result['supply_chain_raw'] = supply_chain_scores.reindex(tickers)
            else:
                logger.info(f"Supply chain coverage too low ({coverage:.1%}), excluding")
        
        self._last_active_signals = list(available_signals.keys())
        logger.info(f"Active signals: {self._last_active_signals}")
        
        # Normalize all available signals
        normalized: Dict[str, pd.Series] = {}
        for name, signal in available_signals.items():
            # Align to tickers
            aligned = signal.reindex(tickers)
            # Winsorize then normalize
            winsorized = self.normalizer.winsorize(aligned.fillna(0))
            normalized[name] = self._normalize(winsorized)
            result[f'{name}_norm'] = normalized[name]
        
        # Re-normalize weights for available signals only
        active_weights = {
            k: v for k, v in adjusted_weights.items() 
            if k in available_signals
        }
        weight_sum = sum(active_weights.values())
        if weight_sum > 0:
            active_weights = {k: v / weight_sum for k, v in active_weights.items()}
        
        self._last_weights_used = active_weights
        logger.info(f"Weights used: {active_weights}")
        
        # Compute blended score
        blended = pd.Series(0.0, index=tickers)
        for name, norm_signal in normalized.items():
            weight = active_weights.get(name, 0.0)
            blended += weight * norm_signal.fillna(0)
        
        # Optionally blend with existing scores
        if existing_scores is not None and not existing_scores.empty:
            existing_aligned = existing_scores.reindex(tickers).fillna(0)
            existing_norm = self._normalize(existing_aligned)
            # Give ML signals 40% weight, existing 60%
            blended = 0.4 * blended + 0.6 * existing_norm
        
        result['blended_score'] = blended
        result['regime'] = regime
        result['regime_confidence'] = confidence
        
        # Add rank
        result['rank'] = result['blended_score'].rank(ascending=False, method='first')
        
        # Reset index for output
        result = result.reset_index()
        
        return result
    
    def get_diagnostic_info(self) -> Dict[str, Any]:
        """Get diagnostic information from last blend.
        
        Returns:
            Dict with active signals, regime, and weights used.
        """
        return {
            'active_signals': self._last_active_signals,
            'regime': self._last_regime,
            'weights_used': self._last_weights_used,
        }
    
    def blend_simple(
        self,
        tickers: List[str],
        signal_dict: Dict[str, pd.Series],
        regime: str = 'Unknown'
    ) -> pd.Series:
        """Simplified blend API for direct signal dict input.
        
        Args:
            tickers: List of ticker symbols.
            signal_dict: Dict mapping signal name to Series.
            regime: Current market regime.
            
        Returns:
            Series of blended scores indexed by ticker.
        """
        adjusted_weights = self.weights.get_adjusted_weights(regime)
        
        # Normalize and weight
        blended = pd.Series(0.0, index=tickers)
        active_weights = {}
        
        for name, signal in signal_dict.items():
            if signal is not None and not signal.empty:
                aligned = signal.reindex(tickers).fillna(0)
                normalized = self._normalize(aligned)
                weight = adjusted_weights.get(name, 0.1)
                blended += weight * normalized
                active_weights[name] = weight
        
        # Re-normalize
        if sum(active_weights.values()) > 0:
            blended = blended / sum(active_weights.values())
        
        return blended


# Convenience function for RankingEngine integration
def create_ml_signal_blender(
    normalization: str = 'robust',
    **weight_overrides
) -> MLSignalBlender:
    """Create MLSignalBlender with optional weight overrides.
    
    Args:
        normalization: Normalization method ('zscore', 'robust', 'rank').
        **weight_overrides: Custom base weights (shap, gbm, residual, supply_chain).
        
    Returns:
        Configured MLSignalBlender instance.
    """
    weights = RegimeWeights(
        shap=weight_overrides.get('shap', 0.10),
        gbm=weight_overrides.get('gbm', 0.30),
        residual=weight_overrides.get('residual', 0.25),
        supply_chain=weight_overrides.get('supply_chain', 0.15),
    )
    
    return MLSignalBlender(base_weights=weights, normalization_method=normalization)
