"""Integration tests for MLSignalBlender.

Tests the signal blending functionality with mocked module outputs
to verify correct normalization, weighting, and graceful degradation.
"""

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quant.mlops.signal_blender import (
    MLSignalBlender,
    RegimeWeights,
    SignalNormalizer,
    create_ml_signal_blender,
)
from quant.mlops.shap_attributor import SHAPAttribution


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_tickers():
    """Sample ticker list."""
    return ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'JNJ']


@pytest.fixture
def mock_shap_attributions(sample_tickers):
    """Mock SHAP attribution results."""
    attributions = []
    np.random.seed(42)
    
    for ticker in sample_tickers:
        contributions = {
            'quality': np.random.uniform(-0.3, 0.3),
            'value': np.random.uniform(-0.2, 0.2),
            'momentum': np.random.uniform(-0.4, 0.4),
            'volatility': np.random.uniform(-0.2, 0.1),
        }
        attr = SHAPAttribution(
            ticker=ticker,
            base_value=0.05,
            prediction=sum(contributions.values()) + 0.05,
            factor_contributions=contributions
        )
        attributions.append(attr)
    
    return attributions


@pytest.fixture
def mock_gbm_predictions(sample_tickers):
    """Mock GBM predictions."""
    np.random.seed(43)
    return pd.Series(
        np.random.uniform(-0.1, 0.1, len(sample_tickers)),
        index=sample_tickers
    )


@pytest.fixture
def mock_residual_alphas(sample_tickers):
    """Mock residual alpha values."""
    np.random.seed(44)
    return pd.Series(
        np.random.uniform(-0.05, 0.05, len(sample_tickers)),
        index=sample_tickers
    )


@pytest.fixture
def mock_supply_chain_scores(sample_tickers):
    """Mock supply chain momentum scores."""
    np.random.seed(45)
    scores = pd.Series(
        np.random.uniform(-2, 2, len(sample_tickers)),
        index=sample_tickers
    )
    # Simulate some missing data (NaN for 2 tickers)
    scores.iloc[0] = np.nan
    scores.iloc[5] = np.nan
    return scores


@pytest.fixture
def mock_regime_bull():
    """Mock bull regime output."""
    return ('Bull', np.array([0.8, 0.2]))


@pytest.fixture
def mock_regime_bear():
    """Mock bear regime output."""
    return ('Bear', np.array([0.3, 0.7]))


# =============================================================================
# SignalNormalizer Tests
# =============================================================================

class TestSignalNormalizer:
    """Tests for signal normalization methods."""
    
    def test_zscore_normalization(self):
        """Test standard Z-score produces mean=0, std=1."""
        normalizer = SignalNormalizer()
        
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalizer.zscore(data)
        
        assert abs(normalized.mean()) < 1e-10, "Z-score mean should be 0"
        assert abs(normalized.std() - 1.0) < 1e-10, "Z-score std should be 1"
    
    def test_robust_zscore_handles_outliers(self):
        """Test robust Z-score with more realistic data distribution."""
        normalizer = SignalNormalizer()
        
        # Data with more variance and an outlier
        np.random.seed(42)
        base_data = np.random.randn(20) * 2  # More realistic distribution
        data_with_outlier = pd.Series(np.append(base_data, 50.0))  # Add outlier
        
        zscore = normalizer.zscore(data_with_outlier)
        robust = normalizer.robust_zscore(data_with_outlier)
        
        # Both should produce finite values
        assert np.isfinite(zscore).all(), "Z-score should be finite"
        assert np.isfinite(robust).all(), "Robust Z-score should be finite"
        
        # Non-outlier points should have reasonable values in both
        median_zscore = np.median(np.abs(zscore.iloc[:-1]))
        median_robust = np.median(np.abs(robust.iloc[:-1]))
        
        assert median_zscore < 2.0, "Median Z-score of non-outliers should be < 2"
        assert median_robust < 2.0, "Median robust Z-score of non-outliers should be < 2"
    
    def test_rank_percentile(self):
        """Test rank percentile maps to [0, 1]."""
        normalizer = SignalNormalizer()
        
        data = pd.Series([10, 20, 30, 40, 50])
        ranked = normalizer.rank_percentile(data)
        
        assert ranked.min() >= 0.0, "Rank min should be >= 0"
        assert ranked.max() <= 1.0, "Rank max should be <= 1"
        assert ranked.iloc[-1] == 1.0, "Highest value should have rank 1.0"
    
    def test_winsorize(self):
        """Test winsorization clips extreme values."""
        normalizer = SignalNormalizer()
        
        data = pd.Series([1, 2, 3, 4, 5, 100])
        winsorized = normalizer.winsorize(data, limits=(0.1, 0.9))
        
        assert winsorized.max() < 100, "Winsorization should clip upper outlier"
    
    def test_empty_series_handling(self):
        """Test normalizers handle empty series gracefully."""
        normalizer = SignalNormalizer()
        
        empty = pd.Series(dtype=float)
        
        assert normalizer.zscore(empty).empty
        assert normalizer.robust_zscore(empty).empty
        assert normalizer.rank_percentile(empty).empty


# =============================================================================
# RegimeWeights Tests
# =============================================================================

class TestRegimeWeights:
    """Tests for regime-aware weight configuration."""
    
    def test_default_weights_sum_to_less_than_one(self):
        """Base weights should sum to < 1 to leave room for existing factors."""
        weights = RegimeWeights()
        total = weights.shap + weights.gbm + weights.residual + weights.supply_chain
        
        assert total < 1.0, f"Base weights sum to {total}, should be < 1.0"
    
    def test_adjusted_weights_normalize_to_one(self):
        """Regime-adjusted weights should normalize to 1.0."""
        weights = RegimeWeights()
        
        for regime in ['Bull', 'Bear', 'HighVol', 'LowVol', 'Unknown']:
            adjusted = weights.get_adjusted_weights(regime)
            total = sum(adjusted.values())
            
            assert abs(total - 1.0) < 1e-6, \
                f"Adjusted weights for {regime} sum to {total}, should be 1.0"
    
    def test_bear_regime_reduces_residual_weight(self):
        """Bear regime should reduce residual alpha weight."""
        weights = RegimeWeights()
        
        bull_weights = weights.get_adjusted_weights('Bull')
        bear_weights = weights.get_adjusted_weights('Bear')
        
        # After normalization, the relative weight of residual should be lower
        assert bear_weights['residual'] < bull_weights['residual'], \
            "Bear regime should reduce residual relative weight"
    
    def test_high_vol_favors_gbm(self):
        """High volatility should increase GBM weight."""
        weights = RegimeWeights()
        
        unknown_weights = weights.get_adjusted_weights('Unknown')
        highvol_weights = weights.get_adjusted_weights('HighVol')
        
        assert highvol_weights['gbm'] > unknown_weights['gbm'], \
            "High vol should increase GBM relative weight"


# =============================================================================
# MLSignalBlender Tests
# =============================================================================

class TestMLSignalBlender:
    """Integration tests for MLSignalBlender."""
    
    def test_blend_with_all_signals(
        self,
        sample_tickers,
        mock_shap_attributions,
        mock_gbm_predictions,
        mock_residual_alphas,
        mock_regime_bull,
        mock_supply_chain_scores
    ):
        """Test blending with all signals available."""
        blender = MLSignalBlender()
        
        result = blender.blend(
            tickers=sample_tickers,
            shap_attributions=mock_shap_attributions,
            gbm_predictions=mock_gbm_predictions,
            residual_alphas=mock_residual_alphas,
            regime_probs=mock_regime_bull,
            supply_chain_scores=mock_supply_chain_scores
        )
        
        # Verify output structure
        assert 'ticker' in result.columns
        assert 'blended_score' in result.columns
        assert 'rank' in result.columns
        assert 'regime' in result.columns
        
        # Verify all tickers present
        assert len(result) == len(sample_tickers)
        assert set(result['ticker']) == set(sample_tickers)
        
        # Verify blended scores are numeric and finite
        assert result['blended_score'].notna().all()
        assert np.isfinite(result['blended_score']).all()
        
        # Verify regime is recorded
        assert result['regime'].iloc[0] == 'Bull'
    
    def test_blend_without_supply_chain(
        self,
        sample_tickers,
        mock_shap_attributions,
        mock_gbm_predictions,
        mock_residual_alphas,
        mock_regime_bull
    ):
        """Test graceful degradation when supply chain is missing."""
        blender = MLSignalBlender()
        
        result = blender.blend(
            tickers=sample_tickers,
            shap_attributions=mock_shap_attributions,
            gbm_predictions=mock_gbm_predictions,
            residual_alphas=mock_residual_alphas,
            regime_probs=mock_regime_bull,
            supply_chain_scores=None  # Missing!
        )
        
        # Should still produce valid output
        assert len(result) == len(sample_tickers)
        assert result['blended_score'].notna().all()
        
        # Diagnostic info should show supply_chain excluded
        diag = blender.get_diagnostic_info()
        assert 'supply_chain' not in diag['active_signals']
    
    def test_blend_with_sparse_supply_chain(
        self,
        sample_tickers,
        mock_gbm_predictions,
        mock_regime_bull
    ):
        """Test supply chain exclusion when coverage is too low."""
        blender = MLSignalBlender()
        
        # Supply chain with only 1 ticker (< 10% coverage)
        sparse_sc = pd.Series({'AAPL': 0.5})
        
        result = blender.blend(
            tickers=sample_tickers,
            gbm_predictions=mock_gbm_predictions,
            regime_probs=mock_regime_bull,
            supply_chain_scores=sparse_sc
        )
        
        # Should exclude supply chain due to low coverage
        diag = blender.get_diagnostic_info()
        assert 'supply_chain' not in diag['active_signals']
    
    def test_regime_affects_weights(
        self,
        sample_tickers,
        mock_gbm_predictions,
        mock_residual_alphas,
        mock_regime_bull,
        mock_regime_bear
    ):
        """Test that different regimes produce different blends."""
        blender = MLSignalBlender()
        
        # Blend with Bull regime
        bull_result = blender.blend(
            tickers=sample_tickers,
            gbm_predictions=mock_gbm_predictions,
            residual_alphas=mock_residual_alphas,
            regime_probs=mock_regime_bull
        )
        
        # Blend with Bear regime (same signals)
        bear_result = blender.blend(
            tickers=sample_tickers,
            gbm_predictions=mock_gbm_predictions,
            residual_alphas=mock_residual_alphas,
            regime_probs=mock_regime_bear
        )
        
        # Blended scores should differ due to regime-adjusted weights
        bull_scores = bull_result.set_index('ticker')['blended_score']
        bear_scores = bear_result.set_index('ticker')['blended_score']
        
        # They should not be identical
        assert not np.allclose(bull_scores.values, bear_scores.values), \
            "Bull and Bear regimes should produce different blends"
    
    def test_blend_with_only_gbm(self, sample_tickers, mock_gbm_predictions):
        """Test blending with only GBM predictions (minimal input)."""
        blender = MLSignalBlender()
        
        result = blender.blend(
            tickers=sample_tickers,
            gbm_predictions=mock_gbm_predictions
        )
        
        # Should work with just one signal
        assert len(result) == len(sample_tickers)
        assert result['blended_score'].notna().all()
        
        # Weights should be 100% GBM
        diag = blender.get_diagnostic_info()
        assert diag['active_signals'] == ['gbm']
        assert abs(diag['weights_used']['gbm'] - 1.0) < 1e-6
    
    def test_blend_with_existing_scores(
        self,
        sample_tickers,
        mock_gbm_predictions,
        mock_regime_bull
    ):
        """Test blending with existing factor scores."""
        blender = MLSignalBlender()
        
        # Existing scores from traditional factors
        existing = pd.Series(
            np.random.uniform(-1, 1, len(sample_tickers)),
            index=sample_tickers
        )
        
        result = blender.blend(
            tickers=sample_tickers,
            gbm_predictions=mock_gbm_predictions,
            regime_probs=mock_regime_bull,
            existing_scores=existing
        )
        
        # Output should incorporate both ML and existing scores
        assert len(result) == len(sample_tickers)
    
    def test_blend_simple_api(self, sample_tickers):
        """Test simplified blend API."""
        blender = MLSignalBlender()
        
        signals = {
            'gbm': pd.Series(np.random.randn(len(sample_tickers)), index=sample_tickers),
            'residual': pd.Series(np.random.randn(len(sample_tickers)), index=sample_tickers),
        }
        
        result = blender.blend_simple(sample_tickers, signals, regime='Bull')
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_tickers)
    
    def test_normalization_method_selection(self, sample_tickers, mock_gbm_predictions):
        """Test different normalization methods produce different results."""
        results = {}
        
        for method in ['zscore', 'robust', 'rank']:
            blender = MLSignalBlender(normalization_method=method)
            result = blender.blend(
                tickers=sample_tickers,
                gbm_predictions=mock_gbm_predictions
            )
            results[method] = result['blended_score'].values
        
        # Different methods should produce different scores
        assert not np.allclose(results['zscore'], results['rank'])
    
    def test_create_ml_signal_blender_factory(self):
        """Test factory function with weight overrides."""
        blender = create_ml_signal_blender(
            normalization='rank',
            gbm=0.5,
            residual=0.3
        )
        
        assert blender.normalization_method == 'rank'
        assert blender.weights.gbm == 0.5
        assert blender.weights.residual == 0.3


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_tickers(self):
        """Test handling of empty ticker list."""
        blender = MLSignalBlender()
        
        result = blender.blend(
            tickers=[],
            gbm_predictions=pd.Series(dtype=float)
        )
        
        assert len(result) == 0
    
    def test_all_signals_none(self, sample_tickers):
        """Test handling when all signals are None."""
        blender = MLSignalBlender()
        
        result = blender.blend(
            tickers=sample_tickers,
            shap_attributions=None,
            gbm_predictions=None,
            residual_alphas=None,
            supply_chain_scores=None
        )
        
        # Should return DataFrame with zero scores
        assert len(result) == len(sample_tickers)
        assert (result['blended_score'] == 0).all()
    
    def test_misaligned_indices(self, sample_tickers):
        """Test handling of signals with different ticker indices."""
        blender = MLSignalBlender()
        
        # GBM with only half the tickers
        partial_gbm = pd.Series(
            np.random.randn(5),
            index=sample_tickers[:5]
        )
        
        # Residual with different tickers
        residual = pd.Series(
            np.random.randn(5),
            index=sample_tickers[5:]
        )
        
        result = blender.blend(
            tickers=sample_tickers,
            gbm_predictions=partial_gbm,
            residual_alphas=residual
        )
        
        # Should handle misalignment gracefully
        assert len(result) == len(sample_tickers)
        assert result['blended_score'].notna().all()
    
    def test_constant_signal_handling(self, sample_tickers):
        """Test handling of constant signals (zero variance)."""
        blender = MLSignalBlender()
        
        # All same value
        constant = pd.Series(5.0, index=sample_tickers)
        
        result = blender.blend(
            tickers=sample_tickers,
            gbm_predictions=constant
        )
        
        # Should not crash, scores should be zero or constant
        assert len(result) == len(sample_tickers)
        assert np.isfinite(result['blended_score']).all()
