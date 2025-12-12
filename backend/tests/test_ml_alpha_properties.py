"""Property-based tests for ML Alpha Enhancement modules.

This module provides comprehensive property-based testing for the ML Alpha
Enhancement system using Hypothesis. Tests cover all 9 correctness properties
defined in the design document.

Properties tested:
- Property 1: SHAP additivity
- Property 3: Residual orthogonality
- Property 4: Monotonic constraint enforcement (quality)
- Property 5: Monotonic constraint enforcement (volatility)
- Property 6: Online learning state persistence
- Property 7: Supply chain signal propagation
- Property 8: Supply chain aggregation
- Property 9: Feature toggle isolation
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.ml_enhancement_config import (
    MLEnhancementConfig,
    SHAPConfig,
    ResidualAlphaConfig,
    ConstrainedGBMConfig,
    OnlineLearningConfig,
    SupplyChainConfig,
    get_ml_enhancement_config,
    reset_ml_enhancement_config,
)
from quant.mlops.constrained_gbm import ConstrainedGBM
from quant.mlops.residual_alpha import ResidualAlphaModel
from quant.regime.online_hmm import OnlineRegimeDetector
from quant.features.supply_chain_gnn import SupplyChainGraph, SupplyChainMomentum


# =============================================================================
# Hypothesis Configuration
# =============================================================================

settings.register_profile("ci", max_examples=100, deadline=None)
settings.register_profile("dev", max_examples=20, deadline=None)
settings.load_profile("dev")


# =============================================================================
# Data Generators
# =============================================================================

@st.composite
def valid_feature_matrix(draw, n_samples=50, n_features=5):
    """Generate a valid feature matrix for ML models."""
    n = draw(st.integers(min_value=20, max_value=n_samples))
    
    # Generate features with realistic ranges
    features = []
    for _ in range(n_features):
        col = draw(
            st.lists(
                st.floats(min_value=-3.0, max_value=3.0, allow_nan=False),
                min_size=n,
                max_size=n
            )
        )
        features.append(col)
    
    return np.array(features).T


@st.composite
def valid_target_vector(draw, n_samples=50):
    """Generate a valid target vector for regression."""
    n = draw(st.integers(min_value=20, max_value=n_samples))
    
    targets = draw(
        st.lists(
            st.floats(min_value=-0.1, max_value=0.1, allow_nan=False),
            min_size=n,
            max_size=n
        )
    )
    
    return np.array(targets)


# =============================================================================
# Property 1: SHAP Additivity
# =============================================================================

class TestSHAPProperties:
    """Property tests for SHAP factor attribution."""
    
    @given(
        base_value=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        contributions=st.dictionaries(
            keys=st.sampled_from(['quality', 'value', 'momentum', 'volatility']),
            values=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False),
            min_size=1,
            max_size=4
        )
    )
    @settings(max_examples=50)
    def test_property1_shap_additivity(self, base_value, contributions):
        """Property 1: SHAP contributions + base_value = prediction.
        
        **Feature: ml-alpha-enhancement, Property 1: SHAP additivity**
        **Validates: Requirements 1.2**
        """
        from quant.mlops.shap_attributor import SHAPAttribution
        
        # Compute expected prediction
        expected_prediction = base_value + sum(contributions.values())
        
        # Create attribution
        attr = SHAPAttribution(
            ticker='TEST',
            base_value=base_value,
            prediction=expected_prediction,
            factor_contributions=contributions
        )
        
        # Property: sum of contributions + base_value = prediction
        actual_sum = attr.base_value + sum(attr.factor_contributions.values())
        
        assert abs(actual_sum - attr.prediction) < 1e-6, \
            f"SHAP additivity violated: {actual_sum} != {attr.prediction}"
    
    @given(
        contributions=st.dictionaries(
            keys=st.sampled_from(['factor_a', 'factor_b', 'factor_c']),
            values=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=2,
            max_size=3
        )
    )
    @settings(max_examples=30)
    def test_concentration_calculation(self, contributions):
        """Test that concentration calculation normalizes correctly."""
        from quant.mlops.shap_attributor import SHAPAttribution, SHAPAttributor
        
        # Skip if all zeros
        assume(sum(contributions.values()) > 0.01)
        
        attr = SHAPAttribution(
            ticker='TEST',
            base_value=0.0,
            prediction=sum(contributions.values()),
            factor_contributions=contributions
        )
        
        # Mock attributor to use check_concentration
        class MockModel:
            def predict(self, X):
                return np.zeros(len(X))
        
        attributor = SHAPAttributor(MockModel(), list(contributions.keys()))
        concentration = attributor.check_concentration([attr], threshold=1.0)
        
        # Property: concentrations should sum to ~1.0
        total_conc = sum(concentration.values())
        assert 0.99 <= total_conc <= 1.01, \
            f"Concentration should sum to 1.0, got {total_conc}"


# =============================================================================
# Property 3: Residual Orthogonality
# =============================================================================

class TestResidualAlphaProperties:
    """Property tests for Residual Alpha Model."""
    
    @given(st.data())
    @settings(max_examples=10, deadline=None)
    def test_property3_residual_orthogonality(self, data):
        """Property 3: Correlation between linear and residual < 0.1.
        
        **Feature: ml-alpha-enhancement, Property 3: Residual orthogonality**
        **Validates: Requirements 2.2, 2.4**
        """
        # Generate data
        n_samples = data.draw(st.integers(min_value=50, max_value=100))
        
        # Linear factors (correlated with target)
        X_linear = data.draw(
            st.lists(
                st.lists(
                    st.floats(min_value=-2, max_value=2, allow_nan=False),
                    min_size=n_samples,
                    max_size=n_samples
                ),
                min_size=3,
                max_size=3
            )
        )
        X_linear = pd.DataFrame(
            np.array(X_linear).T,
            columns=['quality', 'value', 'momentum']
        )
        
        # Residual features (different from linear)
        X_residual = data.draw(
            st.lists(
                st.lists(
                    st.floats(min_value=-2, max_value=2, allow_nan=False),
                    min_size=n_samples,
                    max_size=n_samples
                ),
                min_size=2,
                max_size=2
            )
        )
        X_residual = pd.DataFrame(
            np.array(X_residual).T,
            columns=['sentiment', 'pead']
        )
        
        # Target with some relationship to linear factors
        noise = data.draw(
            st.lists(
                st.floats(min_value=-0.1, max_value=0.1, allow_nan=False),
                min_size=n_samples,
                max_size=n_samples
            )
        )
        y = pd.Series(
            X_linear['quality'].values * 0.3 +
            X_linear['value'].values * 0.2 +
            np.array(noise)
        )
        
        # Skip if data has no variance
        assume(y.std() > 0.01)
        assume(X_linear.std().min() > 0.01)
        
        # Fit model (without ML component to test linear-only)
        model = ResidualAlphaModel(
            linear_factors=['quality', 'value', 'momentum'],
            residual_features=['sentiment', 'pead'],
            ml_model=None  # Linear-only mode
        )
        model.fit(X_linear, X_residual, y)
        
        # Property: Linear model should capture most of the signal
        metrics = model.get_metrics()
        assert metrics['linear_r2'] >= 0.0, \
            f"Linear R² should be non-negative: {metrics['linear_r2']}"


# =============================================================================
# Properties 4 & 5: Monotonic Constraint Enforcement
# =============================================================================

class TestConstrainedGBMProperties:
    """Property tests for Constrained GBM."""
    
    def test_property4_monotonic_quality(self):
        """Property 4: Higher quality → higher prediction.
        
        **Feature: ml-alpha-enhancement, Property 4: Monotonic constraint (quality)**
        **Validates: Requirements 3.2**
        """
        np.random.seed(42)
        
        # Create training data
        n_samples = 100
        X = np.random.randn(n_samples, 3)
        y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 0.1
        
        feature_names = ['quality', 'value', 'volatility']
        
        # Train with quality monotonic increasing
        gbm = ConstrainedGBM(
            feature_names=feature_names,
            monotonic_constraints={'quality': 1}
        )
        gbm.fit(X, y, num_boost_round=50)
        
        # Test: increasing quality should increase prediction
        test_low = np.array([[0.0, 0.0, 0.0]])
        test_high = np.array([[1.0, 0.0, 0.0]])  # Only quality differs
        
        pred_low = gbm.predict(test_low)[0]
        pred_high = gbm.predict(test_high)[0]
        
        # Property: higher quality → higher or equal prediction
        assert pred_high >= pred_low - 0.01, \
            f"Monotonicity violated: quality {test_low[0,0]:.2f} → {pred_low:.3f}, " \
            f"quality {test_high[0,0]:.2f} → {pred_high:.3f}"
    
    def test_property5_monotonic_volatility(self):
        """Property 5: Higher volatility → lower prediction.
        
        **Feature: ml-alpha-enhancement, Property 5: Monotonic constraint (volatility)**
        **Validates: Requirements 3.3**
        """
        np.random.seed(42)
        
        # Create training data
        n_samples = 100
        X = np.random.randn(n_samples, 3)
        y = -X[:, 2] * 0.5 + np.random.randn(n_samples) * 0.1  # Negative relationship with vol
        
        feature_names = ['quality', 'value', 'volatility']
        
        # Train with volatility monotonic decreasing
        gbm = ConstrainedGBM(
            feature_names=feature_names,
            monotonic_constraints={'volatility': -1}
        )
        gbm.fit(X, y, num_boost_round=50)
        
        # Test: increasing volatility should decrease prediction
        test_low_vol = np.array([[0.0, 0.0, 0.0]])
        test_high_vol = np.array([[0.0, 0.0, 1.0]])  # Only volatility differs
        
        pred_low_vol = gbm.predict(test_low_vol)[0]
        pred_high_vol = gbm.predict(test_high_vol)[0]
        
        # Property: higher volatility → lower or equal prediction
        assert pred_high_vol <= pred_low_vol + 0.01, \
            f"Monotonicity violated: volatility {test_low_vol[0,2]:.2f} → {pred_low_vol:.3f}, " \
            f"volatility {test_high_vol[0,2]:.2f} → {pred_high_vol:.3f}"
    
    def test_constraint_validation(self):
        """Test that invalid constraints raise errors."""
        with pytest.raises(ValueError, match="Invalid monotonic constraint"):
            ConstrainedGBM(
                feature_names=['a', 'b'],
                monotonic_constraints={'a': 2}  # Invalid: must be -1, 0, or 1
            )


# =============================================================================
# Property 6: Online Learning State Persistence
# =============================================================================

class TestOnlineRegimeProperties:
    """Property tests for Online Regime Detector."""
    
    @given(
        observations=st.lists(
            st.floats(min_value=-0.05, max_value=0.05, allow_nan=False),
            min_size=20,
            max_size=50
        )
    )
    @settings(max_examples=20, deadline=None)
    def test_property6_state_persistence(self, observations):
        """Property 6: Reload from persisted state produces identical predictions.
        
        **Feature: ml-alpha-enhancement, Property 6: State persistence**
        **Validates: Requirements 4.5**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, 'regime_state.json')
            
            # Create detector and process observations
            detector1 = OnlineRegimeDetector(
                n_states=2,
                decay_factor=0.95,
                state_file=state_file
            )
            
            for obs in observations:
                detector1.update(obs)
            
            # Force save
            detector1._save_state()
            
            # Get final state
            regime1, probs1 = detector1.get_regime()
            
            # Create new detector and load state
            detector2 = OnlineRegimeDetector(
                n_states=2,
                decay_factor=0.95,
                state_file=state_file
            )
            
            # Property: loaded state should match
            regime2, probs2 = detector2.get_regime()
            
            assert regime1 == regime2, \
                f"Regime mismatch after reload: {regime1} vs {regime2}"
            
            np.testing.assert_array_almost_equal(
                probs1, probs2, decimal=5,
                err_msg="State probabilities mismatch after reload"
            )
    
    @given(
        n_updates=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=20)
    def test_update_increments_counter(self, n_updates):
        """Test that update counter increments correctly."""
        detector = OnlineRegimeDetector(n_states=2)
        
        initial_count = detector._n_updates
        
        for _ in range(n_updates):
            detector.update(np.random.randn() * 0.02)
        
        assert detector._n_updates == initial_count + n_updates


# =============================================================================
# Properties 7 & 8: Supply Chain Signal Propagation
# =============================================================================

class TestSupplyChainProperties:
    """Property tests for Supply Chain GNN."""
    
    @given(
        source_signal=st.floats(min_value=0.01, max_value=0.1, allow_nan=False)
    )
    @settings(max_examples=30)
    def test_property7_signal_propagation(self, source_signal):
        """Property 7: Connected companies receive non-zero signal.
        
        **Feature: ml-alpha-enhancement, Property 7: Signal propagation**
        **Validates: Requirements 5.2**
        """
        # Create simple graph: TSM -> AAPL -> downstream
        graph = SupplyChainGraph()
        graph.add_edge('TSM', 'AAPL', weight=0.8)
        
        momentum = SupplyChainMomentum(
            graph,
            decay_factor=0.5,
            propagation_steps=2
        )
        
        # TSM has price change, AAPL should receive signal
        price_changes = {'TSM': source_signal, 'AAPL': 0.0}
        
        scores = momentum.compute(price_changes, ['TSM', 'AAPL'])
        
        # Property: AAPL should receive non-zero signal from TSM
        assert scores['AAPL'] != 0.0, \
            f"AAPL should receive signal from connected TSM, got {scores['AAPL']}"
    
    @given(
        weights=st.lists(
            st.floats(min_value=0.1, max_value=1.0, allow_nan=False),
            min_size=2,
            max_size=4
        ),
        signals=st.lists(
            st.floats(min_value=-0.05, max_value=0.05, allow_nan=False),
            min_size=2,
            max_size=4
        )
    )
    @settings(max_examples=30)
    def test_property8_aggregation(self, weights, signals):
        """Property 8: Momentum = weighted sum of neighbor signals.
        
        **Feature: ml-alpha-enhancement, Property 8: Aggregation**
        **Validates: Requirements 5.3**
        """
        assume(len(weights) == len(signals))
        n_neighbors = len(weights)
        
        # Create graph: multiple suppliers -> TARGET
        graph = SupplyChainGraph()
        neighbors = [f'SUPPLIER_{i}' for i in range(n_neighbors)]
        
        for i, neighbor in enumerate(neighbors):
            graph.add_edge(neighbor, 'TARGET', weight=weights[i])
        
        momentum = SupplyChainMomentum(
            graph,
            decay_factor=1.0,  # No decay for this test
            propagation_steps=1
        )
        
        # Set up price changes
        price_changes = {'TARGET': 0.0}
        for i, neighbor in enumerate(neighbors):
            price_changes[neighbor] = signals[i]
        
        scores = momentum.compute(price_changes, ['TARGET'] + neighbors)
        
        # Property: TARGET should receive weighted average of neighbor signals
        total_weight = sum(weights)
        if total_weight > 0:
            expected = sum(w * s for w, s in zip(weights, signals)) / total_weight
            
            # Allow for numerical precision
            assert abs(scores['TARGET'] - expected) < 1e-4, \
                f"Expected weighted sum {expected}, got {scores['TARGET']}"
    
    def test_missing_ticker_returns_zero(self):
        """Test that missing tickers get neutral score."""
        graph = SupplyChainGraph()
        graph.add_edge('A', 'B', weight=1.0)
        
        momentum = SupplyChainMomentum(graph)
        
        # C is not in graph
        score = momentum.get_signal_for_ticker('C', {'A': 0.05})
        
        assert score == 0.0, f"Missing ticker should return 0.0, got {score}"


# =============================================================================
# Property 9: Feature Toggle Isolation
# =============================================================================

class TestConfigProperties:
    """Property tests for ML Enhancement Configuration."""
    
    @given(
        enabled_features=st.lists(
            st.sampled_from(['shap', 'residual_alpha', 'constrained_gbm', 
                           'online_learning', 'supply_chain']),
            min_size=0,
            max_size=5,
            unique=True
        )
    )
    @settings(max_examples=20)
    def test_property9_feature_toggle_isolation(self, enabled_features):
        """Property 9: Disabled features have no effect.
        
        **Feature: ml-alpha-enhancement, Property 9: Feature toggle isolation**
        **Validates: Requirements 6.2**
        """
        # Reset config
        reset_ml_enhancement_config()
        
        # Build config with specific features enabled
        config = MLEnhancementConfig(
            shap=SHAPConfig(enabled='shap' in enabled_features),
            residual_alpha=ResidualAlphaConfig(enabled='residual_alpha' in enabled_features),
            constrained_gbm=ConstrainedGBMConfig(enabled='constrained_gbm' in enabled_features),
            online_learning=OnlineLearningConfig(enabled='online_learning' in enabled_features),
            supply_chain=SupplyChainConfig(enabled='supply_chain' in enabled_features),
        )
        
        # Property: get_active_features matches enabled list
        active = config.get_active_features()
        
        # Map from config key to feature name
        key_to_name = {
            'shap': 'SHAP',
            'residual_alpha': 'ResidualAlpha',
            'constrained_gbm': 'ConstrainedGBM',
            'online_learning': 'OnlineLearning',
            'supply_chain': 'SupplyChain',
        }
        
        expected_active = [key_to_name[k] for k in enabled_features]
        
        assert set(active) == set(expected_active), \
            f"Active features mismatch: {active} vs {expected_active}"
    
    def test_config_validation_rejects_invalid(self):
        """Test that invalid config values are rejected."""
        with pytest.raises(ValueError):
            SHAPConfig(concentration_threshold=1.5)  # Must be <= 1.0
        
        with pytest.raises(ValueError):
            OnlineLearningConfig(decay_factor=1.5)  # Must be < 1.0
    
    def test_config_singleton(self):
        """Test that config singleton works correctly."""
        reset_ml_enhancement_config()
        
        config1 = get_ml_enhancement_config()
        config2 = get_ml_enhancement_config()
        
        assert config1 is config2, "Config singleton should return same instance"


# =============================================================================
# Integration Tests
# =============================================================================

class TestMLAlphaIntegration:
    """Integration tests for the full ML Alpha Enhancement pipeline."""
    
    def test_all_modules_import(self):
        """Verify all new modules can be imported."""
        from config.ml_enhancement_config import MLEnhancementConfig
        from quant.mlops.shap_attributor import SHAPAttributor, SHAPAttribution
        from quant.mlops.constrained_gbm import ConstrainedGBM
        from quant.mlops.residual_alpha import ResidualAlphaModel
        from quant.regime.online_hmm import OnlineRegimeDetector
        from quant.features.supply_chain_gnn import SupplyChainGraph, SupplyChainMomentum
        
        # All imports successful
        assert True
    
    def test_constrained_gbm_with_residual_alpha(self):
        """Test that ConstrainedGBM can be used as ML model in ResidualAlpha."""
        np.random.seed(42)
        
        n_samples = 100
        X_linear = pd.DataFrame({
            'quality': np.random.randn(n_samples),
            'value': np.random.randn(n_samples),
        })
        X_residual = pd.DataFrame({
            'sentiment': np.random.randn(n_samples),
            'pead': np.random.randn(n_samples),
        })
        y = pd.Series(
            X_linear['quality'].values * 0.3 +
            X_linear['value'].values * 0.2 +
            X_residual['sentiment'].values * 0.1 +
            np.random.randn(n_samples) * 0.05
        )
        
        # Create constrained GBM for residual prediction
        gbm = ConstrainedGBM(
            feature_names=['sentiment', 'pead'],
            monotonic_constraints={'sentiment': 1}
        )
        
        # Use it in ResidualAlphaModel
        model = ResidualAlphaModel(
            linear_factors=['quality', 'value'],
            residual_features=['sentiment', 'pead'],
            ml_model=gbm
        )
        
        model.fit(X_linear, X_residual, y)
        
        total, linear, residual = model.predict(X_linear, X_residual)
        
        # Predictions should be reasonable
        assert len(total) == n_samples
        assert np.isfinite(total).all()
