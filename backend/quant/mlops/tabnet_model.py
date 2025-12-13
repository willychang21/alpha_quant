"""TabNet Model Module.

Implements TabNet with ConstrainedGBM-compatible interface for A/B testing.
TabNet uses sequential attention for feature selection at each decision step.

Key features:
- Same fit/predict interface as ConstrainedGBM
- Sparse feature selection masks for interpretability
- SHAP-compatible feature importance output
- Graceful fallback when pytorch-tabnet unavailable

Example:
    >>> model = TabNetModel(feature_names=['quality', 'value', 'momentum'])
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> importance = model.get_feature_importance()
"""

from typing import Dict, List, Optional, Any
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Optional pytorch-tabnet import
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    logger.info("pytorch-tabnet not available, TabNet will use fallback implementation")


class TabNetModel:
    """TabNet with ConstrainedGBM-compatible interface.
    
    Provides the same interface as ConstrainedGBM to enable A/B testing.
    When pytorch-tabnet is not available, uses a simple fallback model.
    
    Attributes:
        feature_names: Names of input features.
        monotonic_constraints: Constraint direction per feature (compatibility only).
        n_steps: Number of decision steps in TabNet.
        n_a: Attention embedding dimension.
        n_d: Output embedding dimension.
    """
    
    def __init__(
        self,
        feature_names: List[str],
        monotonic_constraints: Optional[Dict[str, int]] = None,
        n_steps: int = 3,
        n_a: int = 8,
        n_d: int = 8
    ):
        """Initialize TabNetModel.
        
        Args:
            feature_names: Names of input features.
            monotonic_constraints: Constraint dict (for interface compatibility).
            n_steps: Number of decision steps.
            n_a: Attention embedding dimension.
            n_d: Output embedding dimension.
        """
        self.feature_names = feature_names
        self.monotonic_constraints = monotonic_constraints or {}
        self.n_steps = n_steps
        self.n_a = n_a
        self.n_d = n_d
        
        self._model: Optional[Any] = None
        self._feature_masks: Optional[np.ndarray] = None
        self._is_fitted: bool = False
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        eval_set: Optional[List] = None,
        max_epochs: int = 100,
        patience: int = 10,
        **kwargs
    ) -> 'TabNetModel':
        """Train TabNet model.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples,).
            eval_set: Optional validation set for early stopping.
            max_epochs: Maximum training epochs.
            patience: Early stopping patience.
            **kwargs: Additional TabNet parameters.
            
        Returns:
            self for method chaining.
        """
        if TABNET_AVAILABLE:
            self._model = TabNetRegressor(
                n_steps=self.n_steps,
                n_a=self.n_a,
                n_d=self.n_d,
                verbose=0
            )
            
            y_reshaped = y.reshape(-1, 1) if y.ndim == 1 else y
            
            self._model.fit(
                X, y_reshaped,
                eval_set=eval_set,
                max_epochs=max_epochs,
                patience=patience,
                **kwargs
            )
        else:
            # Fallback: simple linear model
            self._fallback_weights = np.zeros(len(self.feature_names) + 1)
            # Least squares solution
            X_with_bias = np.column_stack([np.ones(len(X)), X])
            try:
                self._fallback_weights = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                self._fallback_weights = np.zeros(len(self.feature_names) + 1)
        
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with mask capture.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
            
        Returns:
            Predictions of shape (n_samples,).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if TABNET_AVAILABLE and self._model is not None:
            predictions = self._model.predict(X)
            # Capture feature masks for interpretability
            try:
                self._feature_masks = self._model.explain(X)[0]
            except Exception:
                self._feature_masks = None
            return predictions.flatten()
        else:
            # Fallback prediction
            X_with_bias = np.column_stack([np.ones(len(X)), X])
            return X_with_bias @ self._fallback_weights
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get aggregate feature importance (SHAP-compatible format).
        
        Returns:
            Dict mapping feature name to importance score.
        """
        if TABNET_AVAILABLE and self._feature_masks is not None:
            # Aggregate masks across samples
            importance = self._feature_masks.mean(axis=0)
            return dict(zip(self.feature_names, importance))
        elif hasattr(self, '_fallback_weights'):
            # Use absolute weight values (skip bias)
            weights = np.abs(self._fallback_weights[1:])
            return dict(zip(self.feature_names, weights))
        
        return {name: 0.0 for name in self.feature_names}
    
    def get_feature_masks(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Get per-sample feature selection masks.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Mask array of shape (n_samples, n_features) or None.
        """
        if TABNET_AVAILABLE and self._model is not None:
            try:
                return self._model.explain(X)[0]
            except Exception:
                return None
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize model state."""
        return {
            'feature_names': self.feature_names,
            'n_steps': self.n_steps,
            'n_a': self.n_a,
            'n_d': self.n_d,
            'is_fitted': self._is_fitted,
            'fallback_weights': (
                self._fallback_weights.tolist() 
                if hasattr(self, '_fallback_weights') else None
            )
        }
