"""Constrained Gradient Boosting Module.

Provides LightGBM with monotonic and interaction constraints to ensure
predictions conform to economic intuition:

- Monotonic constraints: Enforce that higher quality → higher returns
- Interaction constraints: Only allow interactions within factor groups

Example:
    >>> gbm = ConstrainedGBM(
    ...     feature_names=['quality', 'value', 'volatility'],
    ...     monotonic_constraints={'quality': 1, 'volatility': -1}
    ... )
    >>> gbm.fit(X_train, y_train)
    >>> predictions = gbm.predict(X_test)
"""

from typing import Any, Dict, List, Optional
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ConstrainedGBM:
    """LightGBM with monotonic and interaction constraints.
    
    Supports economic intuition constraints:
    - quality: +1 (higher quality → higher returns)
    - value: +1 (higher value/cheaper → higher returns)
    - momentum: +1 (higher momentum → higher returns)
    - volatility: -1 (higher volatility → lower returns)
    - beta: -1 (higher beta → lower risk-adjusted returns)
    
    Attributes:
        feature_names: List of feature names.
        monotonic_constraints: Dict mapping feature name to direction.
        interaction_constraints: List of feature name groups allowed to interact.
        params: LightGBM training parameters.
        model: Trained LightGBM model (None until fit() is called).
    """
    
    # Default monotonic constraints based on economic intuition
    DEFAULT_MONOTONIC: Dict[str, int] = {
        'quality': 1,      # Higher quality → higher returns
        'value': 1,        # Higher value (cheaper) → higher returns
        'momentum': 1,     # Higher momentum → higher returns
        'volatility': -1,  # Higher volatility → lower returns
        'beta': -1,        # Higher beta → lower risk-adjusted returns
    }
    
    # Default interaction groups
    DEFAULT_INTERACTION_GROUPS: List[List[str]] = [
        ['quality', 'value', 'growth'],      # Fundamental factors
        ['momentum', 'volatility', 'beta'],  # Technical/risk factors
        ['sentiment', 'pead', 'revisions'],  # Information factors
    ]
    
    def __init__(
        self,
        feature_names: List[str],
        monotonic_constraints: Optional[Dict[str, int]] = None,
        interaction_constraints: Optional[List[List[str]]] = None,
        **lgb_params: Any
    ):
        """Initialize ConstrainedGBM.
        
        Args:
            feature_names: List of feature names in order of appearance.
            monotonic_constraints: Dict mapping feature name to constraint:
                +1: monotonically increasing
                -1: monotonically decreasing
                 0: no constraint (default)
            interaction_constraints: List of feature name groups. Features
                can only interact with others in the same group.
            **lgb_params: Additional LightGBM parameters.
            
        Raises:
            ValueError: If constraint values are not in {-1, 0, 1}.
        """
        self.feature_names = feature_names
        self.monotonic_constraints = monotonic_constraints or {}
        self.interaction_constraints = interaction_constraints
        
        # Validate constraint values
        for name, direction in self.monotonic_constraints.items():
            if direction not in (-1, 0, 1):
                raise ValueError(
                    f"Invalid monotonic constraint for {name}: {direction}. "
                    f"Must be -1, 0, or 1."
                )
        
        # Build constraint vectors for LightGBM
        self._mono_vector = self._build_monotonic_vector()
        self._interaction_vector = self._build_interaction_vector()
        
        # LightGBM parameters
        self.params: Dict[str, Any] = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'monotone_constraints': self._mono_vector,
            'verbosity': -1,  # Suppress LightGBM output
            **lgb_params
        }
        
        if self._interaction_vector:
            self.params['interaction_constraints'] = self._interaction_vector
        
        self.model: Optional[Any] = None
        
        logger.debug(
            f"Initialized ConstrainedGBM with {len(feature_names)} features, "
            f"{len(self.monotonic_constraints)} monotonic constraints"
        )
    
    def _build_monotonic_vector(self) -> List[int]:
        """Convert constraint dict to LightGBM format.
        
        Returns:
            List of constraint values aligned with feature_names.
        """
        return [
            self.monotonic_constraints.get(name, 0)
            for name in self.feature_names
        ]
    
    def _build_interaction_vector(self) -> Optional[List[List[int]]]:
        """Convert interaction groups to LightGBM format.
        
        Returns:
            List of index groups, or None if no constraints.
        """
        if not self.interaction_constraints:
            return None
        
        name_to_idx = {name: i for i, name in enumerate(self.feature_names)}
        
        groups = []
        for group in self.interaction_constraints:
            indices = [
                name_to_idx[name] 
                for name in group 
                if name in name_to_idx
            ]
            if indices:
                groups.append(indices)
        
        return groups if groups else None
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        num_boost_round: int = 100,
        **kwargs: Any
    ) -> 'ConstrainedGBM':
        """Train the constrained model.
        
        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target values (n_samples,).
            num_boost_round: Number of boosting iterations.
            **kwargs: Additional arguments passed to lgb.train.
            
        Returns:
            self for method chaining.
            
        Raises:
            ImportError: If lightgbm is not installed.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            logger.error(
                "lightgbm package not installed. Install with: pip install lightgbm"
            )
            raise
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        train_data = lgb.Dataset(
            X, 
            label=y, 
            feature_name=self.feature_names
        )
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            **kwargs
        )
        
        logger.info(
            f"Trained ConstrainedGBM with {num_boost_round} rounds, "
            f"{len(self.feature_names)} features"
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix (n_samples, n_features).
            
        Returns:
            Predicted values (n_samples,).
            
        Raises:
            ValueError: If model not fitted.
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.model.predict(X)
    
    def get_feature_importance(
        self, 
        importance_type: str = 'gain'
    ) -> Dict[str, float]:
        """Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('gain', 'split', 'cover').
            
        Returns:
            Dict mapping feature name to importance score.
        """
        if self.model is None:
            return {}
        
        importance = self.model.feature_importance(importance_type=importance_type)
        return dict(zip(self.feature_names, importance))
    
    def get_monotonic_constraints_info(self) -> Dict[str, str]:
        """Get human-readable constraint information.
        
        Returns:
            Dict mapping feature name to constraint description.
        """
        descriptions = {
            1: "increasing (↑)",
            -1: "decreasing (↓)",
            0: "unconstrained"
        }
        
        return {
            name: descriptions[self.monotonic_constraints.get(name, 0)]
            for name in self.feature_names
        }
    
    def save_model(self, path: str) -> None:
        """Save model to file.
        
        Args:
            path: File path to save the model.
        """
        if self.model is not None:
            self.model.save_model(path)
            logger.info(f"Saved model to {path}")
    
    def load_model(self, path: str) -> 'ConstrainedGBM':
        """Load model from file.
        
        Args:
            path: File path to load the model from.
            
        Returns:
            self for method chaining.
        """
        try:
            import lightgbm as lgb
            self.model = lgb.Booster(model_file=path)
            logger.info(f"Loaded model from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        return self
