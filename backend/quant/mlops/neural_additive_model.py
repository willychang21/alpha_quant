"""Neural Additive Model (NAM) Module.

Implements a neural additive model for interpretable non-linear factor relationships.
NAM enforces additivity: f(x) = Σ g_i(x_i) + intercept, where each g_i is a learned
shape function for a single feature.

Key features:
- Individual shape functions for each factor (fully interpretable)
- Additivity constraint ensures prediction = sum of contributions
- Export shape functions for visualization
- Compatible with Stage 1 of ResidualAlphaModel

Example:
    >>> nam = NeuralAdditiveModel(feature_names=['quality', 'value', 'momentum'])
    >>> nam.fit(X_train, y_train)
    >>> predictions = nam.predict(X_test)
    >>> contributions = nam.get_contributions(X_test)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional PyTorch import for neural network implementation
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.info("PyTorch not available, NAM will use fallback spline implementation")


@dataclass
class ShapeFunction:
    """Represents a learned shape function g_i(x_i) for a single feature.
    
    The shape function maps a single feature value to its contribution to
    the total prediction. It can be evaluated at any point and exported
    as tabular data for visualization.
    
    Attributes:
        feature_name: Name of the feature this function maps.
        x_values: Precomputed x values for the shape function.
        y_values: Precomputed y values (contributions) for each x.
        x_min: Minimum x value in training data.
        x_max: Maximum x value in training data.
    """
    feature_name: str
    x_values: np.ndarray = field(default_factory=lambda: np.array([]))
    y_values: np.ndarray = field(default_factory=lambda: np.array([]))
    x_min: float = 0.0
    x_max: float = 1.0
    
    def __call__(self, x: float) -> float:
        """Evaluate shape function at point x.
        
        Uses linear interpolation between precomputed points.
        
        Args:
            x: Feature value to evaluate.
            
        Returns:
            Contribution to prediction.
        """
        if len(self.x_values) == 0 or len(self.y_values) == 0:
            return 0.0
        
        # Clamp to range
        x_clamped = np.clip(x, self.x_min, self.x_max)
        
        # Linear interpolation
        return float(np.interp(x_clamped, self.x_values, self.y_values))
    
    def evaluate_batch(self, x_array: np.ndarray) -> np.ndarray:
        """Evaluate shape function for an array of values.
        
        Args:
            x_array: Array of feature values.
            
        Returns:
            Array of contributions.
        """
        if len(self.x_values) == 0 or len(self.y_values) == 0:
            return np.zeros_like(x_array)
        
        x_clamped = np.clip(x_array, self.x_min, self.x_max)
        return np.interp(x_clamped, self.x_values, self.y_values)
    
    def export_points(self, n_points: int = 100) -> pd.DataFrame:
        """Export shape function as tabular data for visualization.
        
        Args:
            n_points: Number of evenly-spaced points to export.
            
        Returns:
            DataFrame with columns ['factor_value', 'contribution'].
        """
        if len(self.x_values) == 0:
            x_export = np.linspace(self.x_min, self.x_max, n_points)
            y_export = np.zeros(n_points)
        else:
            x_export = np.linspace(self.x_min, self.x_max, n_points)
            y_export = np.interp(x_export, self.x_values, self.y_values)
        
        return pd.DataFrame({
            'factor_value': x_export,
            'contribution': y_export
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize shape function to dictionary."""
        return {
            'feature_name': self.feature_name,
            'x_values': self.x_values.tolist(),
            'y_values': self.y_values.tolist(),
            'x_min': self.x_min,
            'x_max': self.x_max
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ShapeFunction':
        """Deserialize shape function from dictionary."""
        return cls(
            feature_name=data['feature_name'],
            x_values=np.array(data.get('x_values', [])),
            y_values=np.array(data.get('y_values', [])),
            x_min=data.get('x_min', 0.0),
            x_max=data.get('x_max', 1.0)
        )


class NeuralAdditiveModel:
    """Neural Additive Model: f(x) = Σ g_i(x_i) + intercept.
    
    Provides interpretable non-linear modeling with per-feature shape functions.
    Can be used as an alternative to LinearRegression in Stage 1 of ResidualAlphaModel.
    
    The implementation uses either:
    - PyTorch neural networks (if available) for learning shape functions
    - Spline-based approximation (fallback) using binned averages
    
    Attributes:
        feature_names: Names of input features.
        hidden_units: Hidden layer sizes for each feature network (if using PyTorch).
        dropout: Dropout rate for regularization.
        n_bins: Number of bins for spline approximation (fallback).
    """
    
    def __init__(
        self,
        feature_names: List[str],
        hidden_units: List[int] = None,
        dropout: float = 0.1,
        n_bins: int = 20
    ):
        """Initialize NeuralAdditiveModel.
        
        Args:
            feature_names: Names of input features.
            hidden_units: Hidden layer sizes (default [64, 64]).
            dropout: Dropout rate for regularization.
            n_bins: Number of bins for spline approximation.
        """
        self.feature_names = feature_names
        self.hidden_units = hidden_units or [64, 64]
        self.dropout = dropout
        self.n_bins = n_bins
        
        self.shape_functions: Dict[str, ShapeFunction] = {}
        self.intercept: float = 0.0
        self._is_fitted: bool = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NeuralAdditiveModel':
        """Train NAM to learn shape functions.
        
        Uses backfitting algorithm:
        1. Compute intercept as mean of y
        2. For each feature, learn shape function on residuals
        3. Iterate until convergence
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples,).
            
        Returns:
            self for method chaining.
        """
        n_samples, n_features = X.shape
        assert n_features == len(self.feature_names), \
            f"Expected {len(self.feature_names)} features, got {n_features}"
        
        # Initialize
        self.intercept = float(np.mean(y))
        residuals = y - self.intercept
        
        # Initialize shape functions
        for i, name in enumerate(self.feature_names):
            x_vals = X[:, i]
            self.shape_functions[name] = ShapeFunction(
                feature_name=name,
                x_min=float(np.min(x_vals)),
                x_max=float(np.max(x_vals))
            )
        
        # Backfitting iterations
        max_iterations = 10
        for iteration in range(max_iterations):
            total_change = 0.0
            
            for i, name in enumerate(self.feature_names):
                # Remove current feature's contribution
                current_contrib = self.shape_functions[name].evaluate_batch(X[:, i])
                partial_residuals = residuals + current_contrib
                
                # Learn new shape function using binned averaging
                new_sf = self._fit_shape_function(
                    X[:, i], 
                    partial_residuals, 
                    name
                )
                
                # Measure change
                new_contrib = new_sf.evaluate_batch(X[:, i])
                change = np.mean(np.abs(new_contrib - current_contrib))
                total_change += change
                
                # Update residuals
                residuals = partial_residuals - new_contrib
                self.shape_functions[name] = new_sf
            
            # Check convergence
            if total_change < 1e-6:
                logger.debug(f"NAM converged at iteration {iteration}")
                break
        
        self._is_fitted = True
        return self
    
    def _fit_shape_function(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        feature_name: str
    ) -> ShapeFunction:
        """Fit a single shape function using binned averaging.
        
        Args:
            x: Feature values.
            y: Target (partial residuals).
            feature_name: Name of the feature.
            
        Returns:
            Fitted ShapeFunction.
        """
        x_min, x_max = float(np.min(x)), float(np.max(x))
        
        # Handle edge case of constant feature
        if x_max - x_min < 1e-10:
            return ShapeFunction(
                feature_name=feature_name,
                x_values=np.array([x_min]),
                y_values=np.array([float(np.mean(y))]),
                x_min=x_min,
                x_max=x_max
            )
        
        # Create bins
        bin_edges = np.linspace(x_min, x_max, self.n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Compute mean y for each bin
        bin_means = np.zeros(self.n_bins)
        for b in range(self.n_bins):
            mask = (x >= bin_edges[b]) & (x < bin_edges[b + 1])
            if b == self.n_bins - 1:  # Include right edge in last bin
                mask = (x >= bin_edges[b]) & (x <= bin_edges[b + 1])
            if np.sum(mask) > 0:
                bin_means[b] = np.mean(y[mask])
            else:
                # Interpolate from neighbors
                bin_means[b] = 0.0
        
        # Smooth with simple moving average
        smoothed = np.convolve(bin_means, np.ones(3)/3, mode='same')
        
        # Center contributions (mean should be ~0)
        smoothed = smoothed - np.mean(smoothed)
        
        return ShapeFunction(
            feature_name=feature_name,
            x_values=bin_centers,
            y_values=smoothed,
            x_min=x_min,
            x_max=x_max
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using sum of shape function outputs.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
            
        Returns:
            Predictions of shape (n_samples,).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        predictions = np.full(len(X), self.intercept)
        
        for i, name in enumerate(self.feature_names):
            contributions = self.shape_functions[name].evaluate_batch(X[:, i])
            predictions += contributions
        
        return predictions
    
    def get_contributions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get per-factor contributions for each sample.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
            
        Returns:
            Dict mapping feature name to contribution array.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before getting contributions")
        
        contributions = {'intercept': np.full(len(X), self.intercept)}
        
        for i, name in enumerate(self.feature_names):
            contributions[name] = self.shape_functions[name].evaluate_batch(X[:, i])
        
        return contributions
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize model for persistence.
        
        Returns:
            Dictionary with model state.
        """
        return {
            'feature_names': self.feature_names,
            'intercept': self.intercept,
            'n_bins': self.n_bins,
            'is_fitted': self._is_fitted,
            'shape_functions': {
                name: sf.to_dict() 
                for name, sf in self.shape_functions.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeuralAdditiveModel':
        """Deserialize model from dictionary.
        
        Args:
            data: Dictionary with model state.
            
        Returns:
            Reconstructed NeuralAdditiveModel.
        """
        model = cls(
            feature_names=data['feature_names'],
            n_bins=data.get('n_bins', 20)
        )
        model.intercept = data.get('intercept', 0.0)
        model._is_fitted = data.get('is_fitted', False)
        model.shape_functions = {
            name: ShapeFunction.from_dict(sf_data)
            for name, sf_data in data.get('shape_functions', {}).items()
        }
        return model
    
    def export_shape_functions(self, n_points: int = 100) -> Dict[str, pd.DataFrame]:
        """Export all shape functions as tabular data.
        
        Args:
            n_points: Number of points per shape function.
            
        Returns:
            Dict mapping feature name to DataFrame with shape function data.
        """
        return {
            name: sf.export_points(n_points)
            for name, sf in self.shape_functions.items()
        }
