"""Residual Alpha Model Module.

Implements a two-stage alpha extraction model:
1. Stage 1: Linear decomposition using traditional factors
2. Stage 2: ML prediction on residuals using alternative features

This approach extracts pure idiosyncratic alpha by first removing
exposure to common risk factors, then predicting the unexplained
portion with ML.

Example:
    >>> model = ResidualAlphaModel(
    ...     linear_factors=['quality', 'value', 'momentum'],
    ...     residual_features=['sentiment', 'pead', 'capital_flow']
    ... )
    >>> model.fit(X_linear, X_residual, y)
    >>> total, linear, residual = model.predict(X_linear_new, X_residual_new)
"""

from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

logger = logging.getLogger(__name__)


class ResidualAlphaModel:
    """Two-stage model: Linear factor decomposition + ML residual prediction.
    
    Stage 1: Fit a linear model to decompose returns into factor exposures.
             y = β₁*quality + β₂*value + β₃*momentum + ... + ε
    
    Stage 2: Use ML to predict residuals (ε) from alternative features.
             ε = f(sentiment, pead, capital_flow, ...)
    
    Final prediction: linear_score + residual_alpha
    
    Attributes:
        linear_factors: List of factor names for Stage 1.
        residual_features: List of feature names for Stage 2.
        linear_model: Fitted linear regression model.
        ml_model: Fitted ML model for residual prediction (optional).
    """
    
    def __init__(
        self,
        linear_factors: List[str],
        residual_features: List[str],
        ml_model: Optional[Any] = None,
        use_ridge: bool = False,
        ridge_alpha: float = 1.0
    ):
        """Initialize ResidualAlphaModel.
        
        Args:
            linear_factors: Factor names for linear decomposition (Stage 1).
            residual_features: Feature names for residual prediction (Stage 2).
            ml_model: Pre-configured ML model for Stage 2 (e.g., ConstrainedGBM).
                      If None, Stage 2 is skipped (linear-only mode).
            use_ridge: Use Ridge regression instead of OLS for Stage 1.
            ridge_alpha: Regularization strength for Ridge regression.
        """
        self.linear_factors = linear_factors
        self.residual_features = residual_features
        self.ml_model = ml_model
        
        # Stage 1: Linear model
        if use_ridge:
            self.linear_model = Ridge(alpha=ridge_alpha)
        else:
            self.linear_model = LinearRegression()
        
        # Metrics
        self._linear_r2: Optional[float] = None
        self._ml_r2: Optional[float] = None
        self._is_fitted: bool = False
        
    def fit(
        self, 
        X_linear: pd.DataFrame,
        X_residual: pd.DataFrame,
        y: pd.Series
    ) -> 'ResidualAlphaModel':
        """Fit both linear and ML components.
        
        Args:
            X_linear: Features for linear decomposition (n_samples, n_linear_factors).
            X_residual: Features for residual prediction (n_samples, n_residual_features).
            y: Target returns (n_samples,).
            
        Returns:
            self for method chaining.
        """
        # Convert to numpy
        X_lin = X_linear.values if isinstance(X_linear, pd.DataFrame) else X_linear
        X_res = X_residual.values if isinstance(X_residual, pd.DataFrame) else X_residual
        y_arr = y.values if isinstance(y, pd.Series) else y
        
        # Stage 1: Linear decomposition
        try:
            self.linear_model.fit(X_lin, y_arr)
            linear_pred = self.linear_model.predict(X_lin)
            residuals = y_arr - linear_pred
            
            # Compute R² for linear model
            self._linear_r2 = self.linear_model.score(X_lin, y_arr)
            logger.info(f"Linear model R²: {self._linear_r2:.4f}")
            
        except np.linalg.LinAlgError as e:
            logger.warning(f"Linear fit failed (singular matrix): {e}")
            # Fallback to Ridge with regularization
            self.linear_model = Ridge(alpha=1.0)
            self.linear_model.fit(X_lin, y_arr)
            linear_pred = self.linear_model.predict(X_lin)
            residuals = y_arr - linear_pred
            self._linear_r2 = self.linear_model.score(X_lin, y_arr)
            logger.info(f"Fallback Ridge R²: {self._linear_r2:.4f}")
        
        # Stage 2: ML on residuals
        if self.ml_model is not None:
            try:
                self.ml_model.fit(X_res, residuals)
                ml_pred = self.ml_model.predict(X_res)
                
                # Compute R² for residual prediction
                ss_res = np.sum((residuals - ml_pred) ** 2)
                ss_tot = np.sum((residuals - residuals.mean()) ** 2)
                self._ml_r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
                
                logger.info(f"ML residual model R²: {self._ml_r2:.4f}")
                
            except Exception as e:
                logger.warning(f"ML model training failed: {e}")
                self._ml_r2 = None
        
        self._is_fitted = True
        return self
    
    def predict(
        self, 
        X_linear: pd.DataFrame,
        X_residual: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict returns with decomposition.
        
        Args:
            X_linear: Features for linear prediction.
            X_residual: Features for residual prediction.
            
        Returns:
            Tuple of (total_score, linear_score, residual_alpha).
            
        Raises:
            ValueError: If model not fitted.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert to numpy
        X_lin = X_linear.values if isinstance(X_linear, pd.DataFrame) else X_linear
        X_res = X_residual.values if isinstance(X_residual, pd.DataFrame) else X_residual
        
        # Stage 1: Linear prediction
        linear_score = self.linear_model.predict(X_lin)
        
        # Stage 2: Residual prediction
        if self.ml_model is not None:
            try:
                residual_alpha = self.ml_model.predict(X_res)
            except Exception as e:
                logger.warning(f"ML prediction failed, using zero residuals: {e}")
                residual_alpha = np.zeros(len(X_lin))
        else:
            residual_alpha = np.zeros(len(X_lin))
        
        total_score = linear_score + residual_alpha
        
        return total_score, linear_score, residual_alpha
    
    def get_factor_exposures(self) -> Dict[str, float]:
        """Get linear factor coefficients (beta values).
        
        Returns:
            Dict mapping factor name to regression coefficient.
        """
        if not self._is_fitted:
            return {}
        
        return dict(zip(self.linear_factors, self.linear_model.coef_))
    
    def get_intercept(self) -> float:
        """Get linear model intercept.
        
        Returns:
            Intercept value (expected return at zero factor exposures).
        """
        if not self._is_fitted:
            return 0.0
        
        return float(self.linear_model.intercept_)
    
    def get_metrics(self) -> Dict[str, Optional[float]]:
        """Get model performance metrics.
        
        Returns:
            Dict with 'linear_r2' and 'ml_r2' values.
        """
        return {
            'linear_r2': self._linear_r2,
            'ml_r2': self._ml_r2,
        }
    
    def compute_orthogonality(
        self, 
        X_linear: pd.DataFrame,
        X_residual: pd.DataFrame,
        y: pd.Series
    ) -> float:
        """Compute correlation between linear and residual predictions.
        
        This measures how orthogonal the residual alpha is to the
        linear factor predictions. Lower values indicate better
        separation (less redundancy).
        
        Args:
            X_linear: Features for linear prediction.
            X_residual: Features for residual prediction.
            y: Target returns.
            
        Returns:
            Absolute correlation between linear and residual predictions.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        _, linear_score, residual_alpha = self.predict(X_linear, X_residual)
        
        # Compute correlation
        if np.std(linear_score) < 1e-10 or np.std(residual_alpha) < 1e-10:
            return 0.0
        
        correlation = np.corrcoef(linear_score, residual_alpha)[0, 1]
        
        return abs(correlation)
