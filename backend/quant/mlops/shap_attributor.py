"""SHAP Factor Attribution Module.

Provides SHAP-based factor attribution for ML model predictions.
Decomposes predictions into contributions from each factor, enabling:
- Interpretability of black-box models
- Factor exposure monitoring
- Concentration risk warnings

Example:
    >>> attributor = SHAPAttributor(model, feature_names)
    >>> attributor.fit(X_background)
    >>> attributions = attributor.explain(X, tickers)
    >>> concentration = attributor.check_concentration(attributions)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SHAPAttribution:
    """SHAP attribution result for a single prediction.
    
    Attributes:
        ticker: Stock ticker symbol.
        base_value: SHAP expected value (average prediction over background).
        prediction: Actual model prediction for this sample.
        factor_contributions: Mapping of factor name to SHAP value contribution.
    """
    ticker: str
    base_value: float
    prediction: float
    factor_contributions: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of the attribution.
        """
        return {
            'ticker': self.ticker,
            'base_value': self.base_value,
            'prediction': self.prediction,
            'contributions': self.factor_contributions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SHAPAttribution':
        """Create SHAPAttribution from dictionary.
        
        Args:
            data: Dictionary with attribution data.
            
        Returns:
            SHAPAttribution instance.
        """
        return cls(
            ticker=data['ticker'],
            base_value=data['base_value'],
            prediction=data['prediction'],
            factor_contributions=data.get('contributions', {})
        )
    
    def get_top_factors(self, n: int = 3) -> List[tuple]:
        """Get the top N factors by absolute contribution.
        
        Args:
            n: Number of top factors to return.
            
        Returns:
            List of (factor_name, contribution) tuples sorted by |contribution|.
        """
        sorted_factors = sorted(
            self.factor_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_factors[:n]


class SHAPAttributor:
    """Computes SHAP values for factor model predictions.
    
    Uses TreeExplainer for tree-based models (XGBoost, LightGBM).
    Supports concentration monitoring to detect factor exposure risk.
    
    Attributes:
        model: The trained model to explain (must support predict).
        feature_names: List of feature/factor names.
        explainer: SHAP TreeExplainer instance (initialized by fit()).
    """
    
    def __init__(self, model: Any, feature_names: List[str]):
        """Initialize SHAPAttributor.
        
        Args:
            model: Trained tree-based model (XGBoost, LightGBM, etc.).
            feature_names: List of feature names in the same order as model input.
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer: Optional[Any] = None
        self._base_value: Optional[float] = None
        
    def fit(self, X_background: pd.DataFrame) -> 'SHAPAttributor':
        """Initialize SHAP explainer with background data.
        
        The background data is used to compute the expected (baseline) prediction.
        Using a representative sample (100-1000 samples) is recommended.
        
        Args:
            X_background: Background dataset for SHAP explainer.
            
        Returns:
            self for method chaining.
            
        Raises:
            ImportError: If shap package is not installed.
        """
        try:
            import shap
        except ImportError:
            logger.error("shap package not installed. Install with: pip install shap")
            raise
        
        try:
            self.explainer = shap.TreeExplainer(self.model)
            logger.info(
                f"Initialized SHAP TreeExplainer with {len(X_background)} background samples"
            )
        except Exception as e:
            logger.warning(f"TreeExplainer failed, trying Explainer: {e}")
            # Fallback to generic Explainer
            self.explainer = shap.Explainer(self.model, X_background)
            
        return self
    
    def explain(
        self, 
        X: pd.DataFrame, 
        tickers: List[str]
    ) -> List[SHAPAttribution]:
        """Compute SHAP attributions for predictions.
        
        Args:
            X: Feature matrix to explain (n_samples, n_features).
            tickers: List of ticker symbols corresponding to each row.
            
        Returns:
            List of SHAPAttribution objects, one per sample.
            
        Raises:
            ValueError: If explainer not fitted or input size mismatch.
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        if len(X) != len(tickers):
            raise ValueError(
                f"X has {len(X)} rows but {len(tickers)} tickers provided"
            )
        
        try:
            # Compute SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Multi-output model (e.g., classification)
                shap_values = shap_values[0]
            
            # Get base value
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = float(base_value[0])
            else:
                base_value = float(base_value)
            
            # Get predictions
            predictions = self.model.predict(X)
            if isinstance(predictions, pd.Series):
                predictions = predictions.values
            
            # Build attributions
            attributions = []
            for i, ticker in enumerate(tickers):
                contributions = {}
                for j, name in enumerate(self.feature_names):
                    value = shap_values[i, j] if shap_values.ndim > 1 else shap_values[j]
                    contributions[name] = float(value)
                
                attributions.append(SHAPAttribution(
                    ticker=ticker,
                    base_value=base_value,
                    prediction=float(predictions[i]),
                    factor_contributions=contributions
                ))
            
            logger.debug(f"Computed SHAP attributions for {len(attributions)} samples")
            return attributions
            
        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
            # Return empty attributions on failure (graceful degradation)
            return [
                SHAPAttribution(
                    ticker=ticker,
                    base_value=0.0,
                    prediction=0.0,
                    factor_contributions={}
                )
                for ticker in tickers
            ]
    
    def check_concentration(
        self, 
        attributions: List[SHAPAttribution],
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Check if any factor dominates the portfolio attribution.
        
        Aggregates absolute SHAP contributions across all stocks and
        computes the concentration (fraction of total) for each factor.
        
        Args:
            attributions: List of SHAP attributions for portfolio.
            threshold: Concentration threshold (0-1) for warnings.
            
        Returns:
            Dictionary mapping factor names to their concentration (0-1).
        """
        if not attributions:
            return {}
        
        # Aggregate absolute contributions across portfolio
        total_by_factor: Dict[str, float] = {}
        for attr in attributions:
            for factor, value in attr.factor_contributions.items():
                total_by_factor[factor] = total_by_factor.get(factor, 0) + abs(value)
        
        # Normalize to get concentration
        total = sum(total_by_factor.values())
        if total > 0:
            concentration = {k: v / total for k, v in total_by_factor.items()}
        else:
            concentration = {k: 0.0 for k in total_by_factor}
        
        # Check threshold and log warnings
        for factor, conc in concentration.items():
            if conc > threshold:
                logger.warning(
                    f"Factor concentration warning: {factor} = {conc:.1%} > {threshold:.1%}"
                )
        
        return concentration
    
    def explain_single(
        self, 
        X_single: pd.DataFrame, 
        ticker: str
    ) -> Optional[SHAPAttribution]:
        """Explain a single prediction.
        
        Convenience method for explaining one sample at a time.
        
        Args:
            X_single: Single-row feature DataFrame.
            ticker: Ticker symbol for the sample.
            
        Returns:
            SHAPAttribution for the sample, or None on failure.
        """
        try:
            attributions = self.explain(X_single, [ticker])
            return attributions[0] if attributions else None
        except Exception as e:
            logger.error(f"Failed to explain {ticker}: {e}")
            return None
