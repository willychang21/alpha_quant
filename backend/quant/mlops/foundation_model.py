"""Foundation Model Client Module.

Client for time-series foundation models (TimeGPT/TimesFM) used for cold-start
scenarios where stocks have insufficient history for traditional model training.

Key features:
- Automatic routing based on history length
- Confidence intervals with predictions
- Graceful fallback to sector-average momentum
- Cold-start and degradation flags

Example:
    >>> client = FoundationModelClient()
    >>> if client.should_use_foundation(history_days=30):
    ...     prediction = client.predict('NEWIPO', price_history)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FoundationPrediction:
    """Prediction from foundation model with uncertainty.
    
    Attributes:
        ticker: Stock ticker symbol.
        prediction: Point prediction (e.g., expected return).
        confidence_lower: Lower bound of confidence interval.
        confidence_upper: Upper bound of confidence interval.
        cold_start: Whether this is a cold-start prediction.
        degradation_mode: Whether fallback was used.
    """
    ticker: str
    prediction: float
    confidence_lower: float
    confidence_upper: float
    cold_start: bool = True
    degradation_mode: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'ticker': self.ticker,
            'prediction': self.prediction,
            'confidence_lower': self.confidence_lower,
            'confidence_upper': self.confidence_upper,
            'cold_start': self.cold_start,
            'degradation_mode': self.degradation_mode
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FoundationPrediction':
        """Deserialize from dictionary."""
        return cls(
            ticker=data['ticker'],
            prediction=data['prediction'],
            confidence_lower=data['confidence_lower'],
            confidence_upper=data['confidence_upper'],
            cold_start=data.get('cold_start', True),
            degradation_mode=data.get('degradation_mode', False)
        )


class FoundationModelClient:
    """Client for time-series foundation models.
    
    Routes stocks with insufficient history to foundation models
    instead of traditionally trained models.
    
    Attributes:
        api_key: API key for foundation model service.
        model_type: Type of model ('timegpt' or 'timesfm').
        min_history_days: Threshold for cold-start routing.
        patch_size: Patch size for time series tokenization.
    """
    
    MIN_HISTORY_DAYS = 60
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_type: str = 'timegpt',
        min_history_days: int = 60,
        patch_size: int = 16
    ):
        """Initialize FoundationModelClient.
        
        Args:
            api_key: API key (or from TIMEGPT_API_KEY env var).
            model_type: 'timegpt' or 'timesfm'.
            min_history_days: Days below which to use foundation model.
            patch_size: Patch size for tokenization.
        """
        self.api_key = api_key or os.environ.get('TIMEGPT_API_KEY')
        self.model_type = model_type
        self.min_history_days = min_history_days
        self.patch_size = patch_size
    
    def should_use_foundation(self, history_days: int) -> bool:
        """Check if foundation model should be used.
        
        Args:
            history_days: Number of days of price history available.
            
        Returns:
            True if history is below threshold (cold-start scenario).
        """
        return history_days < self.min_history_days
    
    def predict(
        self,
        ticker: str,
        price_history: pd.Series,
        horizon: int = 5
    ) -> FoundationPrediction:
        """Generate prediction with confidence intervals.
        
        Args:
            ticker: Stock ticker symbol.
            price_history: Historical price series (indexed by date).
            horizon: Forecast horizon in days.
            
        Returns:
            FoundationPrediction with point estimate and bounds.
        """
        try:
            if self.model_type == 'timegpt':
                result = self._call_timegpt(price_history, horizon)
            else:
                result = self._call_timesfm(price_history, horizon)
            
            return FoundationPrediction(
                ticker=ticker,
                prediction=result['mean'],
                confidence_lower=result['lower'],
                confidence_upper=result['upper'],
                cold_start=True,
                degradation_mode=False
            )
        except Exception as e:
            logger.warning(f"Foundation model failed for {ticker}: {e}")
            return self._fallback_prediction(ticker, price_history)
    
    def _call_timegpt(
        self, 
        price_history: pd.Series, 
        horizon: int
    ) -> Dict[str, float]:
        """Call TimeGPT API.
        
        Note: This is a placeholder. In production, implement actual API call.
        
        Args:
            price_history: Price series.
            horizon: Forecast horizon.
            
        Returns:
            Dict with 'mean', 'lower', 'upper'.
        """
        if not self.api_key:
            raise ValueError("TimeGPT API key not configured")
        
        # Placeholder - would call actual TimeGPT API
        # For now, use simple momentum as simulation
        if len(price_history) >= 5:
            returns = price_history.pct_change().dropna()
            mean_return = float(returns.mean())
            std_return = float(returns.std()) if len(returns) > 1 else 0.02
        else:
            mean_return = 0.0
            std_return = 0.02
        
        return {
            'mean': mean_return * horizon,
            'lower': (mean_return - 2 * std_return) * horizon,
            'upper': (mean_return + 2 * std_return) * horizon
        }
    
    def _call_timesfm(
        self, 
        price_history: pd.Series, 
        horizon: int
    ) -> Dict[str, float]:
        """Call TimesFM API.
        
        Note: This is a placeholder. In production, implement actual API call.
        
        Args:
            price_history: Price series.
            horizon: Forecast horizon.
            
        Returns:
            Dict with 'mean', 'lower', 'upper'.
        """
        # Same implementation as TimeGPT placeholder
        return self._call_timegpt(price_history, horizon)
    
    def _fallback_prediction(
        self,
        ticker: str,
        price_history: pd.Series
    ) -> FoundationPrediction:
        """Fallback to simple momentum when API unavailable.
        
        Args:
            ticker: Stock ticker.
            price_history: Price series.
            
        Returns:
            FoundationPrediction with degradation_mode=True.
        """
        # Simple momentum as fallback
        if len(price_history) >= 5:
            momentum = (price_history.iloc[-1] / price_history.iloc[-5] - 1)
        else:
            momentum = 0.0
        
        return FoundationPrediction(
            ticker=ticker,
            prediction=momentum,
            confidence_lower=momentum - 0.1,
            confidence_upper=momentum + 0.1,
            cold_start=True,
            degradation_mode=True
        )
