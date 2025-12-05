"""
Hidden Markov Model (HMM) Regime Detector

Uses Gaussian HMM to classify market into Bull/Bear regimes.
State 0: Low Volatility (Bull) - Risk-On
State 1: High Volatility (Bear) - Risk-Off

Used for dynamic factor weighting and risk management.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import hmmlearn, provide fallback
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not installed. Install with: pip install hmmlearn")


class RegimeDetector:
    """
    Detects market regime (Bull/Bear) using Hidden Markov Model.
    
    States:
        0: Low Volatility / Bull Market (Risk-On)
        1: High Volatility / Bear Market (Risk-Off)
    """
    
    def __init__(self, n_states: int = 2, lookback: int = 252):
        """
        Args:
            n_states: Number of hidden states (2 = Bull/Bear, 3 = Bull/Transition/Bear)
            lookback: Training window in trading days (default: 1 year)
        """
        self.n_states = n_states
        self.lookback = lookback
        self.model = None
        self._is_fitted = False
        
    def fit(self, returns: pd.Series) -> 'RegimeDetector':
        """
        Train the HMM on historical returns.
        
        Args:
            returns: Daily returns series
            
        Returns:
            self
        """
        if not HMM_AVAILABLE:
            logger.error("hmmlearn not available. Cannot fit HMM.")
            return self
        
        if len(returns) < self.lookback:
            logger.warning(f"Insufficient data for HMM: {len(returns)} < {self.lookback}")
            return self
        
        try:
            # Use most recent `lookback` days
            train_data = returns.iloc[-self.lookback:].values.reshape(-1, 1)
            
            # Remove NaN/Inf
            train_data = train_data[np.isfinite(train_data).flatten()]
            
            if len(train_data) < 50:
                logger.warning("Insufficient valid data points for HMM")
                return self
            
            # Initialize and fit Gaussian HMM
            self.model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            
            self.model.fit(train_data.reshape(-1, 1))
            self._is_fitted = True
            
            # Log state parameters
            for i in range(self.n_states):
                mean = self.model.means_[i][0]
                var = self.model.covars_[i][0][0]
                logger.info(f"HMM State {i}: Mean={mean:.4f}, Var={var:.6f}")
            
            # Identify which state is "Bull" (higher mean, lower var)
            self._identify_bull_state()
            
            return self
            
        except Exception as e:
            logger.error(f"HMM fitting failed: {e}")
            return self
    
    def _identify_bull_state(self):
        """Identify which HMM state corresponds to Bull market."""
        if not self._is_fitted:
            return
        
        # Bull = higher mean return and/or lower variance
        means = self.model.means_.flatten()
        variances = np.array([self.model.covars_[i][0][0] for i in range(self.n_states)])
        
        # Score each state: high mean + low variance = bull
        scores = means - 0.5 * variances  # Simple scoring
        
        self._bull_state = int(np.argmax(scores))
        self._bear_state = int(np.argmin(scores))
        
        logger.info(f"Identified Bull State: {self._bull_state}, Bear State: {self._bear_state}")
    
    def predict_regime(self, recent_returns: pd.Series) -> Tuple[str, np.ndarray]:
        """
        Predict current market regime.
        
        Args:
            recent_returns: Recent returns (at least 5 days)
            
        Returns:
            Tuple of (regime_label, probabilities)
        """
        if not self._is_fitted or self.model is None:
            return "Unknown", np.array([0.5, 0.5])
        
        try:
            # Use sliding window for prediction
            data = recent_returns.values.reshape(-1, 1)
            data = data[np.isfinite(data).flatten()].reshape(-1, 1)
            
            if len(data) < 5:
                return "Unknown", np.array([0.5, 0.5])
            
            # Decode hidden states
            hidden_states = self.model.predict(data)
            current_state = hidden_states[-1]
            
            # Get probabilities
            posteriors = self.model.predict_proba(data)
            current_probs = posteriors[-1]
            
            # Map to label
            if current_state == self._bull_state:
                regime = "Bull"
            elif current_state == self._bear_state:
                regime = "Bear"
            else:
                regime = "Transition"
            
            logger.debug(f"Current Regime: {regime}, Probs: {current_probs}")
            
            return regime, current_probs
            
        except Exception as e:
            logger.warning(f"Regime prediction failed: {e}")
            return "Unknown", np.array([0.5, 0.5])
    
    def get_risk_multiplier(self, recent_returns: pd.Series) -> float:
        """
        Get risk allocation multiplier based on regime.
        
        Returns:
            1.0 in Bull, 0.5 in Bear, 0.75 in Transition
        """
        regime, probs = self.predict_regime(recent_returns)
        
        if regime == "Bull":
            return 1.0
        elif regime == "Bear":
            return 0.5
        else:
            return 0.75


class DynamicFactorWeights:
    """
    Adjusts factor weights based on market regime.
    
    Bull: Overweight Momentum, Value
    Bear: Overweight Quality, Low Volatility
    """
    
    # Default weights by regime
    REGIME_WEIGHTS = {
        "Bull": {
            "vsm": 0.25,      # Momentum works in trends
            "bab": 0.08,      # BAB less important
            "qmj": 0.17,      # Quality always relevant
            "upside": 0.17,   # Value works in recoveries
            "pead": 0.13,     # P3: Earnings drift strong in bull
            "sentiment": 0.15,# P4: Sentiment matters in bull
            "accruals": 0.05, # Optional
            "ivol": 0.00      # Optional
        },
        "Bear": {
            "vsm": 0.08,      # Momentum crashes in reversals
            "bab": 0.22,      # Low beta protects
            "qmj": 0.30,      # Quality is defensive
            "upside": 0.13,   # Value traps possible
            "pead": 0.10,     # P3: Weaker in bear
            "sentiment": 0.12,# P4: Sentiment less reliable in bear
            "accruals": 0.05,
            "ivol": 0.00
        },
        "Transition": {
            "vsm": 0.17,
            "bab": 0.17,
            "qmj": 0.22,
            "upside": 0.17,
            "pead": 0.10,     # P3: Moderate
            "sentiment": 0.12,# P4: Moderate
            "accruals": 0.05,
            "ivol": 0.00
        },
        "Unknown": {
            "vsm": 0.20,
            "bab": 0.13,
            "qmj": 0.22,
            "upside": 0.17,
            "pead": 0.10,     # P3: Default
            "sentiment": 0.13,# P4: Default
            "accruals": 0.05,
            "ivol": 0.00
        }
    }
    
    @classmethod
    def get_weights(cls, regime: str) -> dict:
        """Get factor weights for the given regime."""
        return cls.REGIME_WEIGHTS.get(regime, cls.REGIME_WEIGHTS["Unknown"])
    
    @classmethod
    def blend_scores(
        cls, 
        factor_scores: dict, 
        regime: str
    ) -> float:
        """
        Compute regime-adjusted composite score.
        
        Args:
            factor_scores: Dict of factor name -> Z-score
            regime: Current market regime
            
        Returns:
            Weighted composite score
        """
        weights = cls.get_weights(regime)
        
        composite = 0.0
        total_weight = 0.0
        
        for factor, weight in weights.items():
            if factor in factor_scores and weight > 0:
                composite += weight * factor_scores[factor]
                total_weight += weight
        
        if total_weight > 0:
            composite /= total_weight
        
        return composite
