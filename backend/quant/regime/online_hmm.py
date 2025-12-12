"""Online Regime Detector Module.

Implements an online (incremental) Hidden Markov Model for market regime detection.
Uses exponential decay weighting to adapt to recent market conditions without
full batch retraining.

Key features:
- Incremental updates with new observations
- State persistence for recovery after restart
- Exponential decay for recency bias
- Graceful fallback on errors

Example:
    >>> detector = OnlineRegimeDetector(n_states=2, state_file="data/regime.json")
    >>> state, probs = detector.update(0.02)  # New market return
    >>> regime, probabilities = detector.get_regime()
    >>> print(f"Current regime: {regime}")
"""

from pathlib import Path
from typing import Optional, Tuple
import json
import logging

import numpy as np

logger = logging.getLogger(__name__)


class OnlineRegimeDetector:
    """Online learning regime detector with incremental updates.
    
    Uses a Gaussian mixture approach with online EM updates.
    State 0 corresponds to low volatility (Bull), State 1 to high volatility (Bear).
    
    Attributes:
        n_states: Number of hidden states (default 2: Bull/Bear).
        decay_factor: Exponential decay factor (0 < decay < 1).
        state_file: Path to persist model state.
    """
    
    def __init__(
        self,
        n_states: int = 2,
        decay_factor: float = 0.95,
        state_file: Optional[str] = None
    ):
        """Initialize OnlineRegimeDetector.
        
        Args:
            n_states: Number of hidden states (2 for Bull/Bear).
            decay_factor: Weight decay for exponential moving average (0.9-0.99).
            state_file: Optional path to persist state for recovery.
        """
        self.n_states = n_states
        self.decay_factor = decay_factor
        self.state_file = Path(state_file) if state_file else None
        
        # State estimates (running statistics)
        self._means = np.zeros(n_states)
        self._variances = np.ones(n_states) * 0.01  # Initial variance
        self._weights = np.ones(n_states) / n_states
        self._transition_matrix = np.ones((n_states, n_states)) / n_states
        
        self._current_state = 0
        self._state_probs = np.ones(n_states) / n_states
        self._n_updates = 0
        
        # Load persisted state if available
        if self.state_file and self.state_file.exists():
            self._load_state()
    
    def update(self, observation: float) -> Tuple[int, np.ndarray]:
        """Incrementally update with new observation.
        
        Uses online EM algorithm with exponential decay:
        1. E-step: Compute responsibilities (posterior probabilities)
        2. M-step: Update parameters with decay-weighted moving average
        
        Args:
            observation: New observation (e.g., daily market return).
            
        Returns:
            Tuple of (current_state_index, state_probabilities).
        """
        try:
            # E-step: compute responsibilities
            likelihoods = self._compute_likelihoods(observation)
            responsibilities = likelihoods * self._state_probs
            
            # Normalize (with numerical stability)
            resp_sum = responsibilities.sum()
            if resp_sum > 1e-10:
                responsibilities /= resp_sum
            else:
                responsibilities = np.ones(self.n_states) / self.n_states
            
            # M-step: update parameters with exponential decay
            for k in range(self.n_states):
                weight = responsibilities[k]
                
                # Update mean with decay
                self._means[k] = (
                    self.decay_factor * self._means[k] +
                    (1 - self.decay_factor) * weight * observation
                )
                
                # Update variance with decay
                diff_sq = (observation - self._means[k]) ** 2
                self._variances[k] = (
                    self.decay_factor * self._variances[k] +
                    (1 - self.decay_factor) * weight * diff_sq
                )
                # Ensure minimum variance for numerical stability
                self._variances[k] = max(self._variances[k], 1e-6)
            
            # Update state probabilities
            self._state_probs = responsibilities
            self._current_state = int(np.argmax(responsibilities))
            self._n_updates += 1
            
            # Persist state periodically (every 10 updates)
            if self.state_file and self._n_updates % 10 == 0:
                self._save_state()
            
            return self._current_state, self._state_probs.copy()
            
        except Exception as e:
            logger.warning(f"Online update failed: {e}, using last valid state")
            return self._current_state, self._state_probs.copy()
    
    def _compute_likelihoods(self, x: float) -> np.ndarray:
        """Compute Gaussian likelihoods for each state.
        
        Args:
            x: Observation value.
            
        Returns:
            Array of likelihood values for each state.
        """
        likelihoods = np.zeros(self.n_states)
        for k in range(self.n_states):
            var = self._variances[k]
            diff = x - self._means[k]
            # Gaussian PDF
            likelihoods[k] = np.exp(-0.5 * diff**2 / var) / np.sqrt(2 * np.pi * var)
        return likelihoods
    
    def get_regime(self) -> Tuple[str, np.ndarray]:
        """Get current regime label and probabilities.
        
        Identifies Bull/Bear based on mean returns:
        - Bull: state with higher mean return
        - Bear: state with lower mean return
        
        Returns:
            Tuple of (regime_label, state_probabilities).
        """
        # Identify bull state as the one with higher mean
        bull_state = int(np.argmax(self._means))
        
        if self._current_state == bull_state:
            regime = "Bull"
        else:
            regime = "Bear"
        
        return regime, self._state_probs.copy()
    
    def get_state_statistics(self) -> dict:
        """Get current state statistics.
        
        Returns:
            Dict with means, variances, and probabilities for each state.
        """
        return {
            'means': self._means.tolist(),
            'variances': self._variances.tolist(),
            'state_probs': self._state_probs.tolist(),
            'current_state': self._current_state,
            'n_updates': self._n_updates,
        }
    
    def get_risk_multiplier(self) -> float:
        """Get risk allocation multiplier based on current regime.
        
        Returns:
            1.0 in Bull, 0.5 in Bear, interpolated by probability otherwise.
        """
        regime, probs = self.get_regime()
        
        # Bull state probability determines risk allocation
        bull_state = int(np.argmax(self._means))
        bull_prob = probs[bull_state]
        
        # Interpolate between 0.5 (full bear) and 1.0 (full bull)
        return 0.5 + 0.5 * bull_prob
    
    def _save_state(self) -> None:
        """Persist model state to file."""
        if self.state_file is None:
            return
            
        state = {
            'means': self._means.tolist(),
            'variances': self._variances.tolist(),
            'weights': self._weights.tolist(),
            'state_probs': self._state_probs.tolist(),
            'current_state': self._current_state,
            'n_updates': self._n_updates
        }
        
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state, f)
            logger.debug(f"Saved online regime state: {self._n_updates} updates")
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")
    
    def _load_state(self) -> None:
        """Load model state from file."""
        if self.state_file is None or not self.state_file.exists():
            return
            
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            self._means = np.array(state['means'])
            self._variances = np.array(state['variances'])
            self._weights = np.array(state['weights'])
            self._state_probs = np.array(state['state_probs'])
            self._current_state = state['current_state']
            self._n_updates = state['n_updates']
            
            logger.info(f"Loaded online regime state: {self._n_updates} updates")
            
        except Exception as e:
            logger.warning(f"Failed to load state: {e}, using defaults")
            # Reset to defaults on failure
            self._means = np.zeros(self.n_states)
            self._variances = np.ones(self.n_states) * 0.01
            self._state_probs = np.ones(self.n_states) / self.n_states
    
    def reset(self) -> None:
        """Reset detector to initial state."""
        self._means = np.zeros(self.n_states)
        self._variances = np.ones(self.n_states) * 0.01
        self._weights = np.ones(self.n_states) / self.n_states
        self._state_probs = np.ones(self.n_states) / self.n_states
        self._current_state = 0
        self._n_updates = 0
        
        logger.info("Reset online regime detector")
