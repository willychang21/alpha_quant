"""Online Regime Detector Module.

Implements an online (incremental) Hidden Markov Model for market regime detection.
Uses exponential decay weighting to adapt to recent market conditions without
full batch retraining.

Key features:
- Incremental updates with new observations
- State persistence for recovery after restart
- Exponential decay for recency bias
- Graceful fallback on errors
- Concept drift detection with automatic adaptation

Example:
    >>> detector = OnlineRegimeDetector(n_states=2, state_file="data/regime.json")
    >>> state, probs = detector.update(0.02)  # New market return
    >>> regime, probabilities = detector.get_regime()
    >>> print(f"Current regime: {regime}")
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DriftEvent:
    """Represents a detected concept drift event.
    
    Attributes:
        timestamp: When the drift was detected.
        kl_divergence: Magnitude of the drift (KL divergence value).
        old_distribution: Error distribution before drift.
        new_distribution: Error distribution after drift.
    """
    timestamp: datetime
    kl_divergence: float
    old_distribution: np.ndarray
    new_distribution: np.ndarray
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary for JSON persistence."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'kl_divergence': self.kl_divergence,
            'old_distribution': self.old_distribution.tolist(),
            'new_distribution': self.new_distribution.tolist()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DriftEvent':
        """Deserialize from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            kl_divergence=data['kl_divergence'],
            old_distribution=np.array(data['old_distribution']),
            new_distribution=np.array(data['new_distribution'])
        )


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
        state_file: Optional[str] = None,
        drift_threshold: float = 0.1,
        drift_window: int = 20,
        adaptation_factor: float = 0.8
    ):
        """Initialize OnlineRegimeDetector.
        
        Args:
            n_states: Number of hidden states (2 for Bull/Bear).
            decay_factor: Weight decay for exponential moving average (0.9-0.99).
            state_file: Optional path to persist state for recovery.
            drift_threshold: KL divergence threshold for drift detection.
            drift_window: Window size for error history.
            adaptation_factor: Multiplier applied to decay_factor on drift (smaller = faster adapt).
        """
        self.n_states = n_states
        self.decay_factor = decay_factor
        self._base_decay_factor = decay_factor  # Store original for recovery
        self.state_file = Path(state_file) if state_file else None
        
        # Drift detection configuration
        self.drift_threshold = drift_threshold
        self.drift_window = drift_window
        self.adaptation_factor = adaptation_factor
        
        # State estimates (running statistics)
        self._means = np.zeros(n_states)
        self._variances = np.ones(n_states) * 0.01  # Initial variance
        self._weights = np.ones(n_states) / n_states
        self._transition_matrix = np.ones((n_states, n_states)) / n_states
        
        self._current_state = 0
        self._state_probs = np.ones(n_states) / n_states
        self._n_updates = 0
        
        # Drift detection state
        self._error_history: deque = deque(maxlen=drift_window)
        self._drift_events: List[DriftEvent] = []
        self._adaptation_countdown = 0
        
        # Load persisted state if available
        if self.state_file and self.state_file.exists():
            self._load_state()
    
    def update(self, observation: float) -> Tuple[int, np.ndarray]:
        """Incrementally update with new observation.
        
        Uses online EM algorithm with exponential decay:
        1. E-step: Compute responsibilities (posterior probabilities)
        2. Drift detection: Check for concept drift using KL divergence
        3. M-step: Update parameters with decay-weighted moving average
        
        Args:
            observation: New observation (e.g., daily market return).
            
        Returns:
            Tuple of (current_state_index, state_probabilities).
        """
        try:
            # Compute prediction error for drift detection
            predicted = self._get_prediction()
            error = observation - predicted
            self._error_history.append(error)
            
            # Check for drift if we have enough history
            if len(self._error_history) >= self.drift_window:
                drift_event = self._detect_drift()
                if drift_event:
                    self._drift_events.append(drift_event)
                    self._adapt_to_drift()
                    logger.info(
                        f"Concept drift detected: KL={drift_event.kl_divergence:.4f}, "
                        f"adapting learning rate"
                    )
            
            # Handle adaptation countdown (recover decay factor after drift)
            if self._adaptation_countdown > 0:
                self._adaptation_countdown -= 1
                if self._adaptation_countdown == 0:
                    self.decay_factor = self._base_decay_factor
                    logger.debug("Drift adaptation period ended, restored base decay factor")
            
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
    
    def _get_prediction(self) -> float:
        """Get expected value based on current state distribution.
        
        Returns:
            Weighted mean based on state probabilities.
        """
        return float(np.dot(self._state_probs, self._means))
    
    def _detect_drift(self) -> Optional[DriftEvent]:
        """Detect concept drift using KL divergence on error distribution.
        
        Splits the error history into old and new halves and computes
        KL divergence between their empirical distributions.
        
        Returns:
            DriftEvent if drift detected, None otherwise.
        """
        if len(self._error_history) < self.drift_window:
            return None
        
        errors = list(self._error_history)
        mid = len(errors) // 2
        old_errors = np.array(errors[:mid])
        new_errors = np.array(errors[mid:])
        
        # Compute KL divergence using histogram approximation
        kl_div = self._compute_kl_divergence(old_errors, new_errors)
        
        if kl_div > self.drift_threshold:
            return DriftEvent(
                timestamp=datetime.now(),
                kl_divergence=kl_div,
                old_distribution=old_errors,
                new_distribution=new_errors
            )
        return None
    
    def _compute_kl_divergence(
        self, 
        old_samples: np.ndarray, 
        new_samples: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Compute KL divergence between two sample distributions.
        
        Uses histogram approximation with smoothing for numerical stability.
        
        Args:
            old_samples: Samples from the old distribution.
            new_samples: Samples from the new distribution.
            n_bins: Number of histogram bins.
            
        Returns:
            KL divergence D(old || new).
        """
        # Create common bin edges
        all_samples = np.concatenate([old_samples, new_samples])
        min_val, max_val = all_samples.min(), all_samples.max()
        
        # Handle edge case where all samples are identical
        if max_val - min_val < 1e-10:
            return 0.0
        
        bins = np.linspace(min_val - 1e-10, max_val + 1e-10, n_bins + 1)
        
        # Compute histograms with Laplace smoothing
        old_hist, _ = np.histogram(old_samples, bins=bins)
        new_hist, _ = np.histogram(new_samples, bins=bins)
        
        # Add smoothing to avoid log(0)
        epsilon = 1e-10
        old_prob = (old_hist + epsilon) / (old_hist.sum() + epsilon * n_bins)
        new_prob = (new_hist + epsilon) / (new_hist.sum() + epsilon * n_bins)
        
        # KL divergence
        kl_div = np.sum(old_prob * np.log(old_prob / new_prob))
        
        return float(kl_div)
    
    def _adapt_to_drift(self) -> None:
        """Adapt to detected drift by temporarily increasing learning rate.
        
        Reduces decay_factor to allow faster adaptation to new distribution.
        Sets countdown for when to restore original decay_factor.
        """
        self.decay_factor = self._base_decay_factor * self.adaptation_factor
        self._adaptation_countdown = self.drift_window
        logger.debug(
            f"Decreased decay factor to {self.decay_factor:.3f} "
            f"for {self._adaptation_countdown} updates"
        )
    
    def get_drift_events(self) -> List[DriftEvent]:
        """Get list of detected drift events.
        
        Returns:
            List of DriftEvent objects.
        """
        return self._drift_events.copy()
    
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
        """Persist model state to file including drift detection state."""
        if self.state_file is None:
            return
            
        state = {
            'means': self._means.tolist(),
            'variances': self._variances.tolist(),
            'weights': self._weights.tolist(),
            'state_probs': self._state_probs.tolist(),
            'current_state': self._current_state,
            'n_updates': self._n_updates,
            # Drift detection state
            'error_history': list(self._error_history),
            'drift_events': [e.to_dict() for e in self._drift_events],
            'adaptation_countdown': self._adaptation_countdown,
            'decay_factor': self.decay_factor,
            'base_decay_factor': self._base_decay_factor,
        }
        
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state, f)
            logger.debug(f"Saved online regime state: {self._n_updates} updates")
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")
    
    def _load_state(self) -> None:
        """Load model state from file including drift detection state."""
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
            
            # Restore drift detection state if present
            if 'error_history' in state:
                self._error_history = deque(
                    state['error_history'], 
                    maxlen=self.drift_window
                )
                self._drift_events = [
                    DriftEvent.from_dict(e) for e in state.get('drift_events', [])
                ]
                self._adaptation_countdown = state.get('adaptation_countdown', 0)
                self.decay_factor = state.get('decay_factor', self._base_decay_factor)
                self._base_decay_factor = state.get('base_decay_factor', self._base_decay_factor)
            
            logger.info(
                f"Loaded online regime state: {self._n_updates} updates, "
                f"{len(self._drift_events)} drift events"
            )
            
        except Exception as e:
            logger.warning(f"Failed to load state: {e}, using defaults")
            # Reset to defaults on failure
            self._means = np.zeros(self.n_states)
            self._variances = np.ones(self.n_states) * 0.01
            self._state_probs = np.ones(self.n_states) / self.n_states
    
    def reset(self) -> None:
        """Reset detector to initial state including drift detection state."""
        self._means = np.zeros(self.n_states)
        self._variances = np.ones(self.n_states) * 0.01
        self._weights = np.ones(self.n_states) / self.n_states
        self._state_probs = np.ones(self.n_states) / self.n_states
        self._current_state = 0
        self._n_updates = 0
        
        # Reset drift detection state
        self._error_history.clear()
        self._drift_events = []
        self._adaptation_countdown = 0
        self.decay_factor = self._base_decay_factor
        
        logger.info("Reset online regime detector")
