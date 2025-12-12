"""Scorecard System for Advanced Market Rotation.

Multi-factor scoring system that combines:
- RRG position (from Capital Flow Detection)
- RSL rank (Levy Relative Strength)
- MRS signal (Mansfield Relative Strength)
- Volume pattern (absorption/rejection)
- Fundamental momentum (reserved for future)

Generates trading signals based on configurable thresholds.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import logging

from quant.rotation.models import ScorecardResult

logger = logging.getLogger(__name__)


@dataclass
class ScorecardWeights:
    """Configurable weights for scorecard factors.
    
    Weights should sum to 1.0 for proper normalization.
    
    Attributes:
        rrg_position: Weight for RRG quadrant position (default: 0.25)
        rsl_rank: Weight for Levy RSL percentile (default: 0.20)
        mrs_signal: Weight for Mansfield RS signal (default: 0.20)
        volume_pattern: Weight for volume pattern detection (default: 0.20)
        fundamental_momentum: Weight for earnings momentum (default: 0.15, reserved)
    """
    rrg_position: float = 0.25
    rsl_rank: float = 0.20
    mrs_signal: float = 0.20
    volume_pattern: float = 0.20
    fundamental_momentum: float = 0.15  # Reserved for future use
    
    def validate(self) -> bool:
        """Ensure weights sum to 1.0 within tolerance.
        
        Returns:
            True if weights sum is within 0.001 of 1.0
        """
        total = (
            self.rrg_position +
            self.rsl_rank +
            self.mrs_signal +
            self.volume_pattern +
            self.fundamental_momentum
        )
        return abs(total - 1.0) < 0.001
    
    def get_weight(self, factor_name: str) -> float:
        """Get weight for a specific factor.
        
        Args:
            factor_name: Name of the factor
            
        Returns:
            Weight value or 0.0 if factor not found
        """
        weight_map = {
            'rrg_position': self.rrg_position,
            'rsl_rank': self.rsl_rank,
            'mrs_signal': self.mrs_signal,
            'volume_pattern': self.volume_pattern,
            'fundamental_momentum': self.fundamental_momentum
        }
        return weight_map.get(factor_name, 0.0)
    
    def get_all_weights(self) -> Dict[str, float]:
        """Get all weights as a dictionary.
        
        Returns:
            Dictionary mapping factor names to weights
        """
        return {
            'rrg_position': self.rrg_position,
            'rsl_rank': self.rsl_rank,
            'mrs_signal': self.mrs_signal,
            'volume_pattern': self.volume_pattern,
            'fundamental_momentum': self.fundamental_momentum
        }


class ScorecardSystem:
    """Multi-factor scorecard for generating trading signals.
    
    Combines multiple factors into a composite score and
    generates buy/sell/hold signals based on thresholds.
    
    Attributes:
        weights: ScorecardWeights configuration
        buy_threshold: Score threshold for buy signal (default: 0.6)
        sell_threshold: Score threshold for sell signal (default: -0.3)
    """
    
    # Score mappings for categorical factors
    RRG_SCORES = {
        'Leading': 1.0,
        'Improving': 0.5,
        'Weakening': -0.5,
        'Lagging': -1.0,
        'Unknown': 0.0
    }
    
    MRS_SCORES = {
        'breakout': 1.0,
        'improving': 0.5,
        'weakening': -0.5,
        'lagging': -1.0
    }
    
    VOLUME_SCORES = {
        'rejection': 1.0,    # Bullish reversal
        'absorption': 0.5,   # Accumulation
        'distribution': -0.8,  # Bearish
        'neutral': 0.0
    }
    
    def __init__(
        self,
        weights: Optional[ScorecardWeights] = None,
        buy_threshold: float = 0.6,
        sell_threshold: float = -0.3
    ):
        """Initialize Scorecard System.
        
        Args:
            weights: Factor weights configuration (uses defaults if None)
            buy_threshold: Score threshold for buy signal (default: 0.6)
            sell_threshold: Score threshold for sell signal (default: -0.3)
        """
        self.weights = weights or ScorecardWeights()
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        
        if not self.weights.validate():
            logger.warning(
                "Scorecard weights do not sum to 1.0. "
                "Scores will be normalized but may have unexpected behavior."
            )
    
    def compute_factor_scores(
        self,
        rrg_quadrant: Optional[str] = None,
        rsl_percentile: Optional[float] = None,
        mrs_signal: Optional[str] = None,
        volume_pattern: Optional[str] = None,
        fundamental_momentum: Optional[float] = None
    ) -> Dict[str, float]:
        """Compute individual factor scores normalized to [-1, 1].
        
        Args:
            rrg_quadrant: 'Leading', 'Weakening', 'Lagging', 'Improving', or 'Unknown'
            rsl_percentile: RSL percentile rank (0-100)
            mrs_signal: 'breakout', 'improving', 'weakening', 'lagging'
            volume_pattern: 'absorption', 'rejection', 'distribution', 'neutral'
            fundamental_momentum: Earnings revision breadth (-1 to 1), reserved
            
        Returns:
            Dictionary of factor -> score (each in [-1, 1])
        """
        scores = {}
        
        # RRG position score
        if rrg_quadrant is not None:
            scores['rrg_position'] = self.RRG_SCORES.get(rrg_quadrant, 0.0)
        
        # RSL rank score: convert percentile (0-100) to [-1, 1]
        if rsl_percentile is not None:
            # Higher percentile = better, so map 0->-1, 50->0, 100->1
            scores['rsl_rank'] = (rsl_percentile - 50) / 50
        
        # MRS signal score
        if mrs_signal is not None:
            scores['mrs_signal'] = self.MRS_SCORES.get(mrs_signal, 0.0)
        
        # Volume pattern score
        if volume_pattern is not None:
            scores['volume_pattern'] = self.VOLUME_SCORES.get(volume_pattern, 0.0)
        
        # Fundamental momentum (pass-through, already in [-1, 1])
        if fundamental_momentum is not None:
            scores['fundamental_momentum'] = max(-1.0, min(1.0, fundamental_momentum))
        
        return scores
    
    def compute_total_score(
        self,
        factor_scores: Dict[str, float],
        available_factors: Optional[List[str]] = None
    ) -> float:
        """Compute weighted total score, adjusting for missing factors.
        
        Args:
            factor_scores: Dictionary of factor -> score
            available_factors: List of factors with valid data (inferred if None)
            
        Returns:
            Total weighted score in range [-1, 1]
        """
        if not factor_scores:
            return 0.0
        
        # Infer available factors if not provided
        if available_factors is None:
            available_factors = list(factor_scores.keys())
        
        # Calculate weighted sum and total available weight
        weighted_sum = 0.0
        total_weight = 0.0
        
        for factor in available_factors:
            if factor in factor_scores:
                weight = self.weights.get_weight(factor)
                weighted_sum += factor_scores[factor] * weight
                total_weight += weight
        
        # Normalize by available weight to maintain [-1, 1] range
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.0
    
    def generate_signal(self, total_score: float) -> str:
        """Generate trading signal based on total score.
        
        Args:
            total_score: Weighted total score
            
        Returns:
            'buy' if score > buy_threshold
            'sell' if score < sell_threshold
            'hold' otherwise
        """
        if total_score > self.buy_threshold:
            return 'buy'
        elif total_score < self.sell_threshold:
            return 'sell'
        else:
            return 'hold'
    
    def compute_confidence(self, available_factors: List[str]) -> float:
        """Compute confidence level based on data availability.
        
        Args:
            available_factors: List of factors with valid data
            
        Returns:
            Confidence score (0-1) based on weight coverage
        """
        total_available_weight = sum(
            self.weights.get_weight(f) for f in available_factors
        )
        total_possible_weight = sum(self.weights.get_all_weights().values())
        
        if total_possible_weight > 0:
            return total_available_weight / total_possible_weight
        return 0.0
    
    def evaluate(
        self,
        ticker: str,
        rrg_quadrant: Optional[str] = None,
        rsl_percentile: Optional[float] = None,
        mrs_signal: Optional[str] = None,
        volume_pattern: Optional[str] = None,
        fundamental_momentum: Optional[float] = None
    ) -> ScorecardResult:
        """Evaluate a stock and generate complete scorecard result.
        
        Args:
            ticker: Stock ticker symbol
            rrg_quadrant: RRG quadrant position
            rsl_percentile: RSL percentile rank
            mrs_signal: MRS improvement signal
            volume_pattern: Volume pattern classification
            fundamental_momentum: Earnings momentum (reserved)
            
        Returns:
            ScorecardResult with all computed values and signal
        """
        # Compute individual factor scores
        factor_scores = self.compute_factor_scores(
            rrg_quadrant=rrg_quadrant,
            rsl_percentile=rsl_percentile,
            mrs_signal=mrs_signal,
            volume_pattern=volume_pattern,
            fundamental_momentum=fundamental_momentum
        )
        
        # Determine available factors
        available_factors = list(factor_scores.keys())
        
        # Log missing factors
        all_factors = ['rrg_position', 'rsl_rank', 'mrs_signal', 'volume_pattern', 'fundamental_momentum']
        missing_factors = [f for f in all_factors if f not in available_factors]
        if missing_factors:
            logger.debug(f"Missing factors for {ticker}: {missing_factors}")
        
        # Compute total score
        total_score = self.compute_total_score(factor_scores, available_factors)
        
        # Generate signal
        signal = self.generate_signal(total_score)
        
        # Compute confidence
        confidence = self.compute_confidence(available_factors)
        
        return ScorecardResult(
            ticker=ticker,
            factor_scores=factor_scores,
            available_factors=available_factors,
            total_score=total_score,
            signal=signal,
            confidence=confidence,
            timestamp=datetime.now()
        )
