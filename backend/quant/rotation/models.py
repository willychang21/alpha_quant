"""Data models for Advanced Market Rotation System.

Provides serializable dataclasses for:
- LevyRSResult: Levy Relative Strength calculation results
- MansfieldRSResult: Mansfield Relative Strength calculation results
- VolumeAnalysisResult: Volume structure analysis results
- ScorecardResult: Multi-factor scorecard evaluation results

All models support JSON serialization with precision preserved to 6 decimal places.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any
import json


@dataclass
class LevyRSResult:
    """Result of Levy RS calculation.
    
    Levy Relative Strength (RSL) = Close / SMA(Close, 26 weeks)
    Used for momentum screening and stock ranking.
    
    Attributes:
        ticker: Stock ticker symbol
        rsl: Levy Relative Strength value (typically 0.8 - 1.2)
        sma_26w: 26-week (130-day) simple moving average
        signal: Momentum signal ('strong', 'positive', 'breakdown', 'weak')
        percentile_rank: Rank among universe (0-100), optional
        timestamp: Calculation timestamp
    """
    ticker: str
    rsl: float
    sma_26w: float
    signal: str
    percentile_rank: Optional[float]
    timestamp: datetime
    
    def to_dict(self) -> dict:
        """Serialize to dictionary with precision preserved."""
        return {
            'ticker': self.ticker,
            'rsl': round(self.rsl, 6),
            'sma_26w': round(self.sma_26w, 6),
            'signal': self.signal,
            'percentile_rank': round(self.percentile_rank, 2) if self.percentile_rank is not None else None,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'LevyRSResult':
        """Deserialize from dictionary."""
        return cls(
            ticker=data['ticker'],
            rsl=data['rsl'],
            sma_26w=data['sma_26w'],
            signal=data['signal'],
            percentile_rank=data.get('percentile_rank'),
            timestamp=datetime.fromisoformat(data['timestamp'])
        )
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'LevyRSResult':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class MansfieldRSResult:
    """Result of Mansfield RS calculation.
    
    Mansfield Relative Strength (MRS) = ((RS / SMA(RS, 52)) - 1) × 100
    where RS = Stock Price / Index Price
    Used for identifying stocks breaking out relative to the market.
    
    Attributes:
        ticker: Stock ticker symbol
        mrs: Mansfield Relative Strength (centered around 0)
        mrs_slope: Rate of change of MRS (positive = improving)
        raw_rs: Raw relative strength (stock/benchmark)
        signal: Improvement signal ('breakout', 'improving', 'weakening', 'lagging')
        zero_crossover: True if MRS just crossed above zero
        timestamp: Calculation timestamp
    """
    ticker: str
    mrs: float
    mrs_slope: float
    raw_rs: float
    signal: str
    zero_crossover: bool
    timestamp: datetime
    
    def to_dict(self) -> dict:
        """Serialize to dictionary with precision preserved."""
        return {
            'ticker': self.ticker,
            'mrs': round(self.mrs, 6),
            'mrs_slope': round(self.mrs_slope, 6),
            'raw_rs': round(self.raw_rs, 6),
            'signal': self.signal,
            'zero_crossover': self.zero_crossover,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MansfieldRSResult':
        """Deserialize from dictionary."""
        return cls(
            ticker=data['ticker'],
            mrs=data['mrs'],
            mrs_slope=data['mrs_slope'],
            raw_rs=data['raw_rs'],
            signal=data['signal'],
            zero_crossover=data['zero_crossover'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MansfieldRSResult':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class VolumeAnalysisResult:
    """Result of volume structure analysis.
    
    Detects institutional activity through volume anomalies.
    
    Attributes:
        ticker: Stock ticker symbol
        volume_zscore: Volume Z-score for current period
        volume_classification: 'normal', 'elevated', or 'extreme'
        absorption_detected: True if absorption pattern detected
        rejection_detected: True if smart money rejection detected
        pattern: Overall pattern classification
        timestamp: Analysis timestamp
    """
    ticker: str
    volume_zscore: float
    volume_classification: str  # 'normal', 'elevated', 'extreme'
    absorption_detected: bool
    rejection_detected: bool
    pattern: str  # 'absorption', 'rejection', 'distribution', 'neutral'
    timestamp: datetime
    
    def to_dict(self) -> dict:
        """Serialize to dictionary with precision preserved."""
        return {
            'ticker': self.ticker,
            'volume_zscore': round(self.volume_zscore, 6),
            'volume_classification': self.volume_classification,
            'absorption_detected': self.absorption_detected,
            'rejection_detected': self.rejection_detected,
            'pattern': self.pattern,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'VolumeAnalysisResult':
        """Deserialize from dictionary."""
        return cls(
            ticker=data['ticker'],
            volume_zscore=data['volume_zscore'],
            volume_classification=data['volume_classification'],
            absorption_detected=data['absorption_detected'],
            rejection_detected=data['rejection_detected'],
            pattern=data['pattern'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'VolumeAnalysisResult':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class ScorecardResult:
    """Result of scorecard evaluation.
    
    Multi-factor scoring combining RRG position, RSL rank, MRS signal,
    and volume patterns to generate trading signals.
    
    Attributes:
        ticker: Stock ticker symbol
        factor_scores: Individual scores for each factor
        available_factors: List of factors with valid data
        total_score: Weighted total score in range [-1, 1]
        signal: Trading signal ('buy', 'sell', 'hold')
        confidence: Confidence level based on available factors (0-1)
        timestamp: Evaluation timestamp
    """
    ticker: str
    factor_scores: Dict[str, float]
    available_factors: List[str]
    total_score: float
    signal: str  # 'buy', 'sell', 'hold'
    confidence: float
    timestamp: datetime
    
    def to_dict(self) -> dict:
        """Serialize to dictionary with precision preserved."""
        return {
            'ticker': self.ticker,
            'factor_scores': {k: round(v, 6) for k, v in self.factor_scores.items()},
            'available_factors': self.available_factors,
            'total_score': round(self.total_score, 6),
            'signal': self.signal,
            'confidence': round(self.confidence, 6),
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ScorecardResult':
        """Deserialize from dictionary."""
        return cls(
            ticker=data['ticker'],
            factor_scores=data['factor_scores'],
            available_factors=data['available_factors'],
            total_score=data['total_score'],
            signal=data['signal'],
            confidence=data['confidence'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ScorecardResult':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
