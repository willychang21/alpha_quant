"""
Data Models for Capital Flow Detection System.

Provides dataclasses with JSON serialization for:
- SectorRotationResult: RRG sector rotation analysis result
- DivergenceSignal: Price-indicator divergence detection result
- MoneyFlowResult: Money flow analysis result for a stock
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import json


@dataclass
class SectorRotationResult:
    """Result of sector rotation analysis using RRG methodology.
    
    Attributes:
        symbol: Sector ETF symbol (e.g., 'XLK')
        sector_name: Human-readable sector name (e.g., 'Technology')
        rs_ratio: Relative Strength Ratio (normalized around 100)
        rs_momentum: Rate of change of RS Ratio
        quadrant: Current RRG quadrant ('Leading', 'Weakening', 'Lagging', 'Improving')
        previous_quadrant: Previous quadrant for transition detection
        transition_signal: True if quadrant changed (especially Lagging -> Improving)
        timestamp: Analysis timestamp
    """
    symbol: str
    sector_name: str
    rs_ratio: float
    rs_momentum: float
    quadrant: str
    previous_quadrant: Optional[str] = None
    transition_signal: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary with JSON-compatible types."""
        return {
            'symbol': self.symbol,
            'sector_name': self.sector_name,
            'rs_ratio': round(self.rs_ratio, 6),
            'rs_momentum': round(self.rs_momentum, 6),
            'quadrant': self.quadrant,
            'previous_quadrant': self.previous_quadrant,
            'transition_signal': self.transition_signal,
            'timestamp': self.timestamp.isoformat(),
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SectorRotationResult':
        """Deserialize from dictionary."""
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data).__name__}")
        
        required_fields = ['symbol', 'sector_name', 'rs_ratio', 'rs_momentum', 'quadrant']
        for field_name in required_fields:
            if field_name not in data:
                raise ValueError(f"Missing required field: {field_name}")
        
        timestamp = data.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()
        
        return cls(
            symbol=data['symbol'],
            sector_name=data['sector_name'],
            rs_ratio=float(data['rs_ratio']),
            rs_momentum=float(data['rs_momentum']),
            quadrant=data['quadrant'],
            previous_quadrant=data.get('previous_quadrant'),
            transition_signal=data.get('transition_signal', False),
            timestamp=timestamp,
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SectorRotationResult':
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class DivergenceSignal:
    """Divergence detection result between price and indicator.
    
    Attributes:
        divergence_type: 'bullish', 'bearish', or 'none'
        confidence: Confidence score from 0.0 to 1.0
        price_swing: Tuple of (previous swing, current swing) for price
        indicator_swing: Tuple of (previous swing, current swing) for indicator
        lookback_bars: Number of bars used for detection
    """
    divergence_type: str  # 'bullish', 'bearish', 'none'
    confidence: float  # 0.0 to 1.0
    price_swing: Tuple[float, float] = field(default=(0.0, 0.0))
    indicator_swing: Tuple[float, float] = field(default=(0.0, 0.0))
    lookback_bars: int = 20
    
    def __post_init__(self):
        """Validate fields after initialization."""
        valid_types = ('bullish', 'bearish', 'none')
        if self.divergence_type not in valid_types:
            raise ValueError(f"divergence_type must be one of {valid_types}")
        
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'divergence_type': self.divergence_type,
            'confidence': round(self.confidence, 6),
            'price_swing': [round(v, 6) for v in self.price_swing],
            'indicator_swing': [round(v, 6) for v in self.indicator_swing],
            'lookback_bars': self.lookback_bars,
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DivergenceSignal':
        """Deserialize from dictionary."""
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data).__name__}")
        
        price_swing = data.get('price_swing', (0.0, 0.0))
        if isinstance(price_swing, list):
            price_swing = tuple(price_swing)
        
        indicator_swing = data.get('indicator_swing', (0.0, 0.0))
        if isinstance(indicator_swing, list):
            indicator_swing = tuple(indicator_swing)
        
        return cls(
            divergence_type=data.get('divergence_type', 'none'),
            confidence=float(data.get('confidence', 0.0)),
            price_swing=price_swing,
            indicator_swing=indicator_swing,
            lookback_bars=int(data.get('lookback_bars', 20)),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'DivergenceSignal':
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class MoneyFlowResult:
    """Money flow analysis result for a stock.
    
    Attributes:
        ticker: Stock ticker symbol
        mfi: Money Flow Index value (0-100)
        mfi_signal: 'oversold' (<20), 'overbought' (>80), or 'neutral'
        obv_zscore: Z-score normalized On-Balance Volume
        obv_trend: 'accumulation', 'distribution', or 'neutral'
        divergence_score: Composite divergence score from -1 to 1
        composite_score: Combined money flow score
        timestamp: Analysis timestamp
    """
    ticker: str
    mfi: float
    mfi_signal: str  # 'oversold', 'overbought', 'neutral'
    obv_zscore: float
    obv_trend: str  # 'accumulation', 'distribution', 'neutral'
    divergence_score: float  # -1 to 1
    composite_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate fields after initialization."""
        valid_mfi_signals = ('oversold', 'overbought', 'neutral')
        if self.mfi_signal not in valid_mfi_signals:
            raise ValueError(f"mfi_signal must be one of {valid_mfi_signals}")
        
        valid_obv_trends = ('accumulation', 'distribution', 'neutral')
        if self.obv_trend not in valid_obv_trends:
            raise ValueError(f"obv_trend must be one of {valid_obv_trends}")
        
        # Clamp values to valid ranges
        self.mfi = max(0.0, min(100.0, self.mfi))
        self.divergence_score = max(-1.0, min(1.0, self.divergence_score))
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'ticker': self.ticker,
            'mfi': round(self.mfi, 6),
            'mfi_signal': self.mfi_signal,
            'obv_zscore': round(self.obv_zscore, 6),
            'obv_trend': self.obv_trend,
            'divergence_score': round(self.divergence_score, 6),
            'composite_score': round(self.composite_score, 6),
            'timestamp': self.timestamp.isoformat(),
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MoneyFlowResult':
        """Deserialize from dictionary."""
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data).__name__}")
        
        timestamp = data.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()
        
        return cls(
            ticker=data['ticker'],
            mfi=float(data['mfi']),
            mfi_signal=data['mfi_signal'],
            obv_zscore=float(data['obv_zscore']),
            obv_trend=data['obv_trend'],
            divergence_score=float(data['divergence_score']),
            composite_score=float(data['composite_score']),
            timestamp=timestamp,
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MoneyFlowResult':
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


def serialize_results_to_json(results: list) -> str:
    """Serialize a list of results to JSON.
    
    Args:
        results: List of SectorRotationResult, DivergenceSignal, or MoneyFlowResult
        
    Returns:
        JSON string representation
    """
    data = [r.to_dict() for r in results]
    return json.dumps(data, indent=2)


def deserialize_sector_results_from_json(json_str: str) -> list:
    """Deserialize a list of SectorRotationResult from JSON.
    
    Args:
        json_str: JSON string representation
        
    Returns:
        List of SectorRotationResult objects
    """
    data = json.loads(json_str)
    return [SectorRotationResult.from_dict(d) for d in data]
