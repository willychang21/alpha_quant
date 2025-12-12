"""Advanced Market Rotation System.

This module provides advanced market rotation analysis for quantitative trading:

Components:
- LevyRSCalculator: Levy Relative Strength for momentum screening
- MansfieldRSCalculator: Mansfield Relative Strength for breakout detection
- VolumeStructureAnalyzer: Volume analysis for institutional activity detection
- ScorecardSystem: Multi-factor scoring for signal generation
- AdvancedRotationFactor: Integration with RankingEngine

Data Models:
- LevyRSResult: Result of Levy RS calculation
- MansfieldRSResult: Result of Mansfield RS calculation
- VolumeAnalysisResult: Result of volume structure analysis
- ScorecardResult: Result of scorecard evaluation
- ScorecardWeights: Configurable weights for scorecard factors

Usage:
    from quant.rotation import AdvancedRotationFactor
    
    factor = AdvancedRotationFactor()
    score = factor.compute(history, None, ticker='AAPL', sector='Technology')
"""

from quant.rotation.models import (
    LevyRSResult,
    MansfieldRSResult,
    VolumeAnalysisResult,
    ScorecardResult,
)

from quant.rotation.levy_rs import LevyRSCalculator
from quant.rotation.mansfield_rs import MansfieldRSCalculator
from quant.rotation.volume_structure import VolumeStructureAnalyzer
from quant.rotation.scorecard import ScorecardSystem, ScorecardWeights
from quant.rotation.factor import AdvancedRotationFactor

__all__ = [
    # Data Models
    'LevyRSResult',
    'MansfieldRSResult',
    'VolumeAnalysisResult',
    'ScorecardResult',
    
    # Calculators
    'LevyRSCalculator',
    'MansfieldRSCalculator',
    'VolumeStructureAnalyzer',
    
    # Scorecard
    'ScorecardSystem',
    'ScorecardWeights',
    
    # Factor Integration
    'AdvancedRotationFactor',
]
