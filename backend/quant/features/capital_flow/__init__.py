"""
Capital Flow Detection Module.

Implements sector rotation analysis (RRG) and money flow indicators (MFI, OBV)
for detecting "Smart Money" movements.

Components:
- SectorRotationAnalyzer: RRG-based sector rotation analysis
- MoneyFlowCalculator: MFI and OBV calculations
- DivergenceDetector: Price-volume divergence detection
- CapitalFlowFactor: FeatureGenerator integration for RankingEngine
"""

from quant.features.capital_flow.models import (
    SectorRotationResult,
    DivergenceSignal,
    MoneyFlowResult,
)
from quant.features.capital_flow.money_flow import MoneyFlowCalculator
from quant.features.capital_flow.divergence import DivergenceDetector
from quant.features.capital_flow.sector_rotation import SectorRotationAnalyzer
from quant.features.capital_flow.capital_flow_factor import CapitalFlowFactor

__all__ = [
    'SectorRotationResult',
    'DivergenceSignal',
    'MoneyFlowResult',
    'MoneyFlowCalculator',
    'DivergenceDetector',
    'SectorRotationAnalyzer',
    'CapitalFlowFactor',
]
