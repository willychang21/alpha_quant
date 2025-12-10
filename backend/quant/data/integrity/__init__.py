"""Data Integrity & Validation Framework.

This module provides a robust validation layer for market data ingested from yfinance.
It addresses common data quality issues including missing data, price spikes,
logical errors, and adjustment anomalies.

Key features:
- Context-aware validation (DAILY vs BACKFILL modes)
- Pandera-based schema validation
- Configurable action policies
- Smart catch-up for gap detection and backfill
"""

from quant.data.integrity.catchup import SmartCatchUpService
from quant.data.integrity.enums import (
    Action,
    InterpolationMethod,
    ValidationContext,
)
from quant.data.integrity.models import (
    ValidationIssue,
    ValidationReport,
)
from quant.data.integrity.policy import ActionPolicy
from quant.data.integrity.processor import ActionProcessor
from quant.data.integrity.schema import (
    SCHEMA_VERSION,
    OHLCVSchema,
    get_required_columns,
    get_schema_version,
    validate_ohlcv,
)
from quant.data.integrity.validator import (
    DataValidator,
    OHLCVValidator,
)

__all__ = [
    # Enums
    "ValidationContext",
    "Action",
    "InterpolationMethod",
    # Models
    "ValidationIssue",
    "ValidationReport",
    # Policy
    "ActionPolicy",
    # Processor
    "ActionProcessor",
    # CatchUp
    "SmartCatchUpService",
    # Schema
    "OHLCVSchema",
    "SCHEMA_VERSION",
    "validate_ohlcv",
    "get_required_columns",
    "get_schema_version",
    # Validators
    "DataValidator",
    "OHLCVValidator",
]


