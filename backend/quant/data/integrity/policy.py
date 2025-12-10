"""Action policy configuration for the Data Integrity & Validation Framework.

This module defines the ActionPolicy dataclass that configures how different
types of validation failures are handled.
"""

from dataclasses import dataclass, field
from typing import Dict

from quant.data.integrity.enums import Action, InterpolationMethod


@dataclass
class ActionPolicy:
    """
    Configuration for handling validation failures.

    Note: Volume interpolation is restricted to FFILL or ZERO.
    Linear interpolation creates non-integer values (e.g., 100.5 shares).

    Attributes:
        structural_failure: Action for missing columns/type errors (default: DROP)
        logical_failure: Action for OHLCV rule violations (default: DROP)
        temporal_gap_single: Action for single missing day (default: INTERPOLATE)
        temporal_gap_multi: Action for multi-day gaps (default: WARN)
        outlier_spike: Action for confirmed spikes in BACKFILL (default: DROP)
        outlier_potential_spike: Action for potential spikes in DAILY (default: WARN)
        outlier_persistent: Action for persistent moves (default: WARN)
        zero_volume: Action for zero volume records (default: DROP)
        interpolation_methods: Column-specific interpolation methods
        sigma_threshold: Standard deviation threshold for outliers (default: 5.0)
        pct_threshold: Percentage threshold for outliers (default: 0.50 = 50%)
        max_drop_rate: Maximum acceptable drop rate (default: 0.10 = 10%)
    """

    structural_failure: Action = Action.DROP
    logical_failure: Action = Action.DROP
    temporal_gap_single: Action = Action.INTERPOLATE
    temporal_gap_multi: Action = Action.WARN
    outlier_spike: Action = Action.DROP
    outlier_potential_spike: Action = Action.WARN  # DAILY context only
    outlier_persistent: Action = Action.WARN
    zero_volume: Action = Action.DROP  # Zero volume = invalid data


    # Interpolation methods by column
    # Price columns use LINEAR, volume uses FFILL (never linear!)
    interpolation_methods: Dict[str, InterpolationMethod] = field(
        default_factory=lambda: {
            "open": InterpolationMethod.LINEAR,
            "high": InterpolationMethod.LINEAR,
            "low": InterpolationMethod.LINEAR,
            "close": InterpolationMethod.LINEAR,
            "adj_close": InterpolationMethod.LINEAR,
            "volume": InterpolationMethod.FFILL,  # Never linear!
        }
    )

    # Thresholds
    sigma_threshold: float = 5.0  # Standard deviations for outlier detection
    pct_threshold: float = 0.50  # 50% absolute return threshold
    max_drop_rate: float = 0.10  # 10% maximum acceptable drop rate

    def get_interpolation_method(self, column: str) -> InterpolationMethod:
        """Get the interpolation method for a specific column.

        Args:
            column: Column name

        Returns:
            InterpolationMethod for the column, defaults to LINEAR for unknown columns
        """
        return self.interpolation_methods.get(column, InterpolationMethod.LINEAR)

    def is_volume_column(self, column: str) -> bool:
        """Check if a column is the volume column.

        Volume columns should never use linear interpolation.

        Args:
            column: Column name

        Returns:
            True if this is the volume column
        """
        return column.lower() == "volume"
