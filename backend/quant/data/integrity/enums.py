"""Enums for the Data Integrity & Validation Framework.

This module defines the core enumerations used throughout the validation
framework for context-aware validation, action policies, and interpolation methods.
"""

from enum import Enum


class ValidationContext(Enum):
    """
    Context for validation operations.

    DAILY: Real-time validation during daily updates.
           - Cannot verify spike reversion (no T+1 data)
           - Flags "potential spikes" for review

    BACKFILL: Historical validation during catch-up or backfill.
              - Can verify spike reversion using T+1 data
              - Confirms "spike" vs "persistent move"
    """

    DAILY = "daily"
    BACKFILL = "backfill"


class Action(Enum):
    """Actions to take when validation issues are detected."""

    WARN = "warn"
    DROP = "drop"
    INTERPOLATE = "interpolate"
    FFILL = "ffill"  # Forward fill (for volume only)


class InterpolationMethod(Enum):
    """Interpolation methods by column type.

    LINEAR: For price columns only (open, high, low, close, adj_close)
    FFILL: For volume (forward fill) - never use linear for volume
    ZERO: Fill with zero
    """

    LINEAR = "linear"  # For price columns only
    FFILL = "ffill"  # For volume (forward fill)
    ZERO = "zero"  # Fill with zero
