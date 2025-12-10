"""Pandera schema definitions for the Data Integrity & Validation Framework.

This module defines the OHLCVSchema using Pandera's DataFrameModel for
declarative validation of market data with clear error messages.

Requirements: 1.1, 2.1, 2.2, 2.3, 2.4, 2.5, 7.1, 7.3
"""

from typing import Optional

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series


# Schema version for backward compatibility tracking (Requirement 7.4)
SCHEMA_VERSION = "1.0.0"


class OHLCVSchema(pa.DataFrameModel):
    """
    Pandera schema for OHLCV market data.

    This schema validates:
    - Required columns presence (Requirement 1.1)
    - Data types with coercion (Requirement 1.3)
    - Strictly positive prices (Requirement 2.5)
    - Strictly positive volume (Requirement 2.4)
    - OHLCV logical relationships (Requirements 2.1, 2.2, 2.3)

    Note: Volume is strictly positive (gt=0) for active tickers.
    Zero volume typically indicates missing/invalid data from yfinance.
    """

    # Required columns with types and constraints
    ticker: Series[str] = pa.Field(nullable=False, description="Stock ticker symbol")
    date: Series[pa.DateTime] = pa.Field(
        nullable=False, description="Trading date", coerce=True
    )
    open: Series[float] = pa.Field(
        gt=0, nullable=False, description="Opening price", coerce=True
    )
    high: Series[float] = pa.Field(
        gt=0, nullable=False, description="High price", coerce=True
    )
    low: Series[float] = pa.Field(
        gt=0, nullable=False, description="Low price", coerce=True
    )
    close: Series[float] = pa.Field(
        gt=0, nullable=False, description="Closing price", coerce=True
    )
    adj_close: Series[float] = pa.Field(
        gt=0, nullable=False, description="Adjusted closing price", coerce=True
    )
    volume: Series[int] = pa.Field(
        gt=0, nullable=False, description="Trading volume (strictly positive)", coerce=True
    )

    class Config:
        """Schema configuration."""

        # Strict mode: reject extra columns
        strict = "filter"  # Filter out extra columns instead of raising error
        # Enable type coercion (Requirement 1.3)
        coerce = True
        # Schema name for error messages
        name = "OHLCVSchema"
        # Ordered columns
        ordered = False

    @pa.dataframe_check
    def ohlcv_relationships(cls, df: DataFrame) -> Series[bool]:
        """
        Validate OHLCV logical relationships.

        Requirements 2.1, 2.2, 2.3:
        - High >= Low
        - High >= Open AND High >= Close
        - Low <= Open AND Low <= Close

        Returns:
            Boolean series indicating valid rows
        """
        return (
            (df["high"] >= df["low"])
            & (df["high"] >= df["open"])
            & (df["high"] >= df["close"])
            & (df["low"] <= df["open"])
            & (df["low"] <= df["close"])
        )


def validate_ohlcv(
    df: pd.DataFrame, lazy: bool = True
) -> tuple[pd.DataFrame, Optional[pa.errors.SchemaErrors]]:
    """
    Validate a DataFrame against the OHLCVSchema.

    Args:
        df: Input DataFrame to validate
        lazy: If True, collect all errors; if False, fail on first error
              (Requirement 7.5)

    Returns:
        Tuple of (validated DataFrame, errors if any)

    Raises:
        pa.errors.SchemaError: If lazy=False and validation fails
        pa.errors.SchemaErrors: If lazy=True and validation fails
    """
    try:
        validated_df = OHLCVSchema.validate(df, lazy=lazy)
        return validated_df, None
    except pa.errors.SchemaErrors as e:
        # Lazy mode: multiple errors collected
        return df, e
    except pa.errors.SchemaError as e:
        # Eager mode: single error
        # Wrap in SchemaErrors for consistent handling
        return df, pa.errors.SchemaErrors(
            schema_errors=[e],
            data=df,
        )


def get_required_columns() -> list[str]:
    """
    Get the list of required columns for OHLCV data.

    Returns:
        List of required column names
    """
    return ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]


def get_schema_version() -> str:
    """
    Get the current schema version.

    Returns:
        Schema version string
    """
    return SCHEMA_VERSION
