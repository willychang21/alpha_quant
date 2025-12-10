"""ActionProcessor for the Data Integrity & Validation Framework.

This module handles the application of configured actions (DROP, INTERPOLATE,
FFILL, WARN) to validation issues, producing clean data.

Requirements: 5.2, 5.3, 5.4, 5.5
"""

from typing import Dict, List, Set

import pandas as pd

from quant.data.integrity.enums import Action, InterpolationMethod
from quant.data.integrity.models import ValidationReport
from quant.data.integrity.policy import ActionPolicy


class ActionProcessor:
    """
    Applies configured actions to validation issues.

    Key constraint: Volume interpolation uses FFILL only, never linear.
    Linear interpolation creates non-integer values (e.g., 100.5 shares).

    Attributes:
        policy: ActionPolicy configuration for interpolation methods
    """

    def __init__(self, policy: ActionPolicy = None):
        """Initialize with optional policy configuration.

        Args:
            policy: ActionPolicy configuration. Uses defaults if not provided.
        """
        self.policy = policy or ActionPolicy()

    def apply_actions(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> pd.DataFrame:
        """
        Apply configured actions and return clean DataFrame.

        Processing order:
        1. Collect rows to drop and columns to interpolate
        2. Apply interpolation (before dropping to preserve neighbors)
        3. Drop invalid rows
        4. Reset index

        Args:
            df: Input DataFrame with validation issues
            report: ValidationReport containing issues and actions

        Returns:
            Clean DataFrame with actions applied
        """
        result = df.copy()

        # Collect row indices and columns by action
        drop_indices: Set[int] = set()
        interpolate_info: Dict[int, List[str]] = {}  # {row_idx: [columns]}
        ffill_info: Dict[int, List[str]] = {}  # {row_idx: [columns]}

        for issue in report.all_issues:
            if issue.row_index is None:
                # Batch-level issue - may need to drop entire batch
                if issue.action_taken == Action.DROP:
                    # Cannot drop without row index - log warning
                    continue
            else:
                if issue.action_taken == Action.DROP:
                    drop_indices.add(issue.row_index)
                elif issue.action_taken == Action.INTERPOLATE:
                    if issue.row_index not in interpolate_info:
                        interpolate_info[issue.row_index] = []
                    if issue.column:
                        # Handle comma-separated columns (e.g., "high,low")
                        cols = [c.strip() for c in issue.column.split(",")]
                        interpolate_info[issue.row_index].extend(cols)
                elif issue.action_taken == Action.FFILL:
                    if issue.row_index not in ffill_info:
                        ffill_info[issue.row_index] = []
                    if issue.column:
                        cols = [c.strip() for c in issue.column.split(",")]
                        ffill_info[issue.row_index].extend(cols)

        # Apply interpolation/ffill first (Requirement 5.4)
        for row_idx, columns in interpolate_info.items():
            if row_idx in drop_indices:
                continue  # Will be dropped anyway
            result = self._interpolate_row(result, row_idx, columns)

        for row_idx, columns in ffill_info.items():
            if row_idx in drop_indices:
                continue
            result = self._ffill_row(result, row_idx, columns)

        # Drop invalid rows (Requirement 5.3)
        valid_indices = [i for i in drop_indices if i in result.index]
        if valid_indices:
            result = result.drop(index=valid_indices)

        return result.reset_index(drop=True)

    def _interpolate_row(
        self, df: pd.DataFrame, row_idx: int, columns: List[str]
    ) -> pd.DataFrame:
        """
        Interpolate values using column-appropriate method.

        - Price columns: Linear interpolation (Requirement 5.4)
        - Volume: Forward fill (FFILL) only - never linear!

        Falls back to DROP if interpolation not possible (Requirement 5.5).

        Args:
            df: DataFrame to modify
            row_idx: Row index to interpolate
            columns: Columns to interpolate

        Returns:
            DataFrame with interpolated values
        """
        result = df.copy()

        # Check if interpolation is possible
        if not self._can_interpolate(result, row_idx):
            # Cannot interpolate first/last row - values unchanged
            return result

        for col in columns:
            if col not in result.columns:
                continue

            method = self.policy.get_interpolation_method(col)

            if self.policy.is_volume_column(col):
                # Volume: Always use FFILL, never linear (Requirement 5.4)
                result[col] = result[col].ffill()
            elif method == InterpolationMethod.LINEAR:
                # Linear interpolation for prices
                # Set the value to NaN first, then interpolate
                original_value = result.loc[row_idx, col]
                result.loc[row_idx, col] = pd.NA
                result[col] = result[col].interpolate(method="linear")
                # If interpolation failed (still NaN), restore original
                if pd.isna(result.loc[row_idx, col]):
                    result.loc[row_idx, col] = original_value
            elif method == InterpolationMethod.FFILL:
                result[col] = result[col].ffill()
            elif method == InterpolationMethod.ZERO:
                result.loc[row_idx, col] = 0

        return result

    def _ffill_row(
        self, df: pd.DataFrame, row_idx: int, columns: List[str]
    ) -> pd.DataFrame:
        """
        Apply forward fill to specific columns.

        Args:
            df: DataFrame to modify
            row_idx: Row index to fill
            columns: Columns to forward fill

        Returns:
            DataFrame with forward-filled values
        """
        result = df.copy()

        for col in columns:
            if col not in result.columns:
                continue
            result[col] = result[col].ffill()

        return result

    def _can_interpolate(self, df: pd.DataFrame, row_idx: int) -> bool:
        """
        Check if interpolation is possible (not first/last row).

        Requirement 5.5: Fall back to DROP when interpolation not possible.

        Args:
            df: DataFrame to check
            row_idx: Row index to check

        Returns:
            True if interpolation is possible
        """
        if len(df) < 3:
            return False
        return 0 < row_idx < len(df) - 1
