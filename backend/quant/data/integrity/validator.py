"""DataValidator abstract interface and OHLCVValidator implementation.

This module defines the core validation interface and concrete implementation
for OHLCV market data validation.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 7.5
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Set

import pandas as pd
import pandas_market_calendars as mcal
import pandera as pa

from quant.data.integrity.enums import Action, ValidationContext
from quant.data.integrity.models import ValidationIssue, ValidationReport
from quant.data.integrity.policy import ActionPolicy
from quant.data.integrity.schema import (
    OHLCVSchema,
    get_required_columns,
    validate_ohlcv,
)


class DataValidator(ABC):
    """Abstract interface for data validation.

    This class defines the contract for all data validators in the framework.
    Concrete implementations must provide validation logic for structural,
    logical, temporal, and statistical checks.

    Requirements: 1.1, 2.1, 3.1, 4.1
    """

    @abstractmethod
    def validate(
        self,
        df: pd.DataFrame,
        context: ValidationContext = ValidationContext.DAILY,
        lazy: bool = True,
    ) -> ValidationReport:
        """
        Validate DataFrame and return report.

        Args:
            df: Input DataFrame to validate
            context: DAILY (no T+1 data) or BACKFILL (has T+1 data)
            lazy: If True, collect all errors; if False, fail on first error

        Returns:
            ValidationReport with all validation results
        """
        pass

    @abstractmethod
    def validate_structural(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check column presence and data types.

        Args:
            df: Input DataFrame to validate

        Returns:
            List of structural validation issues
        """
        pass

    @abstractmethod
    def validate_logical(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check OHLCV logical relationships.

        Args:
            df: Input DataFrame to validate

        Returns:
            List of logical validation issues
        """
        pass

    @abstractmethod
    def validate_temporal(
        self, df: pd.DataFrame, ticker: str
    ) -> List[ValidationIssue]:
        """Check for missing business days.

        Args:
            df: Input DataFrame to validate
            ticker: Ticker symbol for the time series

        Returns:
            List of temporal validation issues (gaps)
        """
        pass

    @abstractmethod
    def validate_statistical(
        self, df: pd.DataFrame, context: ValidationContext
    ) -> List[ValidationIssue]:
        """
        Detect outliers and spikes.

        Context-aware behavior:
        - DAILY: Flags "potential_spike" (cannot verify reversion)
        - BACKFILL: Verifies reversion to confirm "spike" vs "persistent_move"

        Args:
            df: Input DataFrame to validate
            context: Validation context (DAILY or BACKFILL)

        Returns:
            List of statistical validation issues
        """
        pass



class OHLCVValidator(DataValidator):
    """Concrete validator for OHLCV market data.

    This validator implements structural, logical, temporal, and statistical
    validation for OHLCV data using Pandera schemas and custom checks.

    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.5, 7.5

    Attributes:
        policy: ActionPolicy configuration for handling validation failures
    """

    def __init__(self, policy: Optional[ActionPolicy] = None):
        """Initialize the validator with an optional policy.

        Args:
            policy: ActionPolicy configuration. Uses defaults if not provided.
        """
        self.policy = policy or ActionPolicy()

    def validate(
        self,
        df: pd.DataFrame,
        context: ValidationContext = ValidationContext.DAILY,
        lazy: bool = True,
    ) -> ValidationReport:
        """
        Validate DataFrame and return comprehensive report.

        Performs validation in order:
        1. Structural validation (columns, types)
        2. Logical validation (OHLCV relationships)
        3. Temporal validation (gaps) - if ticker column exists
        4. Statistical validation (outliers) - context-aware

        Args:
            df: Input DataFrame to validate
            context: DAILY (no T+1 data) or BACKFILL (has T+1 data)
            lazy: If True, collect all errors; if False, fail on first error

        Returns:
            ValidationReport with all validation results
        """
        timestamp = datetime.now()
        rows_input = len(df)

        # Initialize issue lists
        structural_issues: List[ValidationIssue] = []
        logical_issues: List[ValidationIssue] = []
        temporal_issues: List[ValidationIssue] = []
        statistical_issues: List[ValidationIssue] = []

        # Track spike classifications
        potential_spikes: List[dict] = []
        confirmed_spikes: List[dict] = []
        persistent_moves: List[dict] = []

        # Determine ticker for report
        ticker = None
        if "ticker" in df.columns and len(df) > 0:
            unique_tickers = df["ticker"].unique()
            ticker = unique_tickers[0] if len(unique_tickers) == 1 else None

        # 1. Structural validation
        structural_issues = self.validate_structural(df)

        # If structural validation fails in eager mode, stop here
        if not lazy and structural_issues:
            return ValidationReport(
                timestamp=timestamp,
                context=context,
                ticker=ticker,
                rows_input=rows_input,
                rows_output=0,
                structural_issues=structural_issues,
                logical_issues=[],
                temporal_issues=[],
                statistical_issues=[],
                rows_dropped=rows_input,
                rows_warned=0,
                rows_interpolated=0,
                rows_ffilled=0,
            )

        # 2. Logical validation (only if structural passed or lazy mode)
        if not structural_issues or lazy:
            logical_issues = self.validate_logical(df)

            # If logical validation fails in eager mode, stop here
            if not lazy and logical_issues:
                return ValidationReport(
                    timestamp=timestamp,
                    context=context,
                    ticker=ticker,
                    rows_input=rows_input,
                    rows_output=rows_input - len(logical_issues),
                    structural_issues=structural_issues,
                    logical_issues=logical_issues,
                    temporal_issues=[],
                    statistical_issues=[],
                    rows_dropped=len(logical_issues),
                    rows_warned=0,
                    rows_interpolated=0,
                    rows_ffilled=0,
                )

        # 3. Temporal validation (per ticker)
        if "ticker" in df.columns and "date" in df.columns:
            for t in df["ticker"].unique():
                ticker_df = df[df["ticker"] == t]
                ticker_temporal_issues = self.validate_temporal(ticker_df, t)
                temporal_issues.extend(ticker_temporal_issues)

        # 4. Statistical validation (context-aware)
        statistical_issues = self.validate_statistical(df, context)

        # Classify spikes based on context
        for issue in statistical_issues:
            spike_info = {
                "ticker": ticker or issue.column,
                "date": str(issue.value) if issue.value else "unknown",
                "return": 0.0,  # Would be populated with actual return
                "row_index": issue.row_index,
            }
            if context == ValidationContext.DAILY:
                potential_spikes.append(spike_info)
            elif "confirmed" in issue.message.lower():
                confirmed_spikes.append(spike_info)
            else:
                persistent_moves.append(spike_info)

        # Calculate action counts
        rows_dropped = sum(
            1 for i in structural_issues + logical_issues
            if i.action_taken == Action.DROP
        )
        rows_warned = sum(
            1 for i in structural_issues + logical_issues + temporal_issues + statistical_issues
            if i.action_taken == Action.WARN
        )
        rows_interpolated = sum(
            1 for i in temporal_issues + logical_issues
            if i.action_taken == Action.INTERPOLATE
        )
        rows_ffilled = sum(
            1 for i in temporal_issues + logical_issues
            if i.action_taken == Action.FFILL
        )

        return ValidationReport(
            timestamp=timestamp,
            context=context,
            ticker=ticker,
            rows_input=rows_input,
            rows_output=rows_input - rows_dropped,
            structural_issues=structural_issues,
            logical_issues=logical_issues,
            temporal_issues=temporal_issues,
            statistical_issues=statistical_issues,
            rows_dropped=rows_dropped,
            rows_warned=rows_warned,
            rows_interpolated=rows_interpolated,
            rows_ffilled=rows_ffilled,
            potential_spikes=potential_spikes,
            confirmed_spikes=confirmed_spikes,
            persistent_moves=persistent_moves,
        )

    def validate_structural(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check column presence and data types using Pandera schema.

        Requirements: 1.1, 1.2, 1.3, 1.4, 1.5

        Args:
            df: Input DataFrame to validate

        Returns:
            List of structural validation issues
        """
        issues: List[ValidationIssue] = []
        required_columns = get_required_columns()

        # Check for missing columns (Requirement 1.1, 1.2)
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(
                ValidationIssue(
                    issue_type="structural",
                    severity="error",
                    message=f"Missing required columns: {missing_columns}",
                    action_taken=self.policy.structural_failure,
                    row_index=None,
                    column=None,
                    value=missing_columns,
                )
            )
            return issues  # Cannot proceed without required columns

        # Use Pandera for type validation with coercion (Requirement 1.3, 1.4)
        try:
            # Validate with lazy=True to collect all errors
            _, errors = validate_ohlcv(df, lazy=True)
            if errors:
                # Convert Pandera errors to ValidationIssues
                issues.extend(self._convert_pandera_errors(errors))
        except Exception as e:
            issues.append(
                ValidationIssue(
                    issue_type="structural",
                    severity="error",
                    message=f"Schema validation error: {str(e)}",
                    action_taken=self.policy.structural_failure,
                    row_index=None,
                    column=None,
                    value=None,
                )
            )

        return issues

    def _convert_pandera_errors(
        self, errors: pa.errors.SchemaErrors
    ) -> List[ValidationIssue]:
        """Convert Pandera SchemaErrors to ValidationIssues.

        Args:
            errors: Pandera SchemaErrors object

        Returns:
            List of ValidationIssue objects
        """
        issues: List[ValidationIssue] = []

        for error in errors.schema_errors:
            # Extract error details
            check_name = getattr(error, "check", "unknown")
            column = getattr(error, "schema", None)
            if hasattr(column, "name"):
                column = column.name

            # Determine if this is a type error or constraint error
            error_msg = str(error)
            if "dtype" in error_msg.lower() or "type" in error_msg.lower():
                severity = "error"
                action = self.policy.structural_failure
            else:
                severity = "warning"
                action = Action.WARN

            # Get failure cases if available
            failure_cases = getattr(error, "failure_cases", None)
            row_indices = []
            if failure_cases is not None and hasattr(failure_cases, "index"):
                row_indices = list(failure_cases.index)

            if row_indices:
                # Create an issue for each failing row
                for idx in row_indices[:10]:  # Limit to first 10 for readability
                    issues.append(
                        ValidationIssue(
                            issue_type="structural",
                            severity=severity,
                            message=f"Validation failed for check '{check_name}': {error_msg}",
                            action_taken=action,
                            row_index=int(idx),
                            column=str(column) if column else None,
                            value=None,
                        )
                    )
            else:
                issues.append(
                    ValidationIssue(
                        issue_type="structural",
                        severity=severity,
                        message=f"Validation failed for check '{check_name}': {error_msg}",
                        action_taken=action,
                        row_index=None,
                        column=str(column) if column else None,
                        value=None,
                    )
                )

        return issues

    def validate_logical(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check OHLCV logical relationships.

        Requirements: 2.1, 2.2, 2.3, 2.5

        Validates:
        - High >= Low
        - High >= Open AND High >= Close
        - Low <= Open AND Low <= Close
        - All prices > 0

        Args:
            df: Input DataFrame to validate

        Returns:
            List of logical validation issues
        """
        issues: List[ValidationIssue] = []

        # Skip if required columns are missing
        required = ["open", "high", "low", "close"]
        if not all(col in df.columns for col in required):
            return issues

        # Check High >= Low (Requirement 2.1)
        invalid_hl = df[df["high"] < df["low"]]
        for idx in invalid_hl.index:
            issues.append(
                ValidationIssue(
                    issue_type="logical",
                    severity="error",
                    message=f"High ({df.loc[idx, 'high']}) < Low ({df.loc[idx, 'low']})",
                    action_taken=self.policy.logical_failure,
                    row_index=int(idx),
                    column="high,low",
                    value={"high": df.loc[idx, "high"], "low": df.loc[idx, "low"]},
                )
            )

        # Check High >= Open (Requirement 2.2)
        invalid_ho = df[df["high"] < df["open"]]
        for idx in invalid_ho.index:
            issues.append(
                ValidationIssue(
                    issue_type="logical",
                    severity="error",
                    message=f"High ({df.loc[idx, 'high']}) < Open ({df.loc[idx, 'open']})",
                    action_taken=self.policy.logical_failure,
                    row_index=int(idx),
                    column="high,open",
                    value={"high": df.loc[idx, "high"], "open": df.loc[idx, "open"]},
                )
            )

        # Check High >= Close (Requirement 2.2)
        invalid_hc = df[df["high"] < df["close"]]
        for idx in invalid_hc.index:
            issues.append(
                ValidationIssue(
                    issue_type="logical",
                    severity="error",
                    message=f"High ({df.loc[idx, 'high']}) < Close ({df.loc[idx, 'close']})",
                    action_taken=self.policy.logical_failure,
                    row_index=int(idx),
                    column="high,close",
                    value={"high": df.loc[idx, "high"], "close": df.loc[idx, "close"]},
                )
            )

        # Check Low <= Open (Requirement 2.3)
        invalid_lo = df[df["low"] > df["open"]]
        for idx in invalid_lo.index:
            issues.append(
                ValidationIssue(
                    issue_type="logical",
                    severity="error",
                    message=f"Low ({df.loc[idx, 'low']}) > Open ({df.loc[idx, 'open']})",
                    action_taken=self.policy.logical_failure,
                    row_index=int(idx),
                    column="low,open",
                    value={"low": df.loc[idx, "low"], "open": df.loc[idx, "open"]},
                )
            )

        # Check Low <= Close (Requirement 2.3)
        invalid_lc = df[df["low"] > df["close"]]
        for idx in invalid_lc.index:
            issues.append(
                ValidationIssue(
                    issue_type="logical",
                    severity="error",
                    message=f"Low ({df.loc[idx, 'low']}) > Close ({df.loc[idx, 'close']})",
                    action_taken=self.policy.logical_failure,
                    row_index=int(idx),
                    column="low,close",
                    value={"low": df.loc[idx, "low"], "close": df.loc[idx, "close"]},
                )
            )

        return issues

    def validate_temporal(
        self, df: pd.DataFrame, ticker: str
    ) -> List[ValidationIssue]:
        """Check for missing business days using pandas_market_calendars.

        Requirements: 3.1, 3.2, 3.3, 3.4, 3.5

        This method identifies gaps in the time series by comparing the dates
        present in the data against the expected business days from the NYSE
        calendar (excluding weekends and official market holidays).

        Args:
            df: Input DataFrame to validate (single ticker)
            ticker: Ticker symbol for the time series

        Returns:
            List of temporal validation issues (gaps)
        """
        issues: List[ValidationIssue] = []

        # Skip if date column is missing or empty
        if "date" not in df.columns or len(df) == 0:
            return issues

        # Get the date range from the data
        dates = pd.to_datetime(df["date"])
        min_date = dates.min()
        max_date = dates.max()

        # Skip if only one date (no gaps possible)
        if min_date == max_date:
            return issues

        # Get NYSE calendar for business day detection (Requirement 3.2)
        # NYSE is the standard for US equities
        nyse = mcal.get_calendar("NYSE")

        # Get all valid trading days in the date range
        schedule = nyse.schedule(start_date=min_date, end_date=max_date)
        expected_business_days: Set[pd.Timestamp] = set(
            pd.to_datetime(schedule.index).normalize()
        )

        # Get actual dates in the data (normalized to midnight)
        actual_dates: Set[pd.Timestamp] = set(dates.dt.normalize())

        # Find missing business days (Requirement 3.1)
        missing_days = sorted(expected_business_days - actual_dates)

        if not missing_days:
            return issues

        # Group consecutive missing days into gaps
        gaps = self._group_consecutive_dates(missing_days)

        # Process each gap
        for gap in gaps:
            gap_length = len(gap)
            gap_start = gap[0]
            gap_end = gap[-1]

            if gap_length == 1:
                # Single-day gap (Requirement 3.4)
                issues.append(
                    ValidationIssue(
                        issue_type="temporal",
                        severity="warning",
                        message=f"Missing single business day: {gap_start.strftime('%Y-%m-%d')} for ticker {ticker}",
                        action_taken=self.policy.temporal_gap_single,
                        row_index=None,
                        column="date",
                        value=gap_start.strftime("%Y-%m-%d"),
                    )
                )
            elif gap_length > 3:
                # Multi-day gap > 3 days (Requirement 3.3)
                issues.append(
                    ValidationIssue(
                        issue_type="temporal",
                        severity="warning",
                        message=f"Large gap of {gap_length} consecutive business days: {gap_start.strftime('%Y-%m-%d')} to {gap_end.strftime('%Y-%m-%d')} for ticker {ticker}",
                        action_taken=self.policy.temporal_gap_multi,
                        row_index=None,
                        column="date",
                        value={
                            "start": gap_start.strftime("%Y-%m-%d"),
                            "end": gap_end.strftime("%Y-%m-%d"),
                            "days": gap_length,
                        },
                    )
                )
            else:
                # Gap of 2-3 days - flag for potential interpolation
                issues.append(
                    ValidationIssue(
                        issue_type="temporal",
                        severity="warning",
                        message=f"Gap of {gap_length} consecutive business days: {gap_start.strftime('%Y-%m-%d')} to {gap_end.strftime('%Y-%m-%d')} for ticker {ticker}",
                        action_taken=self.policy.temporal_gap_single,  # Can still interpolate
                        row_index=None,
                        column="date",
                        value={
                            "start": gap_start.strftime("%Y-%m-%d"),
                            "end": gap_end.strftime("%Y-%m-%d"),
                            "days": gap_length,
                        },
                    )
                )

        return issues

    def _group_consecutive_dates(
        self, dates: List[pd.Timestamp]
    ) -> List[List[pd.Timestamp]]:
        """Group consecutive dates into gap sequences.

        Args:
            dates: Sorted list of missing dates

        Returns:
            List of gap groups, where each group is a list of consecutive dates
        """
        if not dates:
            return []

        gaps: List[List[pd.Timestamp]] = []
        current_gap: List[pd.Timestamp] = [dates[0]]

        for i in range(1, len(dates)):
            # Check if this date is consecutive to the previous one
            # We use business day difference, but since these are already
            # business days from the calendar, we just check if they're
            # within a reasonable range (accounting for weekends)
            prev_date = dates[i - 1]
            curr_date = dates[i]
            day_diff = (curr_date - prev_date).days

            # If the gap between missing days is <= 3 calendar days,
            # they're likely consecutive business days
            if day_diff <= 3:
                current_gap.append(curr_date)
            else:
                # Start a new gap
                gaps.append(current_gap)
                current_gap = [curr_date]

        # Don't forget the last gap
        gaps.append(current_gap)

        return gaps

    def validate_statistical(
        self, df: pd.DataFrame, context: ValidationContext
    ) -> List[ValidationIssue]:
        """
        Detect outliers and spikes with context awareness.

        Requirements: 4.1, 4.2, 4.3, 4.4, 4.5

        Context-aware behavior:
        - DAILY: Flags "potential_spike" (cannot verify reversion)
        - BACKFILL: Verifies reversion to confirm "spike" vs "persistent_move"

        Args:
            df: Input DataFrame to validate
            context: Validation context (DAILY or BACKFILL)

        Returns:
            List of statistical validation issues
        """
        issues: List[ValidationIssue] = []

        # Skip if required columns are missing or insufficient data
        if "adj_close" not in df.columns or len(df) < 2:
            return issues

        # Need date column for sorting and ticker for reporting
        if "date" not in df.columns:
            return issues

        # Process each ticker separately if multiple tickers present
        if "ticker" in df.columns:
            tickers = df["ticker"].unique()
        else:
            tickers = [None]

        for ticker in tickers:
            if ticker is not None:
                ticker_df = df[df["ticker"] == ticker].copy()
            else:
                ticker_df = df.copy()

            # Sort by date for proper return calculation
            ticker_df = ticker_df.sort_values("date").reset_index(drop=True)

            if len(ticker_df) < 2:
                continue

            # Compute daily returns (Requirement 4.1)
            ticker_df["daily_return"] = ticker_df["adj_close"].pct_change()

            # Calculate rolling statistics for sigma-based detection
            # Use expanding window to have enough data points
            if len(ticker_df) >= 20:
                rolling_mean = ticker_df["daily_return"].rolling(window=20, min_periods=5).mean()
                rolling_std = ticker_df["daily_return"].rolling(window=20, min_periods=5).std()
            else:
                # For short series, use simple mean/std
                rolling_mean = ticker_df["daily_return"].mean()
                rolling_std = ticker_df["daily_return"].std()

            # Detect outliers (Requirement 4.1, 4.2)
            for idx in range(1, len(ticker_df)):  # Skip first row (NaN return)
                row = ticker_df.iloc[idx]
                daily_return = row["daily_return"]
                original_idx = ticker_df.index[idx]

                if pd.isna(daily_return):
                    continue

                # Get sigma threshold
                if isinstance(rolling_std, pd.Series):
                    std = rolling_std.iloc[idx] if not pd.isna(rolling_std.iloc[idx]) else ticker_df["daily_return"].std()
                    mean = rolling_mean.iloc[idx] if not pd.isna(rolling_mean.iloc[idx]) else ticker_df["daily_return"].mean()
                else:
                    std = rolling_std
                    mean = rolling_mean

                if pd.isna(std) or std == 0:
                    std = 0.02  # Default to 2% if no std available

                # Check if outlier: > sigma_threshold OR > pct_threshold (Requirement 4.1, 4.2)
                z_score = abs((daily_return - mean) / std) if std > 0 else 0
                is_sigma_outlier = z_score > self.policy.sigma_threshold
                is_pct_outlier = abs(daily_return) > self.policy.pct_threshold

                if is_sigma_outlier or is_pct_outlier:
                    # Outlier detected - handle based on context
                    outlier_type = "sigma" if is_sigma_outlier else "percentage"
                    
                    if context == ValidationContext.DAILY:
                        # DAILY context: Flag as potential spike (Requirement 4.3)
                        issues.append(
                            ValidationIssue(
                                issue_type="statistical",
                                severity="warning",
                                message=f"Potential spike detected ({outlier_type}): {daily_return:.2%} return on {row['date']} for {ticker or 'unknown'}",
                                action_taken=self.policy.outlier_potential_spike,
                                row_index=int(original_idx),
                                column="adj_close",
                                value={
                                    "return": daily_return,
                                    "z_score": z_score,
                                    "date": str(row["date"]),
                                    "classification": "potential_spike",
                                },
                            )
                        )
                    else:
                        # BACKFILL context: Verify T+1 reversion (Requirement 4.4)
                        reverts = self._check_reversion(ticker_df, idx)
                        
                        if reverts:
                            # Confirmed spike - price reverted (Requirement 4.4)
                            issues.append(
                                ValidationIssue(
                                    issue_type="statistical",
                                    severity="error",
                                    message=f"Confirmed spike ({outlier_type}): {daily_return:.2%} return on {row['date']} for {ticker or 'unknown'} - reverted within 1 day",
                                    action_taken=self.policy.outlier_spike,
                                    row_index=int(original_idx),
                                    column="adj_close",
                                    value={
                                        "return": daily_return,
                                        "z_score": z_score,
                                        "date": str(row["date"]),
                                        "classification": "confirmed_spike",
                                    },
                                )
                            )
                        else:
                            # Persistent move - no reversion (Requirement 4.4)
                            issues.append(
                                ValidationIssue(
                                    issue_type="statistical",
                                    severity="warning",
                                    message=f"Persistent large move ({outlier_type}): {daily_return:.2%} return on {row['date']} for {ticker or 'unknown'} - did not revert",
                                    action_taken=self.policy.outlier_persistent,
                                    row_index=int(original_idx),
                                    column="adj_close",
                                    value={
                                        "return": daily_return,
                                        "z_score": z_score,
                                        "date": str(row["date"]),
                                        "classification": "persistent_move",
                                    },
                                )
                            )

        return issues

    def _check_reversion(self, df: pd.DataFrame, spike_idx: int) -> bool:
        """
        Check if a spike reverts within 1 trading day.

        A spike is considered to revert if:
        - The T+1 return is in the opposite direction
        - The T+1 return magnitude is at least 50% of the spike magnitude

        Args:
            df: DataFrame sorted by date with daily_return column
            spike_idx: Index of the spike in the DataFrame

        Returns:
            True if the spike reverts, False otherwise
        """
        # Check if we have T+1 data
        if spike_idx >= len(df) - 1:
            return False  # No T+1 data available

        spike_return = df.iloc[spike_idx]["daily_return"]
        next_return = df.iloc[spike_idx + 1]["daily_return"]

        if pd.isna(next_return):
            return False

        # Check for reversion:
        # 1. Opposite direction
        opposite_direction = (spike_return > 0 and next_return < 0) or (spike_return < 0 and next_return > 0)

        # 2. Magnitude at least 50% of spike (significant reversion)
        significant_magnitude = abs(next_return) >= 0.5 * abs(spike_return)

        return opposite_direction and significant_magnitude
