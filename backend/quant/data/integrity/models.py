"""Data models for the Data Integrity & Validation Framework.

This module defines the core data structures for validation results,
including ValidationIssue and ValidationReport dataclasses with
serialization support.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from quant.data.integrity.enums import Action, ValidationContext


@dataclass
class ValidationIssue:
    """Single validation issue.

    Attributes:
        issue_type: Category - structural, logical, temporal, statistical
        severity: error or warning
        row_index: Affected row index (None for batch-level issues)
        column: Affected column name
        message: Human-readable description
        value: The problematic value
        action_taken: Action applied (WARN, DROP, INTERPOLATE, FFILL)
    """

    issue_type: str
    severity: str
    message: str
    action_taken: Action
    row_index: Optional[int] = None
    column: Optional[str] = None
    value: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "issue_type": self.issue_type,
            "severity": self.severity,
            "row_index": self.row_index,
            "column": self.column,
            "message": self.message,
            "value": self.value,
            "action_taken": self.action_taken.value,
        }


    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationIssue":
        """Create from dictionary."""
        return cls(
            issue_type=data["issue_type"],
            severity=data["severity"],
            row_index=data.get("row_index"),
            column=data.get("column"),
            message=data["message"],
            value=data.get("value"),
            action_taken=Action(data["action_taken"]),
        )


@dataclass
class ValidationReport:
    """Structured validation results.

    Attributes:
        timestamp: When validation ran
        context: DAILY or BACKFILL
        ticker: Ticker symbol (None for multi-ticker batches)
        rows_input: Input row count
        rows_output: Output row count after actions
        structural_issues: Column/type issues
        logical_issues: OHLCV rule violations
        temporal_issues: Missing day gaps
        statistical_issues: Outliers and spikes
        rows_dropped: Rows removed
        rows_interpolated: Rows with interpolated price values
        rows_ffilled: Rows with forward-filled volume
        rows_warned: Rows with warnings only
        potential_spikes: DAILY - Unverified spike candidates
        confirmed_spikes: BACKFILL - Verified spikes (reverted)
        persistent_moves: BACKFILL - Large moves that didn't revert
    """

    timestamp: datetime
    context: ValidationContext
    rows_input: int
    rows_output: int
    ticker: Optional[str] = None
    structural_issues: List[ValidationIssue] = field(default_factory=list)
    logical_issues: List[ValidationIssue] = field(default_factory=list)
    temporal_issues: List[ValidationIssue] = field(default_factory=list)
    statistical_issues: List[ValidationIssue] = field(default_factory=list)
    rows_dropped: int = 0
    rows_interpolated: int = 0
    rows_ffilled: int = 0
    rows_warned: int = 0
    potential_spikes: List[Dict[str, Any]] = field(default_factory=list)
    confirmed_spikes: List[Dict[str, Any]] = field(default_factory=list)
    persistent_moves: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Check if structural validation passed."""
        return len(self.structural_issues) == 0

    @property
    def drop_rate(self) -> float:
        """Calculate the drop rate as a fraction."""
        return self.rows_dropped / self.rows_input if self.rows_input > 0 else 0.0

    @property
    def all_issues(self) -> List[ValidationIssue]:
        """Get all issues across all categories."""
        return (
            self.structural_issues
            + self.logical_issues
            + self.temporal_issues
            + self.statistical_issues
        )


    def to_json(self) -> str:
        """Serialize to JSON string."""
        data = {
            "timestamp": self.timestamp.isoformat(),
            "context": self.context.value,
            "ticker": self.ticker,
            "rows_input": self.rows_input,
            "rows_output": self.rows_output,
            "structural_issues": [i.to_dict() for i in self.structural_issues],
            "logical_issues": [i.to_dict() for i in self.logical_issues],
            "temporal_issues": [i.to_dict() for i in self.temporal_issues],
            "statistical_issues": [i.to_dict() for i in self.statistical_issues],
            "rows_dropped": self.rows_dropped,
            "rows_interpolated": self.rows_interpolated,
            "rows_ffilled": self.rows_ffilled,
            "rows_warned": self.rows_warned,
            "potential_spikes": self.potential_spikes,
            "confirmed_spikes": self.confirmed_spikes,
            "persistent_moves": self.persistent_moves,
        }
        return json.dumps(data, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "ValidationReport":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            context=ValidationContext(data["context"]),
            ticker=data.get("ticker"),
            rows_input=data["rows_input"],
            rows_output=data["rows_output"],
            structural_issues=[
                ValidationIssue.from_dict(i) for i in data.get("structural_issues", [])
            ],
            logical_issues=[
                ValidationIssue.from_dict(i) for i in data.get("logical_issues", [])
            ],
            temporal_issues=[
                ValidationIssue.from_dict(i) for i in data.get("temporal_issues", [])
            ],
            statistical_issues=[
                ValidationIssue.from_dict(i) for i in data.get("statistical_issues", [])
            ],
            rows_dropped=data.get("rows_dropped", 0),
            rows_interpolated=data.get("rows_interpolated", 0),
            rows_ffilled=data.get("rows_ffilled", 0),
            rows_warned=data.get("rows_warned", 0),
            potential_spikes=data.get("potential_spikes", []),
            confirmed_spikes=data.get("confirmed_spikes", []),
            persistent_moves=data.get("persistent_moves", []),
        )


    def to_text(self) -> str:
        """Human-readable format for display."""
        lines = [
            "=" * 60,
            f"VALIDATION REPORT - {self.context.value.upper()} MODE",
            "=" * 60,
            f"Timestamp: {self.timestamp}",
            f"Ticker: {self.ticker or 'Multiple'}",
            "",
            "SUMMARY",
            "-" * 40,
            f"Rows Input:        {self.rows_input}",
            f"Rows Output:       {self.rows_output}",
            f"Rows Dropped:      {self.rows_dropped} ({self.drop_rate:.1%})",
            f"Rows Interpolated: {self.rows_interpolated}",
            f"Rows FFilled:      {self.rows_ffilled}",
            f"Rows Warned:       {self.rows_warned}",
            "",
        ]

        # Add issue sections
        if self.structural_issues:
            lines.extend(
                [
                    "STRUCTURAL ISSUES",
                    "-" * 40,
                ]
            )
            for issue in self.structural_issues:
                lines.append(f"  [{issue.severity.upper()}] {issue.message}")

        if self.logical_issues:
            lines.extend(
                [
                    "",
                    "LOGICAL ISSUES",
                    "-" * 40,
                ]
            )
            for issue in self.logical_issues:
                col_info = f" ({issue.column})" if issue.column else ""
                row_info = f" @ row {issue.row_index}" if issue.row_index is not None else ""
                lines.append(f"  [{issue.severity.upper()}]{row_info}{col_info}: {issue.message}")

        if self.temporal_issues:
            lines.extend(
                [
                    "",
                    "TEMPORAL ISSUES (Gaps)",
                    "-" * 40,
                ]
            )
            for issue in self.temporal_issues:
                lines.append(f"  [{issue.severity.upper()}] {issue.message}")

        if self.statistical_issues:
            lines.extend(
                [
                    "",
                    "STATISTICAL ISSUES (Outliers)",
                    "-" * 40,
                ]
            )
            for issue in self.statistical_issues:
                lines.append(f"  [{issue.severity.upper()}] {issue.message}")

        # Context-specific spike information
        if self.context == ValidationContext.DAILY and self.potential_spikes:
            lines.extend(
                [
                    "",
                    "POTENTIAL SPIKES (Unverified - DAILY mode)",
                    "-" * 40,
                ]
            )
            for spike in self.potential_spikes:
                ret = spike.get("return", 0)
                lines.append(f"  {spike.get('ticker', 'N/A')} @ {spike.get('date', 'N/A')}: {ret:.1%}")

        if self.context == ValidationContext.BACKFILL:
            if self.confirmed_spikes:
                lines.extend(
                    [
                        "",
                        "CONFIRMED SPIKES (Reverted)",
                        "-" * 40,
                    ]
                )
                for spike in self.confirmed_spikes:
                    ret = spike.get("return", 0)
                    lines.append(f"  {spike.get('ticker', 'N/A')} @ {spike.get('date', 'N/A')}: {ret:.1%}")

            if self.persistent_moves:
                lines.extend(
                    [
                        "",
                        "PERSISTENT MOVES (Did not revert)",
                        "-" * 40,
                    ]
                )
                for move in self.persistent_moves:
                    ret = move.get("return", 0)
                    lines.append(f"  {move.get('ticker', 'N/A')} @ {move.get('date', 'N/A')}: {ret:.1%}")

        lines.append("=" * 60)
        return "\n".join(lines)
