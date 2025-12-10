"""Exception hierarchy for the Data Integrity & Validation Framework.

This module defines custom exceptions for different validation failure types,
enabling granular error handling in the validation pipeline.

Requirements: 1.2, 1.4, 7.2
"""


class ValidationError(Exception):
    """Base validation exception.

    All validation-related exceptions inherit from this class,
    allowing for broad exception catching when needed.
    """

    def __init__(self, message: str, details: dict = None):
        """Initialize the exception.

        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class StructuralValidationError(ValidationError):
    """Missing columns or type errors.

    Raised when:
    - Required columns are missing from the DataFrame
    - Column data types cannot be coerced to expected types
    - Schema validation fails at the structural level

    Requirement: 1.2
    """

    pass


class LogicalValidationError(ValidationError):
    """OHLCV rule violations.

    Raised when:
    - High < Low
    - High < Open or High < Close
    - Low > Open or Low > Close
    - Prices are not strictly positive
    - Volume is zero or negative

    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
    """

    pass


class TemporalValidationError(ValidationError):
    """Gap detection errors.

    Raised when:
    - Missing business days are detected
    - Gap calculations fail
    - Market calendar operations fail

    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
    """

    pass


class StatisticalValidationError(ValidationError):
    """Outlier detection errors.

    Raised when:
    - Statistical calculations fail
    - Outlier threshold exceeded
    - Spike detection produces unexpected results

    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
    """

    pass


class CatchUpError(ValidationError):
    """Smart catch-up service errors.

    Raised when:
    - Data lake query fails
    - Gap calculation fails
    - Backfill operation fails
    - BACKFILL context validation fails

    Requirements: 6.1
    """

    pass


class ProcessingError(ValidationError):
    """Action processing errors.

    Raised when:
    - Interpolation fails
    - Forward fill fails
    - DROP action cannot be applied
    - Invalid action configuration

    Requirements: 5.2, 5.3, 5.4, 5.5
    """

    pass


class DropRateExceededError(ValidationError):
    """Drop rate threshold exceeded.

    Raised when the percentage of dropped rows exceeds
    the configured threshold (default: 10%).

    Requirement: 6.3
    """

    def __init__(self, message: str, drop_rate: float, threshold: float):
        """Initialize with drop rate details.

        Args:
            message: Human-readable error message
            drop_rate: Actual drop rate (0.0 to 1.0)
            threshold: Configured threshold (0.0 to 1.0)
        """
        super().__init__(
            message,
            details={"drop_rate": drop_rate, "threshold": threshold},
        )
        self.drop_rate = drop_rate
        self.threshold = threshold
