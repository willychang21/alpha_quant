"""Centralized Error Handler with Categorization and Retry Logic.

Provides granular error handling with:
- Error categorization (transient, data quality, permanent)
- Retry decorator with exponential backoff for transient errors
- Graceful handling of data quality errors
"""

import logging
import time
from enum import Enum
from typing import TypeVar, Callable, Optional, Any, Tuple, Type
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorCategory(Enum):
    """Categories of errors for handling decisions.
    
    TRANSIENT: Temporary failures that may succeed on retry (network, timeout)
    DATA_QUALITY: Data issues that should skip the record (missing/malformed data)
    PERMANENT: Unrecoverable errors that should fail the operation
    """
    TRANSIENT = "transient"
    DATA_QUALITY = "data_quality"
    PERMANENT = "permanent"


class ErrorHandler:
    """Centralized error handling with categorization.
    
    Provides consistent error handling across the codebase:
    - Categorizes exceptions into transient, data quality, or permanent
    - Offers retry decorator for transient errors
    - Logs errors appropriately based on category
    """
    
    # Transient errors that should be retried
    TRANSIENT_ERRORS: Tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
    )
    
    # Data quality errors that should skip with warning
    DATA_QUALITY_ERRORS: Tuple[Type[Exception], ...] = (
        ValueError,
        KeyError,
        TypeError,
    )
    
    @classmethod
    def categorize(cls, error: Exception) -> ErrorCategory:
        """Categorize an exception for appropriate handling.
        
        Args:
            error: The exception to categorize
            
        Returns:
            ErrorCategory indicating how to handle the error
        """
        if isinstance(error, cls.TRANSIENT_ERRORS):
            return ErrorCategory.TRANSIENT
        elif isinstance(error, cls.DATA_QUALITY_ERRORS):
            return ErrorCategory.DATA_QUALITY
        return ErrorCategory.PERMANENT
    
    @classmethod
    def with_retry(
        cls,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        on_data_error: Optional[T] = None,
        component: str = "ErrorHandler",
    ) -> Callable:
        """Decorator for functions that should retry on transient errors.
        
        Implements exponential backoff: delay = base_delay * (2 ^ attempt)
        
        Args:
            max_retries: Maximum number of retry attempts for transient errors
            base_delay: Initial delay in seconds between retries
            max_delay: Maximum delay between retries (capped)
            on_data_error: Default value to return on data quality errors
            component: Component name for logging prefix
            
        Returns:
            Decorated function with retry behavior
            
        Example:
            @ErrorHandler.with_retry(max_retries=3, on_data_error=None)
            def fetch_ticker_data(ticker: str) -> dict:
                return yf.Ticker(ticker).info
        """
        def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Optional[T]:
                last_error: Optional[Exception] = None
                
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    
                    except Exception as e:
                        last_error = e
                        category = cls.categorize(e)
                        
                        if category == ErrorCategory.DATA_QUALITY:
                            # Data quality errors: log warning and return default
                            logger.warning(
                                f"[{component}] Data quality error in {func.__name__}: "
                                f"{type(e).__name__}: {e}"
                            )
                            return on_data_error
                        
                        elif category == ErrorCategory.TRANSIENT:
                            # Transient errors: retry with backoff
                            if attempt < max_retries:
                                delay = min(base_delay * (2 ** attempt), max_delay)
                                logger.warning(
                                    f"[{component}] Transient error in {func.__name__} "
                                    f"(attempt {attempt + 1}/{max_retries + 1}): "
                                    f"{type(e).__name__}: {e}. "
                                    f"Retrying in {delay:.1f}s..."
                                )
                                time.sleep(delay)
                            else:
                                logger.error(
                                    f"[{component}] Transient error in {func.__name__} "
                                    f"after {max_retries + 1} attempts: "
                                    f"{type(e).__name__}: {e}",
                                    exc_info=True
                                )
                                raise
                        
                        else:
                            # Permanent errors: log and re-raise immediately
                            logger.error(
                                f"[{component}] Permanent error in {func.__name__}: "
                                f"{type(e).__name__}: {e}",
                                exc_info=True
                            )
                            raise
                
                # Should not reach here, but just in case
                if last_error:
                    raise last_error
                return on_data_error
            
            return wrapper
        return decorator
    
    @classmethod
    def handle_gracefully(
        cls,
        default: Optional[T] = None,
        component: str = "ErrorHandler",
    ) -> Callable:
        """Decorator for functions that should never raise but return default on error.
        
        Useful for non-critical operations where failure should not break the flow.
        
        Args:
            default: Default value to return on any error
            component: Component name for logging prefix
            
        Returns:
            Decorated function that returns default on any error
        """
        def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Optional[T]:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    category = cls.categorize(e)
                    if category == ErrorCategory.PERMANENT:
                        logger.error(
                            f"[{component}] Error in {func.__name__}: "
                            f"{type(e).__name__}: {e}",
                            exc_info=True
                        )
                    else:
                        logger.warning(
                            f"[{component}] Error in {func.__name__}: "
                            f"{type(e).__name__}: {e}"
                        )
                    return default
            return wrapper
        return decorator


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    on_data_error: Optional[Any] = None,
    component: str = "ErrorHandler",
) -> Callable:
    """Convenience function for ErrorHandler.with_retry decorator.
    
    See ErrorHandler.with_retry for full documentation.
    """
    return ErrorHandler.with_retry(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        on_data_error=on_data_error,
        component=component,
    )


def handle_gracefully(
    default: Optional[Any] = None,
    component: str = "ErrorHandler",
) -> Callable:
    """Convenience function for ErrorHandler.handle_gracefully decorator.
    
    See ErrorHandler.handle_gracefully for full documentation.
    """
    return ErrorHandler.handle_gracefully(
        default=default,
        component=component,
    )
