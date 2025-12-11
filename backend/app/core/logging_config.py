"""Structured JSON logging with correlation ID support.

This module provides:
- JSON-formatted log output for machine parsing
- Correlation ID propagation via ContextVar
- Request ID middleware integration
- Console and file handlers with rotation
"""

import json
import logging
import logging.handlers
import os
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Optional

# Context variable for correlation ID (request tracing)
request_id_var: ContextVar[str] = ContextVar('request_id', default='-')


def get_request_id() -> str:
    """Get current request ID from context."""
    return request_id_var.get()


def set_request_id(request_id: Optional[str] = None) -> str:
    """Set request ID in context. Generates UUID if not provided."""
    rid = request_id or str(uuid.uuid4())[:8]
    request_id_var.set(rid)
    return rid


class JSONFormatter(logging.Formatter):
    """JSON log formatter with structured output.
    
    Outputs logs as JSON with fields:
    - timestamp: ISO 8601 formatted timestamp
    - level: Log level (INFO, WARNING, ERROR, etc.)
    - logger: Logger name
    - message: Log message
    - request_id: Correlation ID for request tracing
    - exception: Formatted exception (if present)
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


def setup_logging(json_format: bool = False):
    """Configure centralized logging with optional JSON format.
    
    Args:
        json_format: If True, use JSON formatter. If False, use human-readable format.
    
    Features:
    - Rotating file handler (5MB, 3 backups)
    - Console output
    - JSON or human-readable format
    - Correlation ID support
    """
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
        'logs'
    )
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'app.log')
    
    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5*1024*1024, backupCount=3
    )
    
    # Create formatters
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    

    # Create Filter to inject request_id if missing
    class RequestIdFilter(logging.Filter):
        def filter(self, record):
            if not hasattr(record, 'request_id'):
                record.request_id = request_id_var.get()
            return True

    request_id_filter = RequestIdFilter()
    console_handler.addFilter(request_id_filter)
    file_handler.addFilter(request_id_filter)
    
    # Configure root logger
    logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])
    
    # Log startup
    logger = logging.getLogger(__name__)
    logger.info("Logging configured", extra={'extra_fields': {'log_file': log_file}})


# Convenience function for adding correlation ID to log context
class CorrelationAdapter(logging.LoggerAdapter):
    """Logger adapter that automatically includes request_id in logs."""
    
    def process(self, msg, kwargs):
        kwargs.setdefault('extra', {})
        kwargs['extra']['request_id'] = request_id_var.get()
        return msg, kwargs
