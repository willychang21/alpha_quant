"""Monitoring Service with Metrics Persistence.

Provides job status tracking with JSONL persistence via MetricsCollector.
"""

import logging
import time
from datetime import datetime
from contextlib import contextmanager
from typing import Optional

from sqlalchemy.orm import Session

from core.metrics import get_metrics_collector, MetricsCollector

logger = logging.getLogger(__name__)


class MonitoringService:
    """Job monitoring with metrics persistence.
    
    Records job success/failure with duration tracking
    and persists to JSONL via MetricsCollector.
    """
    
    def __init__(
        self, 
        db: Session = None,
        metrics_collector: MetricsCollector = None
    ):
        self.db = db
        self._metrics = metrics_collector or get_metrics_collector()
    
    def record_success(self, job_name: str, duration_seconds: float = 0.0):
        """Record a successful job run.
        
        Args:
            job_name: Name/type of the job
            duration_seconds: How long the job took
        """
        logger.info(f"MONITORING: Job '{job_name}' SUCCEEDED at {datetime.now()}")
        
        # Persist to JSONL
        self._metrics.record_job(
            job_name=job_name,
            status="success",
            duration_seconds=duration_seconds
        )
        
    def record_failure(self, job_name: str, error: str, duration_seconds: float = 0.0):
        """Record a failed job run.
        
        Args:
            job_name: Name/type of the job
            error: Error message
            duration_seconds: How long before failure
        """
        logger.error(f"MONITORING: Job '{job_name}' FAILED at {datetime.now()}. Error: {error}")
        
        # Persist to JSONL
        self._metrics.record_job(
            job_name=job_name,
            status="failure",
            duration_seconds=duration_seconds
        )
    
    @contextmanager
    def track_job(self, job_name: str):
        """Context manager for tracking job execution.
        
        Usage:
            with monitoring.track_job("my_job") as tracker:
                # do work
                pass  # success is recorded automatically
        
        Args:
            job_name: Name of the job to track
            
        Yields:
            None
        """
        start_time = time.time()
        try:
            yield
            duration = time.time() - start_time
            self.record_success(job_name, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.record_failure(job_name, str(e), duration)
            raise
    
    def check_data_freshness(self, ticker: str = None) -> bool:
        """Check if data is fresh (within threshold).
        
        Args:
            ticker: Stock ticker (optional, for future per-ticker checks)
            
        Returns:
            True if data is fresh
        """
        from core.freshness import get_data_freshness_service
        
        freshness_service = get_data_freshness_service()
        is_fresh, _, _ = freshness_service.get_freshness_status()
        return is_fresh


# Singleton instance
_monitoring_service: Optional[MonitoringService] = None


def get_monitoring_service(db: Session = None) -> MonitoringService:
    """Get or create MonitoringService singleton."""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService(db=db)
    return _monitoring_service
