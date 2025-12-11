"""Job Store with SQLite persistence and retry support.

Provides:
- Job model for tracking background job state
- JobStore class for CRUD operations
- Exponential backoff retry logic
- State machine enforcement
"""

import enum
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import Column, String, DateTime, JSON, Integer, Enum, text
from sqlalchemy.orm import Session

from app.core.database import Base, SessionLocal
from config.settings import get_settings
import logging

logger = logging.getLogger(__name__)


class JobStatus(str, enum.Enum):
    """Job lifecycle states.
    
    Valid transitions:
    - pending -> running
    - running -> completed
    - running -> failed
    - running -> dead (if retry_count >= max_retries)
    - failed -> pending (if retry_count < max_retries)
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD = "dead"  # Permanently failed after max retries


class Job(Base):
    """SQLAlchemy model for background jobs.
    
    Tracks job state, parameters, results, and retry information.
    """
    __tablename__ = "jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    task_type = Column(String, nullable=False)
    status = Column(String, default=JobStatus.PENDING.value)
    params = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    error = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    retry_count = Column(Integer, default=0)


class JobStore:
    """Persistent job tracking with retry support.
    
    Provides CRUD operations for jobs with state machine enforcement
    and exponential backoff retry logic.
    """
    
    def __init__(self, session: Optional[Session] = None):
        self._session = session
        self._settings = get_settings()
    
    @property
    def session(self) -> Session:
        if self._session is None:
            self._session = SessionLocal()
        return self._session
    
    def create_job(
        self, 
        task_type: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> Job:
        """Create a new job with pending status.
        
        Args:
            task_type: Type of job (e.g., 'valuation', 'backtest')
            params: Job parameters
            
        Returns:
            Created Job instance
        """
        job = Job(
            id=str(uuid.uuid4()),
            task_type=task_type,
            status=JobStatus.PENDING.value,
            params=params,
            created_at=datetime.now(timezone.utc),
            retry_count=0
        )
        
        self.session.add(job)
        self.session.commit()
        self.session.refresh(job)
        
        return job
    
    def start_job(self, job_id: str) -> Optional[Job]:
        """Mark job as running.
        
        Args:
            job_id: Job ID
            
        Returns:
            Updated Job or None if not found
        """
        job = self.get_job(job_id)
        if job is None:
            return None
        
        if job.status != JobStatus.PENDING.value:
            raise ValueError(f"Cannot start job in {job.status} status")
        
        job.status = JobStatus.RUNNING.value
        job.started_at = datetime.now(timezone.utc)
        
        self.session.commit()
        self.session.refresh(job)
        
        return job
    
    def complete_job(
        self, 
        job_id: str, 
        result: Optional[Dict[str, Any]] = None
    ) -> Optional[Job]:
        """Mark job as completed.
        
        Args:
            job_id: Job ID
            result: Job result data
            
        Returns:
            Updated Job or None if not found
        """
        job = self.get_job(job_id)
        if job is None:
            return None
        
        if job.status != JobStatus.RUNNING.value:
            raise ValueError(f"Cannot complete job in {job.status} status")
        
        job.status = JobStatus.COMPLETED.value
        job.result = result
        job.completed_at = datetime.now(timezone.utc)
        
        self.session.commit()
        self.session.refresh(job)
        
        return job
    
    def fail_job(
        self, 
        job_id: str, 
        error: str
    ) -> Optional[Job]:
        """Mark job as failed or dead (if max retries exceeded).
        
        Args:
            job_id: Job ID
            error: Error message
            
        Returns:
            Updated Job or None if not found
        """
        job = self.get_job(job_id)
        if job is None:
            return None
        
        if job.status != JobStatus.RUNNING.value:
            raise ValueError(f"Cannot fail job in {job.status} status")
        
        job.error = error
        job.retry_count += 1
        job.completed_at = datetime.now(timezone.utc)
        
        # Check if max retries exceeded -> mark as DEAD
        if job.retry_count >= self._settings.job_max_retries:
            job.status = JobStatus.DEAD.value
            self._alert_dead_job(job)
        else:
            job.status = JobStatus.FAILED.value
        
        self.session.commit()
        self.session.refresh(job)
        
        return job
    
    def _alert_dead_job(self, job: Job):
        """Log alert for permanently failed job (DLQ)."""
        logger.error(
            f"[DLQ] Job permanently failed: "
            f"job_id={job.id}, task_type={job.task_type}, "
            f"error={job.error}"
        )
    
    def retry_job(self, job_id: str) -> Optional[Job]:
        """Retry a failed job if under max retries.
        
        Uses exponential backoff: min(base_delay * 2^retry_count, max_delay)
        
        Args:
            job_id: Job ID
            
        Returns:
            Updated Job or None if not eligible for retry
        """
        job = self.get_job(job_id)
        if job is None:
            return None
        
        if job.status != JobStatus.FAILED.value:
            raise ValueError(f"Can only retry failed jobs, current: {job.status}")
        
        if job.retry_count >= self._settings.job_max_retries:
            return None  # Max retries exceeded
        
        # Reset to pending for retry
        job.status = JobStatus.PENDING.value
        job.error = None
        job.started_at = None
        job.completed_at = None
        
        self.session.commit()
        self.session.refresh(job)
        
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job or None if not found
        """
        return self.session.query(Job).filter(Job.id == job_id).first()
    
    def get_dead_jobs(self) -> list:
        """Query all permanently failed jobs (DLQ).
        
        Returns:
            List of jobs with DEAD status
        """
        return self.session.query(Job).filter(
            Job.status == JobStatus.DEAD.value
        ).all()
    
    def get_retry_delay(self, retry_count: int) -> float:
        """Calculate retry delay with exponential backoff.
        
        Formula: min(base_delay * 2^retry_count, max_delay)
        
        Args:
            retry_count: Current retry attempt number
            
        Returns:
            Delay in seconds before next retry
        """
        base = self._settings.job_retry_base_delay
        max_delay = self._settings.job_retry_max_delay
        
        delay = base * (2 ** retry_count)
        return min(delay, max_delay)
    
    def close(self):
        """Close the session."""
        if self._session:
            self._session.close()
            self._session = None


# Singleton instance
_job_store: Optional[JobStore] = None


def get_job_store() -> JobStore:
    """Get or create JobStore singleton."""
    global _job_store
    if _job_store is None:
        _job_store = JobStore()
    return _job_store
