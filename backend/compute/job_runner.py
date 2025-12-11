"""Job Runner with persistent JobStore integration.

Provides job submission with:
- Persistent job tracking via JobStore
- Queue-based processing via broker
- Status updates throughout lifecycle
"""

import json
import logging
from typing import Any, Dict, Optional

from .broker import get_broker
from .job_store import JobStore, JobStatus, get_job_store
from .tasks import TaskType, JobPayload

logger = logging.getLogger(__name__)


class JobRunner:
    """Job runner with persistent storage integration.
    
    Combines queue-based job execution with SQLite-backed persistence
    for reliable job tracking and retry support.
    """
    
    def __init__(self, job_store: Optional[JobStore] = None):
        self.broker = get_broker()
        self.job_store = job_store or get_job_store()

    def submit_job(self, task_type: TaskType, params: Dict[str, Any]) -> str:
        """Submit a job to the compute queue with persistence.
        
        Creates a persistent job record, then enqueues for processing.
        
        Args:
            task_type: Type of job (valuation, backtest, etc.)
            params: Job parameters
            
        Returns:
            Job ID for tracking
        """
        # Create persistent job record
        job_record = self.job_store.create_job(
            task_type=task_type.value if hasattr(task_type, 'value') else str(task_type),
            params=params
        )
        job_id = job_record.id
        
        # Create queue payload
        job_payload = JobPayload.create(task_type, params)
        # Override with our persistent job_id
        job_payload.job_id = job_id
        
        # Determine queue based on task type
        queue_name = "quant_tasks_default"
        
        logger.info(f"Submitting job {job_id} of type {task_type} to {queue_name}")
        
        # Enqueue for processing
        job_dict = json.loads(job_payload.json())
        self.broker.enqueue_job(queue_name, job_dict)
        
        return job_id

    def start_job(self, job_id: str) -> bool:
        """Mark job as running (called by worker).
        
        Args:
            job_id: Job ID
            
        Returns:
            True if successfully started
        """
        try:
            self.job_store.start_job(job_id)
            logger.info(f"Job {job_id} started")
            return True
        except Exception as e:
            logger.error(f"Failed to start job {job_id}: {e}")
            return False

    def complete_job(self, job_id: str, result: Optional[Dict[str, Any]] = None) -> bool:
        """Mark job as completed (called by worker).
        
        Args:
            job_id: Job ID
            result: Job result data
            
        Returns:
            True if successfully completed
        """
        try:
            self.job_store.complete_job(job_id, result)
            logger.info(f"Job {job_id} completed")
            return True
        except Exception as e:
            logger.error(f"Failed to complete job {job_id}: {e}")
            return False

    def fail_job(self, job_id: str, error: str) -> bool:
        """Mark job as failed (called by worker).
        
        Args:
            job_id: Job ID
            error: Error message
            
        Returns:
            True if successfully marked as failed
        """
        try:
            job = self.job_store.fail_job(job_id, error)
            logger.warning(f"Job {job_id} failed: {error}")
            
            # Check if eligible for retry
            if job and self.job_store.retry_job(job_id):
                delay = self.job_store.get_retry_delay(job.retry_count)
                logger.info(f"Job {job_id} scheduled for retry in {delay:.1f}s")
            
            return True
        except Exception as e:
            logger.error(f"Failed to mark job {job_id} as failed: {e}")
            return False

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get current job status.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status dict or None if not found
        """
        job = self.job_store.get_job(job_id)
        if job is None:
            return None
        
        return {
            "id": job.id,
            "task_type": job.task_type,
            "status": job.status,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "retry_count": job.retry_count,
            "result": job.result,
            "error": job.error,
        }


# Singleton
_runner: Optional[JobRunner] = None


def get_job_runner() -> JobRunner:
    """Get or create JobRunner singleton."""
    global _runner
    if _runner is None:
        _runner = JobRunner()
    return _runner


def submit_job(task_type: TaskType, params: Dict[str, Any]) -> str:
    """Convenience function to submit a job."""
    return get_job_runner().submit_job(task_type, params)
