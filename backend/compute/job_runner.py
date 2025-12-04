from .broker import get_broker
from .tasks import TaskType, JobPayload
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class JobRunner:
    def __init__(self):
        self.broker = get_broker()

    def submit_job(self, task_type: TaskType, params: Dict[str, Any]) -> str:
        """
        Submit a job to the compute queue.
        Returns the job_id.
        """
        job = JobPayload.create(task_type, params)
        
        # Determine queue based on task type (can be specialized later)
        queue_name = "quant_tasks_default"
        
        logger.info(f"Submitting job {job.job_id} of type {task_type} to {queue_name}")
        
        # Serialize Pydantic model to JSON-compatible dict
        import json
        job_dict = json.loads(job.json())
        self.broker.enqueue_job(queue_name, job_dict)
        
        return job.job_id

# Singleton
_runner = JobRunner()

def submit_job(task_type: TaskType, params: Dict[str, Any]) -> str:
    return _runner.submit_job(task_type, params)
