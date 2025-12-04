from enum import Enum
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

class TaskType(str, Enum):
    VALUATION = "valuation"
    BACKTEST = "backtest"
    OPTIMIZATION = "optimization"
    FACTOR_COMPUTE = "factor_compute"

class JobPayload(BaseModel):
    job_id: str
    task_type: TaskType
    params: Dict[str, Any]
    created_at: datetime = datetime.now()
    status: str = "pending"

    @staticmethod
    def create(task_type: TaskType, params: Dict[str, Any]) -> 'JobPayload':
        return JobPayload(
            job_id=str(uuid.uuid4()),
            task_type=task_type,
            params=params,
            created_at=datetime.now()
        )
