from abc import ABC, abstractmethod
import redis
import json
import os
from typing import Optional, Dict, Any

class JobBroker(ABC):
    @abstractmethod
    def enqueue_job(self, queue_name: str, job_data: Dict[str, Any]):
        pass

    @abstractmethod
    def dequeue_job(self, queue_name: str, timeout: int = 0) -> Optional[Dict[str, Any]]:
        pass

class RedisJobBroker(JobBroker):
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        # Allow override via env vars
        host = os.getenv('REDIS_HOST', host)
        port = int(os.getenv('REDIS_PORT', port))
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    def enqueue_job(self, queue_name: str, job_data: Dict[str, Any]):
        """Push job to the tail of the list"""
        self.redis.rpush(queue_name, json.dumps(job_data))

    def dequeue_job(self, queue_name: str, timeout: int = 0) -> Optional[Dict[str, Any]]:
        """Pop job from the head of the list (blocking)"""
        # blpop returns (queue_name, data) or None
        result = self.redis.blpop(queue_name, timeout=timeout)
        if result:
            return json.loads(result[1])
        return None

# Singleton instance or factory
def get_broker() -> JobBroker:
    return RedisJobBroker()
