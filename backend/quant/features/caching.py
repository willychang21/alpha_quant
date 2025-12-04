from abc import ABC, abstractmethod
from typing import Optional, Any
import redis
import pickle
import os
import logging

logger = logging.getLogger(__name__)

class FeatureCache(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int = 3600):
        pass

class RedisFeatureCache(FeatureCache):
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 1):
        # Use DB 1 for features (DB 0 is for jobs)
        host = os.getenv('REDIS_HOST', host)
        port = int(os.getenv('REDIS_PORT', port))
        try:
            self.redis = redis.Redis(host=host, port=port, db=db)
            self.redis.ping()
            logger.info(f"Connected to Redis Feature Cache at {host}:{port}/{db}")
        except redis.ConnectionError:
            logger.warning("Could not connect to Redis. Cache will be disabled.")
            self.redis = None

    def get(self, key: str) -> Optional[Any]:
        if not self.redis:
            return None
        try:
            data = self.redis.get(key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
        return None

    def set(self, key: str, value: Any, ttl: int = 3600):
        if not self.redis:
            return
        try:
            data = pickle.dumps(value)
            self.redis.setex(key, ttl, data)
        except Exception as e:
            logger.error(f"Error setting cache: {e}")

class NoOpCache(FeatureCache):
    def get(self, key: str) -> Optional[Any]:
        return None

    def set(self, key: str, value: Any, ttl: int = 3600):
        pass

def get_cache() -> FeatureCache:
    # Factory to return appropriate cache
    # For now, try Redis, fallback to NoOp
    return RedisFeatureCache()
