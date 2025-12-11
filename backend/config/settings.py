"""Centralized configuration using Pydantic Settings.

This module provides a Settings class that loads configuration from:
1. Environment variables (highest priority)
2. .env file (fallback)
3. Default values (lowest priority)

Usage:
    from config.settings import get_settings
    settings = get_settings()
    print(settings.database_url)
"""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support.
    
    All settings can be overridden via environment variables.
    For list types like cors_origins, use JSON format:
        CORS_ORIGINS='["http://localhost:5173","http://example.com"]'
    """
    
    # Database
    database_url: str = "sqlite:///./data/database.sqlite"
    
    # Data Lake
    data_lake_path: str = "./data_lake"
    
    # API
    cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    # MLflow
    mlflow_tracking_uri: str = "./mlruns"
    
    # Feature Flags
    debug_mode: bool = False
    
    # Job Configuration
    job_max_retries: int = 3
    job_retry_base_delay: float = 4.0
    job_retry_max_delay: float = 60.0
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache
def get_settings() -> Settings:
    """Get cached Settings instance.
    
    Uses lru_cache to ensure the same Settings instance is returned
    on every call, providing singleton behavior.
    """
    return Settings()
