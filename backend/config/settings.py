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
    
    # Data Freshness (Phase 3)
    data_freshness_threshold_hours: float = 24.0
    
    # Circuit Breaker (Phase 3)
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: float = 60.0
    
    # Rate Limiting
    rate_limit_requests_per_second: float = 2.0
    rate_limit_burst_size: int = 5
    
    # Logging
    log_dev_mode: bool = False
    
    # Valuation Defaults
    valuation_wacc: float = 0.09
    valuation_growth_rate: float = 0.05
    valuation_terminal_growth: float = 0.02
    
    # Optimization Defaults
    optimization_max_weight: float = 0.10
    optimization_risk_aversion: float = 1.0
    
    # Factor Weights (JSON string)
    factor_weights_bull: str = '{"value": 0.3, "growth": 0.4, "momentum": 0.3}'
    factor_weights_bear: str = '{"value": 0.5, "growth": 0.2, "momentum": 0.3}'

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"  # Allow extra fields in .env
    }


@lru_cache
def get_settings() -> Settings:
    """Get cached Settings instance.
    
    Uses lru_cache to ensure the same Settings instance is returned
    on every call, providing singleton behavior.
    """
    return Settings()
