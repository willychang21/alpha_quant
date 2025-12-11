"""Health and readiness endpoints for monitoring.

Provides:
- /health - Basic liveness check
- /ready - Readiness check with database and data freshness
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    """Liveness probe - returns healthy if the application is running.
    
    Returns:
        JSON with status "healthy"
    """
    return {"status": "healthy"}


@router.get("/ready")
async def ready(db: Session = Depends(get_db)):
    """Readiness probe - checks system dependencies.
    
    Checks:
    - Database connectivity
    - Data freshness (hours since last price update)
    
    Returns:
        JSON with status ("ready" or "not_ready") and check details
    """
    checks = {}
    
    # Database connectivity check
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        logger.error(f"Database check failed: {e}")
        checks["database"] = f"error: {str(e)}"
    
    # Data freshness check
    data_lag_hours = _compute_data_lag()
    checks["data_lag_hours"] = data_lag_hours
    
    # Determine overall status
    all_ok = all(
        v == "ok" 
        for k, v in checks.items() 
        if k != "data_lag_hours"
    )
    
    # Warn if data is stale (>24 hours)
    if data_lag_hours and data_lag_hours > 24:
        logger.warning(f"Data staleness warning: {data_lag_hours:.1f} hours since last update")
    
    return {
        "status": "ready" if all_ok else "not_ready",
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


def _compute_data_lag() -> float | None:
    """Compute hours since last data update.
    
    Checks the most recent price file modification time.
    
    Returns:
        Hours since last update, or None if data lake not found
    """
    try:
        from quant.data.parquet_io import get_data_lake_path
        
        data_lake = get_data_lake_path()
        prices_path = data_lake / 'raw' / 'prices'
        
        if not prices_path.exists():
            return None
        
        # Find most recent file modification
        latest_mtime = None
        for parquet_file in prices_path.rglob('*.parquet'):
            mtime = parquet_file.stat().st_mtime
            if latest_mtime is None or mtime > latest_mtime:
                latest_mtime = mtime
        
        if latest_mtime is None:
            return None
        
        # Calculate hours since last update
        now = datetime.now().timestamp()
        hours_ago = (now - latest_mtime) / 3600
        return round(hours_ago, 2)
        
    except Exception as e:
        logger.warning(f"Could not compute data lag: {e}")
        return None


@router.get("/health/deep")
async def deep_health(db: Session = Depends(get_db)):
    """Deep health check - validates all system dependencies.
    
    Checks:
    - Database connectivity
    - Data lake path exists
    - Data freshness status
    
    Returns:
        JSON with status ("healthy" or "degraded") and check details
    """
    from config.settings import get_settings
    from core.freshness import get_data_freshness_service
    
    settings = get_settings()
    checks = {}
    
    # Database check
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "healthy"
    except Exception as e:
        checks["database"] = f"unhealthy: {str(e)}"
    
    # Data lake check
    data_lake_path = Path(settings.data_lake_path)
    if data_lake_path.exists():
        checks["data_lake"] = "healthy"
    else:
        checks["data_lake"] = "unhealthy: path not found"
    
    # Data freshness check
    freshness_service = get_data_freshness_service()
    is_fresh, lag_hours, last_date = freshness_service.get_freshness_status()
    
    data_freshness_hours = lag_hours if lag_hours != float('inf') else None
    data_last_date = str(last_date) if last_date else None
    
    if is_fresh:
        checks["data_freshness"] = "healthy"
    else:
        checks["data_freshness"] = f"stale: {lag_hours:.1f}h lag" if lag_hours != float('inf') else "unhealthy: no data"
    
    # Determine overall status (freshness warning doesn't make system degraded)
    core_healthy = all(
        v == "healthy" for k, v in checks.items() 
        if k in ("database", "data_lake")
    )
    
    return {
        "status": "healthy" if core_healthy else "degraded",
        "checks": checks,
        "data_freshness_hours": data_freshness_hours,
        "data_last_date": data_last_date,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

