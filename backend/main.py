"""DCA Valuation Engine - Main Application Entry Point.

FastAPI application with:
- Centralized configuration via Settings
- Structured JSON logging with correlation IDs
- Health and readiness endpoints
- CORS from settings
"""

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.v1.router import api_router
from app.api.v1.endpoints import quant
from app.api.v1.endpoints import health as health_endpoints
from app.core.database import init_db
from app.core.logging_config import setup_logging, set_request_id, request_id_var
from app.core.startup import startup_service
from config.settings import get_settings

# Configure Logging
setup_logging()
logger = logging.getLogger(__name__)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware to set correlation ID for request tracing.
    
    Uses X-Request-ID header if provided, otherwise generates a UUID.
    The correlation ID is available in all logs during the request.
    """
    
    async def dispatch(self, request: Request, call_next):
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        set_request_id(request_id)
        
        # Process request
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.
    
    Startup:
    - Initialize database
    - Run Smart Catch-Up Service to backfill missing data
    
    Shutdown:
    - Cleanup resources
    """
    # Startup
    logger.info("[START] Starting DCA Valuation Engine...")
    init_db()
    await startup_service.run_startup_tasks()
    logger.info("[READY] Server ready to accept requests")
    
    yield
    
    # Shutdown
    logger.info("[SHUTDOWN] Shutting down...")


# Get settings
settings = get_settings()

app = FastAPI(
    title="DCA Valuation Engine", 
    version="1.0.0",
    lifespan=lifespan
)

# Add correlation ID middleware (must be added before CORS)
app.add_middleware(CorrelationIdMiddleware)

# CORS Configuration from settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router, prefix="/api/v1")
app.include_router(quant.router, prefix="/api/v1/quant", tags=["quant"])
app.include_router(health_endpoints.router, prefix="/api/v1", tags=["health"])


@app.get("/")
def read_root():
    return {"message": "DCA Valuation Engine API is running"}
