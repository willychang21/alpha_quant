from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import api_router
from app.api.v1.endpoints import quant
from app.core.database import init_db
from app.core.logging_config import setup_logging
import logging

# Configure Logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="DCA Valuation Engine", version="1.0.0")

# Initialize Database
init_db()

# CORS Configuration
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router, prefix="/api/v1")
app.include_router(quant.router, prefix="/api/v1/quant", tags=["quant"])

@app.get("/")
def read_root():
    return {"message": "DCA Valuation Engine API is running"}
