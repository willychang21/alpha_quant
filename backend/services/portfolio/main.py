from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints import portfolios, market
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Portfolio Service", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Portfolio and Market Routers (Market is often shared or separate, putting here for now)
app.include_router(portfolios.router, prefix="/api/v1/portfolio", tags=["portfolio"])
app.include_router(market.router, prefix="/api/v1/market", tags=["market"])

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "portfolio"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
