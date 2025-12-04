from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from app.services.market_data import MarketDataService
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Data Service", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

market_data_service = MarketDataService()

@app.on_event("startup")
async def startup_event():
    await market_data_service.start_stream()

@app.on_event("shutdown")
async def shutdown_event():
    await market_data_service.stop_stream()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await market_data_service.connect_websocket(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            # Handle incoming messages (e.g., subscription requests)
            if data.get("action") == "subscribe":
                tickers = data.get("tickers", [])
                await market_data_service.connect_websocket(websocket, tickers)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        market_data_service.disconnect_websocket(websocket)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "data"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
