import sys
import os
import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.market_data import MarketDataService

# Configure logging to file
logging.basicConfig(
    level=logging.INFO,
    filename='test_stage4_debug.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def test_realtime_streaming():
    logging.info("Starting test_realtime_streaming")
    print("Testing Real-time Streaming...", flush=True)
    
    try:
        # 1. Initialize Service
        service = MarketDataService()
        logging.info("Service initialized")
        
        # 2. Mock WebSocket
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()
        
        # 3. Connect and Subscribe
        tickers = ["AAPL", "GOOGL"]
        await service.connect_websocket(mock_ws, tickers)
        logging.info("WebSocket connected")
        print("WebSocket connected and subscribed.", flush=True)
        
        # 4. Start Stream
        await service.start_stream()
        logging.info("Stream started")
        print("Stream started. Waiting for updates...", flush=True)
        
        # 5. Wait for a few updates
        await asyncio.sleep(2.5) 
        
        # 6. Stop Stream
        await service.stop_stream()
        service.disconnect_websocket(mock_ws)
        logging.info("Stream stopped")
        print("Stream stopped.", flush=True)
        
        # 7. Verify
        call_count = mock_ws.send_json.call_count
        logging.info(f"Call count: {call_count}")
        print(f"Mock WebSocket received {call_count} messages.", flush=True)
        
        if call_count >= 2:
            print("Real-time Streaming Test Passed!", flush=True)
        else:
            print("Real-time Streaming Test Failed: Not enough messages received.", flush=True)
            
    except Exception as e:
        logging.error(f"Error in test: {e}")
        print(f"Error: {e}", flush=True)

if __name__ == "__main__":
    asyncio.run(test_realtime_streaming())
