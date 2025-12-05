import sys
import os
import asyncio
import websockets
import json
import pytest
import logging

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.data.websocket import WebSocketManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

received_messages = []

def message_handler(data):
    """Callback for received messages."""
    logger.info(f"Received: {data}")
    received_messages.append(data)

async def mock_server(websocket):
    """
    Mock WebSocket server that sends dummy data.
    """
    try:
        async for message in websocket:
            data = json.loads(message)
            logger.info(f"Server received: {data}")
            
            if data.get("action") == "subscribe":
                # Send confirmation
                await websocket.send(json.dumps({"status": "subscribed", "channel": data.get("channel")}))
                
                # Send dummy ticks
                for i in range(3):
                    tick = {
                        "type": "trade",
                        "symbol": "AAPL",
                        "price": 150.0 + i,
                        "size": 100
                    }
                    await websocket.send(json.dumps(tick))
                    await asyncio.sleep(0.1)
    except websockets.exceptions.ConnectionClosed:
        pass

async def run_test():
    print("\nTesting WebSocket Manager...")
    
    # 1. Start Mock Server
    server = await websockets.serve(mock_server, "localhost", 8765)
    
    # 2. Initialize Manager
    manager = WebSocketManager("ws://localhost:8765", message_handler)
    
    # 3. Connect
    await manager.connect()
    
    # 4. Subscribe
    await manager.subscribe("trades", ["AAPL"])
    
    # 5. Wait for messages
    await asyncio.sleep(1)
    
    # 6. Verify
    print(f"Received {len(received_messages)} messages.")
    
    # Expect: 1 subscription confirmation + 3 ticks = 4 messages
    assert len(received_messages) >= 4
    assert received_messages[0]["status"] == "subscribed"
    assert received_messages[1]["symbol"] == "AAPL"
    
    # 7. Cleanup
    await manager.disconnect()
    server.close()
    await server.wait_closed()
    
    print("✅ WebSocket Manager Passed")

def test_websocket_realtime():
    # Run async test
    asyncio.run(run_test())

if __name__ == "__main__":
    test_websocket_realtime()
