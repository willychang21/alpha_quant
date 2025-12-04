from fastapi import WebSocket
from typing import List, Dict, Set
import logging
import json

logger = logging.getLogger(__name__)

class ConnectionManager:
    """
    Manages WebSocket connections and broadcasts messages.
    """
    def __init__(self):
        # List of active connections
        self.active_connections: List[WebSocket] = []
        # Map of ticker -> Set[WebSocket] for targeted broadcasting (optional optimization)
        self.subscriptions: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
        # Remove from subscriptions
        for ticker in list(self.subscriptions.keys()):
            if websocket in self.subscriptions[ticker]:
                self.subscriptions[ticker].remove(websocket)
                if not self.subscriptions[ticker]:
                    del self.subscriptions[ticker]
                    
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        # In a real app, we might filter by subscription
        # For now, broadcast everything to everyone (simple MVP)
        
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                dead_connections.append(connection)
                
        for dead in dead_connections:
            self.disconnect(dead)
            
    async def subscribe(self, websocket: WebSocket, tickers: List[str]):
        """
        Register client interest in specific tickers.
        """
        for ticker in tickers:
            if ticker not in self.subscriptions:
                self.subscriptions[ticker] = set()
            self.subscriptions[ticker].add(websocket)
