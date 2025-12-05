import asyncio
import websockets
import json
import logging
from typing import Callable, List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self, url: str, message_handler: Callable[[Dict], None]):
        """
        Manages a WebSocket connection for real-time data.
        
        Args:
            url (str): WebSocket URL.
            message_handler (Callable): Function to call with parsed JSON message.
        """
        self.url = url
        self.handler = message_handler
        self.connection = None
        self.running = False
        self.subscriptions: List[Dict] = []
        
    async def connect(self):
        """Establishes the WebSocket connection."""
        try:
            logger.info(f"Connecting to {self.url}...")
            self.connection = await websockets.connect(self.url)
            self.running = True
            logger.info("Connected.")
            
            # Resubscribe if we have pending subscriptions (reconnection logic)
            if self.subscriptions:
                await self._resubscribe()
                
            # Start listening loop
            asyncio.create_task(self._listen())
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.running = False
            # Retry logic could go here
            
    async def disconnect(self):
        """Closes the connection."""
        self.running = False
        if self.connection:
            await self.connection.close()
            logger.info("Disconnected.")
            
    async def subscribe(self, channel: str, symbols: List[str]):
        """
        Sends a subscription message.
        Format depends on the API (e.g., Alpaca, Coinbase).
        This is a generic implementation.
        """
        msg = {
            "action": "subscribe",
            "channel": channel,
            "symbols": symbols
        }
        self.subscriptions.append(msg)
        
        if self.connection:
            try:
                await self.connection.send(json.dumps(msg))
                logger.info(f"Subscribed: {msg}")
            except Exception as e:
                logger.error(f"Failed to send subscription: {e}")
            
    async def _resubscribe(self):
        """Resends all subscriptions after reconnection."""
        for msg in self.subscriptions:
            await self.connection.send(json.dumps(msg))
            
    async def _listen(self):
        """Main loop to receive messages."""
        try:
            while self.running and self.connection:
                try:
                    message = await self.connection.recv()
                    data = json.loads(message)
                    
                    # Handle heartbeat or specific protocol messages here if needed
                    
                    # Pass to handler
                    if self.handler:
                        self.handler(data)
                        
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("Connection closed. Reconnecting...")
                    await self.connect()
                    break
                except Exception as e:
                    logger.error(f"Error receiving message: {e}")
                    
        except Exception as e:
            logger.error(f"Listen loop error: {e}")
