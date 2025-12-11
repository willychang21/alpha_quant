"""Market Data Service with rate limiting and error handling.

Provides high-performance market data fetching with:
- Token bucket rate limiting for yfinance API
- Automatic retry with exponential backoff for transient errors
- In-memory caching with TTL
- Async/await support for parallel fetching
"""

import asyncio
import logging
import concurrent.futures
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from fastapi import WebSocket

from quant.data.realtime.interface import StreamClient
from quant.data.realtime.mock_stream import MockStreamClient
from quant.data.realtime.connection_manager import ConnectionManager

# Import new infrastructure
from core.rate_limiter import get_yfinance_rate_limiter, TokenBucketRateLimiter
from core.error_handler import with_retry, ErrorHandler
from core.structured_logger import get_structured_logger, set_correlation_id

logger = get_structured_logger("MarketDataService")


class MarketDataService:
    """
    Service for retrieving market data asynchronously with caching.
    
    Features:
    - Rate limiting via token bucket algorithm
    - Automatic retry with exponential backoff
    - TTL-based caching to reduce API calls
    - Thread pool for blocking I/O
    
    Usage:
        service = MarketDataService()
        data = await service.get_ticker_data("AAPL")
    """
    
    _instance: Optional['MarketDataService'] = None
    _cache: Dict[str, Any] = {}
    _cache_expiry: Dict[str, datetime] = {}
    _executor: concurrent.futures.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
    _rate_limiter: Optional[TokenBucketRateLimiter] = None
    
    # Streaming components
    stream_client: Optional[StreamClient] = None
    connection_manager: Optional[ConnectionManager] = None
    
    # Cache TTL settings
    TTL_INFO: timedelta = timedelta(minutes=60)
    TTL_PRICE: timedelta = timedelta(minutes=1)
    TTL_FINANCIALS: timedelta = timedelta(days=1)

    def __new__(cls) -> 'MarketDataService':
        if cls._instance is None:
            cls._instance = super(MarketDataService, cls).__new__(cls)
            cls._instance._initialize_streaming()
            cls._instance._rate_limiter = get_yfinance_rate_limiter()
        return cls._instance

    def _initialize_streaming(self) -> None:
        """Initialize streaming components."""
        self.connection_manager = ConnectionManager()
        # In production, this would be a real client (e.g., Polygon, Alpaca)
        self.stream_client = MockStreamClient()
        self.stream_client.add_callback(self._handle_stream_update)

    async def start_stream(self) -> None:
        """Start the stream client."""
        if self.stream_client:
            await self.stream_client.connect()

    async def stop_stream(self) -> None:
        """Stop the stream client."""
        if self.stream_client:
            await self.stream_client.disconnect()

    async def connect_websocket(
        self, 
        websocket: WebSocket, 
        tickers: Optional[List[str]] = None
    ) -> None:
        """Handle new WebSocket connection."""
        if self.connection_manager:
            await self.connection_manager.connect(websocket)
            if tickers and self.stream_client:
                await self.connection_manager.subscribe(websocket, tickers)
                await self.stream_client.subscribe(tickers)

    def disconnect_websocket(self, websocket: WebSocket) -> None:
        """Handle WebSocket disconnection."""
        if self.connection_manager:
            self.connection_manager.disconnect(websocket)

    async def _handle_stream_update(self, data: Dict[str, Any]) -> None:
        """Callback for stream updates."""
        if self.connection_manager:
            await self.connection_manager.broadcast(data)

    async def get_ticker_data(self, ticker: str) -> Dict[str, Any]:
        """
        Orchestrates fetching all necessary data for a ticker in parallel.
        
        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            
        Returns:
            Dictionary containing info, financials, and history.
            
        Raises:
            Exception: If all retry attempts fail
        """
        logger.info(f"Fetching data for {ticker}")
        
        loop = asyncio.get_running_loop()
        
        # Rate limit before creating yfinance ticker
        await self._rate_limiter.acquire()
        
        t = yf.Ticker(ticker)
        
        try:
            results = await asyncio.gather(
                self._fetch_info(loop, t, ticker),
                self._fetch_financials(loop, t, ticker),
                self._fetch_history(loop, t, ticker),
                return_exceptions=True
            )
            
            # Handle any exceptions from gather
            info, financials, history = results
            
            # Check for exceptions and handle gracefully
            if isinstance(info, Exception):
                logger.warning(f"Failed to fetch info for {ticker}", exc_info=False)
                info = {}
            if isinstance(financials, Exception):
                logger.warning(f"Failed to fetch financials for {ticker}", exc_info=False)
                financials = {"income": None, "balance": None, "cashflow": None}
            if isinstance(history, Exception):
                logger.warning(f"Failed to fetch history for {ticker}", exc_info=False)
                history = pd.DataFrame()
            
            return {
                "info": info,
                "income": financials.get("income") if isinstance(financials, dict) else None,
                "balance": financials.get("balance") if isinstance(financials, dict) else None,
                "cashflow": financials.get("cashflow") if isinstance(financials, dict) else None,
                "history": history
            }
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            raise e

    async def get_exchange_rate(
        self, 
        from_currency: str, 
        to_currency: str = "USD"
    ) -> float:
        """
        Fetches exchange rate asynchronously with caching.
        
        Args:
            from_currency: Source currency code (e.g., "EUR")
            to_currency: Target currency code (default: "USD")
            
        Returns:
            Exchange rate, or 1.0 if same currency or on error
        """
        if from_currency == to_currency:
            return 1.0
            
        pair = f"{from_currency}{to_currency}=X"
        cache_key = f"fx_{pair}"
        
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
            
        try:
            # Rate limit before API call
            await self._rate_limiter.acquire()
            
            loop = asyncio.get_running_loop()
            rate = await loop.run_in_executor(
                self._executor, 
                self._fetch_fx_rate_sync, 
                pair
            )
            self._set_cache(cache_key, rate, self.TTL_INFO)
            return rate
        except Exception as e:
            logger.error(f"Error fetching exchange rate {pair}: {e}", exc_info=False)
            return 1.0  # Fallback

    @with_retry(max_retries=3, base_delay=1.0, on_data_error=1.0)
    def _fetch_fx_rate_sync(self, pair: str) -> float:
        """Synchronous FX rate fetch with retry."""
        ticker = yf.Ticker(pair)
        rate = ticker.info.get('regularMarketPrice') or ticker.info.get('previousClose')
        if not rate:
            hist = ticker.history(period="1d")
            if not hist.empty:
                rate = hist['Close'].iloc[-1]
        return rate if rate else 1.0

    async def _fetch_info(
        self, 
        loop: asyncio.AbstractEventLoop, 
        ticker_obj: yf.Ticker, 
        ticker_symbol: str
    ) -> Dict[str, Any]:
        """Fetch ticker info with caching and rate limiting."""
        cache_key = f"info_{ticker_symbol}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        # Rate limit API call
        await self._rate_limiter.acquire()
            
        @with_retry(max_retries=2, base_delay=0.5, on_data_error={})
        def fetch() -> Dict[str, Any]:
            data = ticker_obj.info
            # Ensure it's a dict (yfinance sometimes behaves oddly)
            if isinstance(data, pd.DataFrame):
                if not data.empty:
                    return data.iloc[0].to_dict()
                return {}
            return data if isinstance(data, dict) else {}
            
        info = await loop.run_in_executor(self._executor, fetch)
        if not isinstance(info, dict):
            logger.warning(f"Info for {ticker_symbol} is not a dict: {type(info)}")
            info = {}
              
        self._set_cache(cache_key, info, self.TTL_INFO)
        return info

    async def _fetch_financials(
        self, 
        loop: asyncio.AbstractEventLoop, 
        ticker_obj: yf.Ticker, 
        ticker_symbol: str
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """Fetch financial statements with caching and rate limiting."""
        cache_key = f"fin_{ticker_symbol}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        # Rate limit API call
        await self._rate_limiter.acquire()
            
        @with_retry(max_retries=2, base_delay=0.5, on_data_error={"income": None, "balance": None, "cashflow": None})
        def fetch() -> Dict[str, Optional[pd.DataFrame]]:
            return {
                "income": ticker_obj.income_stmt,
                "balance": ticker_obj.balance_sheet,
                "cashflow": ticker_obj.cashflow
            }
            
        financials = await loop.run_in_executor(self._executor, fetch)
        self._set_cache(cache_key, financials, self.TTL_FINANCIALS)
        return financials

    async def _fetch_history(
        self, 
        loop: asyncio.AbstractEventLoop, 
        ticker_obj: yf.Ticker, 
        ticker_symbol: str
    ) -> pd.DataFrame:
        """Fetch price history with caching and rate limiting."""
        cache_key = f"hist_{ticker_symbol}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        
        # Rate limit API call
        await self._rate_limiter.acquire()
            
        @with_retry(max_retries=2, base_delay=0.5, on_data_error=pd.DataFrame())
        def fetch() -> pd.DataFrame:
            return ticker_obj.history(period="2y")
            
        history = await loop.run_in_executor(self._executor, fetch)
        self._set_cache(cache_key, history, self.TTL_PRICE)
        return history

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self._cache:
            if datetime.now() < self._cache_expiry[key]:
                return self._cache[key]
            else:
                del self._cache[key]
                del self._cache_expiry[key]
        return None

    def _set_cache(self, key: str, value: Any, ttl: timedelta) -> None:
        """Set value in cache with TTL."""
        self._cache[key] = value
        self._cache_expiry[key] = datetime.now() + ttl
    
    def clear_cache(self, ticker: Optional[str] = None) -> None:
        """Clear cache for a specific ticker or all caches.
        
        Args:
            ticker: If provided, clear only this ticker's cache. Otherwise clear all.
        """
        if ticker:
            keys_to_remove = [k for k in self._cache if ticker in k]
            for key in keys_to_remove:
                self._cache.pop(key, None)
                self._cache_expiry.pop(key, None)
            logger.info(f"Cleared cache for {ticker}")
        else:
            self._cache.clear()
            self._cache_expiry.clear()
            logger.info("Cleared all cache")

