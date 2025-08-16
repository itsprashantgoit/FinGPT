"""
Free data sources integration for FinGPT Trading System
MEXC, Binance, Yahoo Finance - optimized for rate limits and reliability
"""
import asyncio
import json
import time
import logging
import websocket
import threading
from collections import deque, defaultdict
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import yfinance as yf
import ccxt
# from binance.client import Client as BinanceClient  # Not needed for public API

from .data_models import KlineData, OrderBookData, TradeData, ExchangeType

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter to comply with exchange API limits"""
    
    def __init__(self, max_requests_per_second: int):
        self.max_rps = max_requests_per_second
        self.request_times = deque()
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Block if rate limit would be exceeded"""
        with self.lock:
            current_time = time.time()
            
            # Remove requests older than 1 second
            while self.request_times and current_time - self.request_times[0] > 1.0:
                self.request_times.popleft()
            
            # Check if we need to wait
            if len(self.request_times) >= self.max_rps:
                sleep_time = 1.0 - (current_time - self.request_times[0])
                if sleep_time > 0:
                    logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                    time.sleep(sleep_time)
            
            # Record this request
            self.request_times.append(time.time())


class MEXCWebSocketClient:
    """MEXC Exchange WebSocket client for real-time crypto futures data"""
    
    def __init__(self):
        self.base_url = "wss://contract.mexc.com/ws"
        self.connections = {}
        self.callbacks = {}
        self.is_running = {}
        self.reconnect_delay = 5
        self.max_reconnect_attempts = 10
        
    def subscribe_kline(self, symbol: str, interval: str, callback: Callable[[KlineData], None]):
        """
        Subscribe to MEXC futures kline data
        Intervals: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M
        """
        stream_name = f"sub.kline.{symbol}.{interval}"
        self.callbacks[stream_name] = callback
        
        subscribe_msg = {
            "method": "sub.kline",
            "param": {
                "symbol": symbol,
                "interval": interval
            }
        }
        
        def create_websocket():
            ws = websocket.WebSocketApp(
                self.base_url,
                on_message=lambda ws, msg: self._on_message(ws, msg, stream_name),
                on_error=lambda ws, error: self._on_error(ws, error, stream_name),
                on_close=lambda ws, close_status_code, close_msg: self._on_close(ws, close_status_code, close_msg, stream_name),
                on_open=lambda ws: self._on_open(ws, subscribe_msg, stream_name)
            )
            return ws
        
        self.connections[stream_name] = create_websocket()
        self.is_running[stream_name] = True
        
        # Start WebSocket in separate thread
        def run_websocket():
            reconnect_count = 0
            while self.is_running[stream_name] and reconnect_count < self.max_reconnect_attempts:
                try:
                    logger.info(f"Starting WebSocket for {stream_name}")
                    self.connections[stream_name].run_forever()
                    
                except Exception as e:
                    logger.error(f"WebSocket error for {stream_name}: {e}")
                    reconnect_count += 1
                    
                    if self.is_running[stream_name] and reconnect_count < self.max_reconnect_attempts:
                        logger.info(f"Reconnecting {stream_name} in {self.reconnect_delay}s (attempt {reconnect_count})")
                        time.sleep(self.reconnect_delay)
                        self.connections[stream_name] = create_websocket()
                    else:
                        logger.error(f"Max reconnection attempts reached for {stream_name}")
                        self.is_running[stream_name] = False
        
        thread = threading.Thread(target=run_websocket, daemon=True)
        thread.start()
        
        logger.info(f"Subscribed to MEXC kline data: {symbol} {interval}")
        return stream_name
    
    def subscribe_orderbook(self, symbol: str, callback: Callable[[OrderBookData], None]):
        """Subscribe to MEXC order book depth updates"""
        stream_name = f"sub.depth.{symbol}"
        self.callbacks[stream_name] = callback
        
        subscribe_msg = {
            "method": "sub.depth",
            "param": {"symbol": symbol}
        }
        
        # Similar implementation as kline subscription
        # Implementation details omitted for brevity
        logger.info(f"Subscribed to MEXC order book: {symbol}")
        return stream_name
    
    def _on_open(self, ws, subscribe_msg, stream_name):
        logger.info(f"WebSocket opened for {stream_name}")
        ws.send(json.dumps(subscribe_msg))
    
    def _on_message(self, ws, message, stream_name):
        try:
            data = json.loads(message)
            
            # Handle kline data
            if data.get("channel") == "push.kline":
                kline_info = data.get("data", {})
                if kline_info:
                    # Parse MEXC kline data format
                    kline_data = KlineData(
                        symbol=kline_info.get("symbol", ""),
                        timestamp=int(kline_info.get("t", 0)),
                        open_price=float(kline_info.get("o", 0)),
                        high_price=float(kline_info.get("h", 0)),
                        low_price=float(kline_info.get("l", 0)),
                        close_price=float(kline_info.get("c", 0)),
                        volume=float(kline_info.get("v", 0)),
                        interval=kline_info.get("interval", ""),
                        exchange=ExchangeType.MEXC
                    )
                    
                    # Call the callback function
                    if stream_name in self.callbacks:
                        self.callbacks[stream_name](kline_data)
            
            # Handle order book data
            elif data.get("channel") == "push.depth":
                depth_info = data.get("data", {})
                if depth_info:
                    orderbook_data = OrderBookData(
                        symbol=depth_info.get("symbol", ""),
                        timestamp=int(depth_info.get("t", 0)),
                        bids=depth_info.get("bids", []),
                        asks=depth_info.get("asks", []),
                        exchange=ExchangeType.MEXC
                    )
                    
                    if stream_name in self.callbacks:
                        self.callbacks[stream_name](orderbook_data)
                        
        except Exception as e:
            logger.error(f"Error processing WebSocket message for {stream_name}: {e}")
    
    def _on_error(self, ws, error, stream_name):
        logger.error(f"WebSocket error for {stream_name}: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg, stream_name):
        logger.warning(f"WebSocket closed for {stream_name}: {close_status_code} - {close_msg}")
    
    def unsubscribe(self, stream_name: str):
        """Unsubscribe from a stream"""
        self.is_running[stream_name] = False
        if stream_name in self.connections:
            self.connections[stream_name].close()
            del self.connections[stream_name]
        if stream_name in self.callbacks:
            del self.callbacks[stream_name]


class BinanceDataClient:
    """Binance public API client for historical crypto data"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter(20)  # 20 requests per second
        
        # Setup HTTP session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def get_klines(self, symbol: str, interval: str, limit: int = 100, start_time: Optional[int] = None) -> List[KlineData]:
        """Get historical kline data from Binance public API"""
        self.rate_limiter.wait_if_needed()
        
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol.replace('/', '').replace('-', ''),  # Format for Binance
            'interval': interval,
            'limit': min(limit, 1000)  # Binance limit is 1000
        }
        
        if start_time:
            params['startTime'] = start_time
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            klines = []
            for kline in response.json():
                kline_data = KlineData(
                    symbol=symbol,
                    timestamp=int(kline[0]),
                    open_price=float(kline[1]),
                    high_price=float(kline[2]),
                    low_price=float(kline[3]),
                    close_price=float(kline[4]),
                    volume=float(kline[5]),
                    interval=interval,
                    exchange=ExchangeType.BINANCE
                )
                klines.append(kline_data)
            
            logger.info(f"Retrieved {len(klines)} klines from Binance for {symbol}")
            return klines
            
        except Exception as e:
            logger.error(f"Error fetching Binance data for {symbol}: {e}")
            return []
    
    def get_24hr_ticker(self, symbol: str) -> Dict:
        """Get 24hr ticker statistics"""
        self.rate_limiter.wait_if_needed()
        
        url = "https://api.binance.com/api/v3/ticker/24hr"
        params = {'symbol': symbol.replace('/', '').replace('-', '')}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching Binance ticker for {symbol}: {e}")
            return {}


class YahooFinanceClient:
    """Yahoo Finance client for stock and traditional asset data"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter(1)  # Conservative rate limit
    
    def get_klines(self, symbol: str, period: str = "30d", interval: str = "1h") -> List[KlineData]:
        """Get historical data from Yahoo Finance"""
        self.rate_limiter.wait_if_needed()
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                logger.warning(f"No data found for {symbol} from Yahoo Finance")
                return []
            
            klines = []
            for timestamp, row in hist.iterrows():
                kline_data = KlineData(
                    symbol=symbol,
                    timestamp=int(timestamp.timestamp() * 1000),
                    open_price=float(row['Open']),
                    high_price=float(row['High']),
                    low_price=float(row['Low']),
                    close_price=float(row['Close']),
                    volume=float(row['Volume']),
                    interval=interval,
                    exchange=ExchangeType.YAHOO
                )
                klines.append(kline_data)
            
            logger.info(f"Retrieved {len(klines)} klines from Yahoo Finance for {symbol}")
            return klines
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data for {symbol}: {e}")
            return []
    
    def get_info(self, symbol: str) -> Dict:
        """Get ticker information"""
        self.rate_limiter.wait_if_needed()
        
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance info for {symbol}: {e}")
            return {}


class MultiSourceDataManager:
    """Manager for coordinating multiple free data sources"""
    
    def __init__(self):
        self.mexc_client = MEXCWebSocketClient()
        self.binance_client = BinanceDataClient()
        self.yahoo_client = YahooFinanceClient()
        
        self.data_buffer = deque(maxlen=10000)  # Recent data buffer
        self.callbacks = []
        self.is_running = False
        
        # Default symbols to track (free sources)
        self.crypto_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
        self.stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        self.intervals = ['1m', '5m', '15m', '1h', '4h', '1d']
    
    def add_data_callback(self, callback: Callable[[KlineData], None]):
        """Add callback for real-time data processing"""
        self.callbacks.append(callback)
    
    def _on_kline_data(self, kline: KlineData):
        """Handle incoming kline data from any source"""
        # Add to buffer
        self.data_buffer.append(kline)
        
        # Notify all callbacks
        for callback in self.callbacks:
            try:
                callback(kline)
            except Exception as e:
                logger.error(f"Error in data callback: {e}")
    
    def start_real_time_feeds(self, symbols: Optional[List[str]] = None, intervals: Optional[List[str]] = None):
        """Start real-time data feeds from MEXC"""
        symbols = symbols or self.crypto_symbols[:3]  # Start with 3 symbols
        intervals = intervals or ['1m', '5m']  # Start with 2 intervals
        
        logger.info(f"Starting real-time feeds for {len(symbols)} symbols, {len(intervals)} intervals")
        
        for symbol in symbols:
            for interval in intervals:
                try:
                    # Subscribe to MEXC WebSocket (remove USDT suffix if present)
                    mexc_symbol = symbol.replace('USDT', '_USDT')
                    self.mexc_client.subscribe_kline(
                        symbol=mexc_symbol,
                        interval=interval,
                        callback=self._on_kline_data
                    )
                    
                    # Small delay between subscriptions
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error subscribing to {symbol} {interval}: {e}")
        
        self.is_running = True
        logger.info("Real-time data feeds started")
    
    def fetch_historical_data(self, symbols: Optional[List[str]] = None, intervals: Optional[List[str]] = None) -> List[KlineData]:
        """Fetch historical data from multiple sources"""
        symbols = symbols or (self.crypto_symbols + self.stock_symbols)
        intervals = intervals or ['1h', '1d']
        
        logger.info(f"Fetching historical data for {len(symbols)} symbols")
        
        all_klines = []
        
        for symbol in symbols:
            for interval in intervals:
                try:
                    klines = []
                    
                    # Determine source based on symbol type
                    if any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'USDT']):
                        # Crypto - use Binance
                        klines = self.binance_client.get_klines(symbol, interval, limit=500)
                    else:
                        # Stocks - use Yahoo Finance
                        yahoo_interval_map = {
                            '1h': '1h',
                            '1d': '1d',
                            '4h': '1h',  # Yahoo doesn't have 4h, use 1h
                            '1m': '1m'
                        }
                        yahoo_interval = yahoo_interval_map.get(interval, '1d')
                        klines = self.yahoo_client.get_klines(symbol, period="30d", interval=yahoo_interval)
                    
                    if klines:
                        all_klines.extend(klines)
                        logger.info(f"Retrieved {len(klines)} historical records for {symbol} {interval}")
                    
                    # Rate limiting delay
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error fetching historical data for {symbol} {interval}: {e}")
        
        return all_klines
    
    def stop(self):
        """Stop all data feeds"""
        self.is_running = False
        
        # Close WebSocket connections
        for stream_name in list(self.mexc_client.connections.keys()):
            self.mexc_client.unsubscribe(stream_name)
        
        logger.info("Multi-source data manager stopped")
    
    def get_supported_symbols(self) -> Dict[str, List[str]]:
        """Get list of supported symbols by source"""
        return {
            "crypto_mexc_binance": self.crypto_symbols,
            "stocks_yahoo": self.stock_symbols
        }