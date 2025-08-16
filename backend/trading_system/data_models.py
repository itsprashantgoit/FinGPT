"""
Data models for the FinGPT Trading System
Optimized for memory efficiency and MongoDB storage
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class ExchangeType(str, Enum):
    MEXC = "mexc"
    BINANCE = "binance"
    YAHOO = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class KlineData:
    """Standard kline/candlestick data structure"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    timestamp: int  # Unix timestamp in milliseconds
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    interval: str  # 1m, 5m, 15m, 1h, 4h, 1d
    exchange: ExchangeType
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class OrderBookData:
    """Order book depth data structure"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    timestamp: int
    bids: List[List[float]]  # [[price, quantity], ...]
    asks: List[List[float]]  # [[price, quantity], ...]
    exchange: ExchangeType
    created_at: datetime = Field(default_factory=datetime.utcnow)


@dataclass
class TradeData:
    """Individual trade/transaction data"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    timestamp: int
    price: float
    quantity: float
    side: OrderSide
    exchange: ExchangeType
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TradingSignal(BaseModel):
    """AI-generated trading signal"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    action: OrderSide  # buy/sell
    confidence: float = Field(ge=0.0, le=1.0)  # 0-1 confidence score
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str  # AI reasoning for the signal
    strategy_used: str  # Which strategy generated this signal
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


class TechnicalIndicators(BaseModel):
    """Technical analysis indicators"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    timestamp: int
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    volume_sma: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Portfolio(BaseModel):
    """Portfolio/account information"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    name: str
    initial_balance: float
    current_balance: float
    available_balance: float
    total_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    max_drawdown: float = 0.0
    sharpe_ratio: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Position(BaseModel):
    """Active trading position"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    portfolio_id: str
    symbol: str
    side: OrderSide
    size: float  # Position size
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Order(BaseModel):
    """Trading order"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    portfolio_id: str
    symbol: str
    side: OrderSide
    type: OrderType
    size: float
    price: Optional[float] = None  # None for market orders
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    filled_price: Optional[float] = None
    strategy_id: Optional[str] = None
    signal_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    filled_at: Optional[datetime] = None


class PerformanceMetrics(BaseModel):
    """Performance tracking metrics"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    portfolio_id: str
    date: datetime
    total_return: float
    daily_return: float
    cumulative_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: Optional[float] = None  # in hours
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MarketSentiment(BaseModel):
    """Market sentiment analysis data"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    timestamp: int
    sentiment_score: float = Field(ge=-1.0, le=1.0)  # -1 to 1
    confidence: float = Field(ge=0.0, le=1.0)
    source: str  # news, social, technical
    text_analyzed: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RiskMetrics(BaseModel):
    """Real-time risk management metrics"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    portfolio_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    var_1d: float  # Value at Risk (1 day)
    var_1w: float  # Value at Risk (1 week)
    portfolio_beta: float
    concentration_risk: float  # Concentration in single asset
    correlation_risk: float  # Average correlation between positions
    leverage_ratio: float
    margin_usage: float = Field(ge=0.0, le=1.0)  # 0-100%
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Database collection names
COLLECTIONS = {
    "klines": "market_klines",
    "orderbook": "market_orderbook", 
    "trades": "market_trades",
    "signals": "trading_signals",
    "indicators": "technical_indicators",
    "portfolios": "portfolios",
    "positions": "positions",
    "orders": "orders",
    "performance": "performance_metrics",
    "sentiment": "market_sentiment",
    "risk": "risk_metrics"
}