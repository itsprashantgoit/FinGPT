"""
FastAPI routes for FinGPT Trading System
RESTful API endpoints for trading engine interaction
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from .trading_engine import TradingEngine
from .data_models import TradingSignal, Portfolio

logger = logging.getLogger(__name__)

# Global trading engine instance (will be initialized in main server)
trading_engine: Optional[TradingEngine] = None

def get_trading_engine() -> TradingEngine:
    """Dependency to get trading engine instance"""
    if trading_engine is None:
        raise HTTPException(status_code=500, detail="Trading engine not initialized")
    return trading_engine

def set_trading_engine(engine: TradingEngine):
    """Set the global trading engine instance"""
    global trading_engine
    trading_engine = engine

# Create router
router = APIRouter(prefix="/api/trading", tags=["trading"])


# Request/Response Models
class EngineStartRequest(BaseModel):
    symbols: Optional[List[str]] = Field(default=None, description="List of symbols to trade")
    paper_trading: bool = Field(default=True, description="Enable paper trading mode")


class PortfolioCreateRequest(BaseModel):
    user_id: str = Field(description="User ID")
    name: str = Field(description="Portfolio name")
    initial_balance: float = Field(gt=0, description="Initial balance in USD")


class SymbolSubscriptionRequest(BaseModel):
    symbols: List[str] = Field(description="List of symbols to subscribe to")
    intervals: List[str] = Field(default=["1m", "5m"], description="List of intervals")


class EngineStatusResponse(BaseModel):
    is_running: bool
    is_paper_trading: bool
    subscribed_symbols: List[str]
    active_portfolios: int
    total_positions: int
    signals_generated: int
    data_points_processed: int
    strategy_status: Dict[str, Any]
    storage_stats: Dict[str, Any]


class PortfolioSummaryResponse(BaseModel):
    portfolio: Dict[str, Any]
    positions: List[Dict[str, Any]]
    risk_summary: Dict[str, Any]
    performance_metrics: Optional[Dict[str, Any]]


class SignalResponse(BaseModel):
    symbol: str
    action: str
    confidence: float
    price_target: Optional[float]
    reasoning: str
    strategy_used: str
    created_at: str


# Engine Control Endpoints
@router.post("/engine/start")
async def start_trading_engine(
    request: EngineStartRequest,
    background_tasks: BackgroundTasks,
    engine: TradingEngine = Depends(get_trading_engine)
):
    """Start the trading engine with specified symbols"""
    try:
        if engine.is_running:
            return {"message": "Trading engine is already running", "status": "running"}
        
        # Start engine in background
        background_tasks.add_task(engine.start_engine, request.symbols)
        
        return {
            "message": "Trading engine start initiated",
            "symbols": request.symbols or ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
            "paper_trading": request.paper_trading,
            "status": "starting"
        }
        
    except Exception as e:
        logger.error(f"Failed to start trading engine: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/engine/stop")
async def stop_trading_engine(
    background_tasks: BackgroundTasks,
    engine: TradingEngine = Depends(get_trading_engine)
):
    """Stop the trading engine"""
    try:
        if not engine.is_running:
            return {"message": "Trading engine is not running", "status": "stopped"}
        
        # Stop engine in background
        background_tasks.add_task(engine.stop_engine)
        
        return {"message": "Trading engine stop initiated", "status": "stopping"}
        
    except Exception as e:
        logger.error(f"Failed to stop trading engine: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/engine/status", response_model=EngineStatusResponse)
async def get_engine_status(engine: TradingEngine = Depends(get_trading_engine)):
    """Get current engine status and statistics"""
    try:
        status = engine.get_engine_status()
        return EngineStatusResponse(**status)
        
    except Exception as e:
        logger.error(f"Failed to get engine status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Portfolio Management Endpoints
@router.post("/portfolios")
async def create_portfolio(
    request: PortfolioCreateRequest,
    engine: TradingEngine = Depends(get_trading_engine)
):
    """Create a new trading portfolio"""
    try:
        portfolio = await engine.create_portfolio(
            request.user_id, request.name, request.initial_balance
        )
        
        return {
            "message": "Portfolio created successfully",
            "portfolio_id": portfolio.id,
            "name": portfolio.name,
            "initial_balance": portfolio.initial_balance
        }
        
    except Exception as e:
        logger.error(f"Failed to create portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolios/{portfolio_id}", response_model=PortfolioSummaryResponse)
async def get_portfolio_summary(
    portfolio_id: str,
    engine: TradingEngine = Depends(get_trading_engine)
):
    """Get detailed portfolio summary including positions and performance"""
    try:
        summary = engine.get_portfolio_summary(portfolio_id)
        
        if not summary:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        return PortfolioSummaryResponse(**summary)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolios")
async def list_portfolios(engine: TradingEngine = Depends(get_trading_engine)):
    """List all active portfolios"""
    try:
        portfolios = []
        for portfolio_id, portfolio in engine.active_portfolios.items():
            portfolios.append({
                "portfolio_id": portfolio.id,
                "name": portfolio.name,
                "user_id": portfolio.user_id,
                "current_balance": portfolio.current_balance,
                "total_pnl": portfolio.total_pnl,
                "total_trades": portfolio.total_trades,
                "created_at": portfolio.created_at.isoformat()
            })
        
        return {"portfolios": portfolios}
        
    except Exception as e:
        logger.error(f"Failed to list portfolios: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Trading Signals and Analysis Endpoints
@router.get("/signals", response_model=List[SignalResponse])
async def get_recent_signals(
    limit: int = 20,
    engine: TradingEngine = Depends(get_trading_engine)
):
    """Get recent trading signals"""
    try:
        signals = engine.get_recent_signals(limit)
        return [SignalResponse(**signal) for signal in signals]
        
    except Exception as e:
        logger.error(f"Failed to get signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/{symbol}")
async def get_technical_analysis(
    symbol: str,
    interval: str = "5m",
    engine: TradingEngine = Depends(get_trading_engine)
):
    """Get technical analysis for a specific symbol"""
    try:
        # Get trend analysis
        trend_analysis = engine.ta_engine.get_trend_analysis(symbol, interval)
        
        # Get technical indicators
        indicators = engine.ta_engine.calculate_indicators(symbol, interval)
        
        # Get support/resistance levels
        support_resistance = engine.ta_engine.get_support_resistance(symbol, interval)
        
        # Get volatility metrics
        volatility_metrics = engine.ta_engine.get_volatility_metrics(symbol, interval)
        
        return {
            "symbol": symbol,
            "interval": interval,
            "trend_analysis": trend_analysis,
            "indicators": indicators.dict() if indicators else None,
            "support_resistance": support_resistance,
            "volatility_metrics": volatility_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get technical analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market-data/{symbol}")
async def get_market_data(
    symbol: str,
    interval: str = "5m",
    limit: int = 100,
    engine: TradingEngine = Depends(get_trading_engine)
):
    """Get recent market data for a symbol"""
    try:
        # Get data from MongoDB
        klines = await engine.mongo_manager.get_klines(symbol, interval, limit)
        
        if not klines:
            # Try to get from storage
            storage_klines = engine.storage.load_kline_data("binance", symbol, interval, limit)
            klines = [kline.to_dict() for kline in storage_klines]
        
        return {
            "symbol": symbol,
            "interval": interval,
            "data_points": len(klines),
            "klines": klines[-limit:] if klines else []  # Return most recent
        }
        
    except Exception as e:
        logger.error(f"Failed to get market data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Strategy Management Endpoints
@router.get("/strategies")
async def get_strategy_status(engine: TradingEngine = Depends(get_trading_engine)):
    """Get status and performance of all trading strategies"""
    try:
        status = engine.strategy_manager.get_status()
        rankings = engine.strategy_manager.get_strategy_rankings()
        
        return {
            "strategy_status": status,
            "strategy_rankings": rankings,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get strategy status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Risk Management Endpoints
@router.get("/risk/{portfolio_id}")
async def get_risk_analysis(
    portfolio_id: str,
    engine: TradingEngine = Depends(get_trading_engine)
):
    """Get risk analysis for a specific portfolio"""
    try:
        if portfolio_id not in engine.active_portfolios:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        portfolio = engine.active_portfolios[portfolio_id]
        positions = engine.active_positions.get(portfolio_id, [])
        
        risk_summary = engine.risk_manager.get_risk_summary(portfolio, positions)
        
        return {
            "portfolio_id": portfolio_id,
            "risk_analysis": risk_summary,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get risk analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Data Management Endpoints
@router.post("/data/subscribe")
async def subscribe_symbols(
    request: SymbolSubscriptionRequest,
    engine: TradingEngine = Depends(get_trading_engine)
):
    """Subscribe to additional symbols for data collection"""
    try:
        if not engine.is_running:
            raise HTTPException(status_code=400, detail="Trading engine is not running")
        
        # Add symbols to subscribed set
        engine.subscribed_symbols.update(request.symbols)
        
        # Update symbol intervals
        for symbol in request.symbols:
            engine.symbol_intervals[symbol] = request.intervals
        
        return {
            "message": "Symbols subscription updated",
            "symbols": request.symbols,
            "intervals": request.intervals,
            "total_subscribed": len(engine.subscribed_symbols)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to subscribe to symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/storage-stats")
async def get_storage_statistics(engine: TradingEngine = Depends(get_trading_engine)):
    """Get data storage statistics"""
    try:
        stats = engine.storage.get_storage_stats()
        
        # Add additional information
        stats.update({
            "symbols_tracked": len(engine.subscribed_symbols),
            "active_data_feeds": len(engine.ta_engine.symbol_data),
            "data_summary": engine.ta_engine.get_data_summary()
        })
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get storage statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health Check Endpoint
@router.get("/health")
async def health_check(engine: TradingEngine = Depends(get_trading_engine)):
    """Health check endpoint for monitoring"""
    try:
        status = engine.get_engine_status()
        
        # Determine overall health
        health_status = "healthy"
        if not engine.is_running:
            health_status = "stopped"
        elif status["total_positions"] == 0 and status["signals_generated"] == 0:
            health_status = "starting"
        
        return {
            "status": health_status,
            "timestamp": datetime.utcnow().isoformat(),
            "engine_running": engine.is_running,
            "data_feeds_active": len(engine.subscribed_symbols) > 0,
            "portfolios_active": len(engine.active_portfolios) > 0
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }