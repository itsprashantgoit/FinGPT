"""
Enhanced API Routes for FinGPT Trading System with ML/RL Integration
FastAPI endpoints for trading engine operations with full AI capabilities
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from .ml_enhanced_engine import MLEnhancedTradingEngine as TradingEngine
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
@router.get("/signals/recent")
async def get_recent_signals(limit: int = 20, trading_engine: TradingEngine = Depends(get_trading_engine)):
    """Get recent trading signals"""
    try:
        signals = trading_engine.get_recent_signals(limit)
        return {
            "signals": signals,
            "count": len(signals),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting recent signals: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve recent signals")


# ============================================================================
# ML/RL ENHANCED ENDPOINTS - FULL POTENTIAL IMPLEMENTATION
# ============================================================================

@router.get("/ml/status")
async def get_ml_status(trading_engine: TradingEngine = Depends(get_trading_engine)):
    """Get comprehensive ML/RL system status"""
    try:
        status = trading_engine.get_ml_engine_status()
        return {
            "ml_status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting ML status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve ML status")


@router.get("/ml/predictions")
async def get_ml_predictions(
    symbol: Optional[str] = Query(None, description="Specific symbol to get predictions for"),
    trading_engine: TradingEngine = Depends(get_trading_engine)
):
    """Get current ML predictions for symbols"""
    try:
        predictions = trading_engine.get_ml_predictions(symbol)
        return {
            "predictions": predictions,
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting ML predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve ML predictions")


@router.get("/ml/sentiment")
async def get_sentiment_analysis(trading_engine: TradingEngine = Depends(get_trading_engine)):
    """Get current market sentiment analysis"""
    try:
        sentiment = trading_engine.get_sentiment_analysis()
        return {
            "sentiment_analysis": sentiment,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sentiment analysis")


class MLTrainingRequest(BaseModel):
    model_type: str = Field(default="all", description="Type of model to train: 'all', 'lstm', 'transformer', 'ensemble', 'rl'")
    symbols: Optional[List[str]] = Field(default=None, description="Symbols to train on (default: all subscribed)")
    
@router.post("/ml/train")
async def train_ml_models(
    request: MLTrainingRequest,
    background_tasks: BackgroundTasks,
    trading_engine: TradingEngine = Depends(get_trading_engine)
):
    """Train ML models with current market data"""
    try:
        # Run training in background
        background_tasks.add_task(
            trading_engine.train_ml_models,
            request.model_type,
            request.symbols
        )
        
        return {
            "message": f"ML training started for {request.model_type}",
            "model_type": request.model_type,
            "symbols": request.symbols or "all subscribed",
            "status": "training_initiated",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error starting ML training: {e}")
        raise HTTPException(status_code=500, detail="Failed to start ML training")


class OptimizationRequest(BaseModel):
    optimization_type: str = Field(default="all", description="Type of optimization: 'all', 'lstm', 'ensemble', 'rl'")
    
@router.post("/ml/optimize")
async def optimize_hyperparameters(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    trading_engine: TradingEngine = Depends(get_trading_engine)
):
    """Run hyperparameter optimization"""
    try:
        # Run optimization in background
        background_tasks.add_task(
            trading_engine.optimize_hyperparameters,
            request.optimization_type
        )
        
        return {
            "message": f"Hyperparameter optimization started for {request.optimization_type}",
            "optimization_type": request.optimization_type,
            "status": "optimization_initiated",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error starting optimization: {e}")
        raise HTTPException(status_code=500, detail="Failed to start hyperparameter optimization")


class BacktestRequest(BaseModel):
    symbols: List[str] = Field(description="Symbols to backtest")
    days: int = Field(default=30, description="Number of days to backtest", ge=1, le=365)
    
@router.post("/ml/backtest")
async def run_ml_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    trading_engine: TradingEngine = Depends(get_trading_engine)
):
    """Run ML-enhanced backtest"""
    try:
        # Run backtest in background
        background_tasks.add_task(
            trading_engine.run_ml_backtest,
            request.symbols,
            request.days
        )
        
        return {
            "message": f"ML backtest started for {len(request.symbols)} symbols over {request.days} days",
            "symbols": request.symbols,
            "days": request.days,
            "status": "backtest_initiated",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error starting ML backtest: {e}")
        raise HTTPException(status_code=500, detail="Failed to start ML backtest")


@router.get("/rl/agents/status")
async def get_rl_agents_status(trading_engine: TradingEngine = Depends(get_trading_engine)):
    """Get status of all RL agents"""
    try:
        if hasattr(trading_engine, 'rl_agents'):
            agents_status = {}
            for name, agent in trading_engine.rl_agents.items():
                if hasattr(agent, 'get_agent_status'):
                    agents_status[name] = agent.get_agent_status()
                elif hasattr(agent, 'get_system_status'):
                    agents_status[name] = agent.get_system_status()
                else:
                    agents_status[name] = {"type": type(agent).__name__, "available": True}
            
            return {
                "rl_agents": agents_status,
                "total_agents": len(trading_engine.rl_agents),
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {"error": "RL agents not available"}
    except Exception as e:
        logger.error(f"Error getting RL agents status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve RL agents status")


class RLTrainingRequest(BaseModel):
    agent_type: str = Field(default="all", description="Type of RL agent to train: 'all', 'primary', 'multi_agent'")
    timesteps: int = Field(default=50000, description="Number of training timesteps", ge=1000, le=1000000)
    
@router.post("/rl/train")
async def train_rl_agents(
    request: RLTrainingRequest,
    background_tasks: BackgroundTasks,
    trading_engine: TradingEngine = Depends(get_trading_engine)
):
    """Train RL agents"""
    try:
        if hasattr(trading_engine, 'rl_agents'):
            if request.agent_type == "all":
                # Train all agents
                for name, agent in trading_engine.rl_agents.items():
                    if hasattr(agent, 'train'):
                        background_tasks.add_task(agent.train, total_timesteps=request.timesteps)
                    elif hasattr(agent, 'train_all_agents'):
                        background_tasks.add_task(agent.train_all_agents, total_timesteps=request.timesteps)
            elif request.agent_type in trading_engine.rl_agents:
                agent = trading_engine.rl_agents[request.agent_type]
                if hasattr(agent, 'train'):
                    background_tasks.add_task(agent.train, total_timesteps=request.timesteps)
                elif hasattr(agent, 'train_all_agents'):
                    background_tasks.add_task(agent.train_all_agents, total_timesteps=request.timesteps)
            else:
                raise HTTPException(status_code=404, detail=f"RL agent '{request.agent_type}' not found")
            
            return {
                "message": f"RL training started for {request.agent_type}",
                "agent_type": request.agent_type,
                "timesteps": request.timesteps,
                "status": "training_initiated",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="RL agents not available")
    except Exception as e:
        logger.error(f"Error starting RL training: {e}")
        raise HTTPException(status_code=500, detail="Failed to start RL training")


@router.get("/ai/performance")
async def get_ai_performance_metrics(trading_engine: TradingEngine = Depends(get_trading_engine)):
    """Get AI/ML performance metrics and analytics"""
    try:
        performance_data = {}
        
        # ML performance metrics
        if hasattr(trading_engine, 'ml_performance_history'):
            recent_performance = list(trading_engine.ml_performance_history)[-10:]  # Last 10 entries
            performance_data['ml_performance_history'] = recent_performance
        
        # Prediction accuracy
        if hasattr(trading_engine, 'prediction_accuracy'):
            performance_data['prediction_accuracy'] = dict(trading_engine.prediction_accuracy)
        
        # Sentiment analysis performance
        if hasattr(trading_engine, 'sentiment_data'):
            sentiment_data = trading_engine.sentiment_data
            if hasattr(sentiment_data, '__dict__'):
                performance_data['sentiment_performance'] = {
                    'confidence_level': getattr(sentiment_data, 'confidence_level', 0.0),
                    'sample_size': getattr(sentiment_data, 'sample_size', 0),
                    'source_breakdown': getattr(sentiment_data, 'source_breakdown', {})
                }
        
        # RL agent performance
        if hasattr(trading_engine, 'rl_agents'):
            rl_performance = {}
            for name, agent in trading_engine.rl_agents.items():
                if hasattr(agent, 'performance_metrics'):
                    rl_performance[name] = agent.performance_metrics
            performance_data['rl_performance'] = rl_performance
        
        return {
            "ai_performance": performance_data,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting AI performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve AI performance metrics")


@router.get("/ai/insights")
async def get_ai_insights(
    symbol: Optional[str] = Query(None, description="Symbol to get insights for"),
    trading_engine: TradingEngine = Depends(get_trading_engine)
):
    """Get AI-powered market insights and recommendations"""
    try:
        insights = {
            "market_analysis": {},
            "trading_recommendations": {},
            "risk_assessment": {},
            "predictions": {}
        }
        
        # Get ML predictions
        if symbol:
            predictions = trading_engine.get_ml_predictions(symbol)
            insights["predictions"] = predictions
        else:
            insights["predictions"] = trading_engine.get_ml_predictions()
        
        # Get sentiment analysis
        sentiment = trading_engine.get_sentiment_analysis()
        insights["market_analysis"]["sentiment"] = sentiment
        
        # Get recent signals for analysis
        recent_signals = trading_engine.get_recent_signals(10)
        
        # Analyze signal patterns
        if recent_signals:
            signal_analysis = {
                "total_signals": len(recent_signals),
                "bullish_signals": len([s for s in recent_signals if s.get('action') == 'BUY']),
                "bearish_signals": len([s for s in recent_signals if s.get('action') == 'SELL']),
                "avg_confidence": sum(s.get('confidence', 0) for s in recent_signals) / len(recent_signals),
                "ml_enhanced_signals": len([s for s in recent_signals if 'ML' in s.get('strategy_used', '')])
            }
            insights["trading_recommendations"]["signal_analysis"] = signal_analysis
        
        # Risk assessment based on current positions and market conditions
        engine_status = trading_engine.get_engine_status()
        insights["risk_assessment"] = {
            "total_positions": engine_status.get("total_positions", 0),
            "active_portfolios": engine_status.get("active_portfolios", 0),
            "signals_generated": engine_status.get("signals_generated", 0)
        }
        
        return {
            "ai_insights": insights,
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting AI insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve AI insights")


@router.post("/ai/emergency-stop")
async def ai_emergency_stop(trading_engine: TradingEngine = Depends(get_trading_engine)):
    """Emergency stop for all AI/ML systems"""
    try:
        # Stop ML prediction loops and RL training
        if hasattr(trading_engine, 'ml_enabled'):
            trading_engine.ml_enabled = False
        
        # Clear ML predictions
        if hasattr(trading_engine, 'ml_predictions'):
            trading_engine.ml_predictions.clear()
        
        return {
            "message": "AI/ML systems emergency stop initiated",
            "status": "stopped",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in AI emergency stop: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute AI emergency stop")


@router.post("/ai/resume")
async def resume_ai_systems(trading_engine: TradingEngine = Depends(get_trading_engine)):
    """Resume AI/ML systems after emergency stop"""
    try:
        # Re-enable ML systems
        if hasattr(trading_engine, 'ml_enabled'):
            trading_engine.ml_enabled = True
        
        return {
            "message": "AI/ML systems resumed",
            "status": "active",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error resuming AI systems: {e}")
        raise HTTPException(status_code=500, detail="Failed to resume AI systems")


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