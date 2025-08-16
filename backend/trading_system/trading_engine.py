"""
Main Trading Engine for FinGPT Trading System
Coordinates all components: data feeds, analysis, strategies, risk management
Optimized for High-Performance Cloud Infrastructure (16-core ARM, 62GB RAM)
"""
import asyncio
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
import concurrent.futures
import threading

from .data_sources import MultiSourceDataManager
from .technical_analysis import TechnicalAnalysisEngine, IndicatorConfig
from .strategies import StrategyManager, StrategyConfig
from .risk_manager import RiskManager, RiskLimits
from .storage import MemoryEfficientDataStorage, MongoDBManager
from .data_models import (
    KlineData, TradingSignal, Portfolio, Position, Order, 
    PerformanceMetrics, OrderSide, OrderStatus, ExchangeType
)

# Import optimized configuration
try:
    from config.performance_config import PerformanceConfig, get_environment_config
except ImportError:
    # Fallback if config not available
    class PerformanceConfig:
        @classmethod
        def get_worker_counts(cls):
            return {"data_workers": 8, "strategy_workers": 4, "risk_workers": 2, "db_connections": 8}

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Main trading engine orchestrating all system components
    Optimized for High-Performance Cloud Infrastructure (16-core ARM Neoverse-N1, 62GB RAM)
    """
    
    def __init__(self, mongo_manager: MongoDBManager, storage_path: str = "./trading_data"):
        # Initialize core components with optimized configuration
        self.config = PerformanceConfig.get_worker_counts()
        self.mongo_manager = mongo_manager
        self.storage = MemoryEfficientDataStorage(storage_path)
        self.data_manager = MultiSourceDataManager()
        
        # Analysis and strategy components with parallel processing
        self.ta_engine = TechnicalAnalysisEngine(IndicatorConfig())
        self.strategy_manager = StrategyManager(self.ta_engine)
        self.risk_manager = RiskManager(RiskLimits())
        
        # Thread pools for parallel processing
        self.data_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.get("data_workers", 8),
            thread_name_prefix="DataWorker"
        )
        self.strategy_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.get("strategy_workers", 4),
            thread_name_prefix="StrategyWorker"
        )
        self.risk_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.get("risk_workers", 2),
            thread_name_prefix="RiskWorker"
        )
        
        # Engine state
        self.is_running = False
        self.is_paper_trading = True  # Start with paper trading
        self.active_portfolios: Dict[str, Portfolio] = {}
        self.active_positions: Dict[str, List[Position]] = {}
        self.pending_orders: Dict[str, List[Order]] = {}
        
        # Performance tracking with enhanced capacity
        self.performance_metrics: Dict[str, PerformanceMetrics] = {}
        self.signal_history: deque = deque(maxlen=10000)  # Increased capacity
        
        # Subscribed symbols and intervals
        self.subscribed_symbols: Set[str] = set()
        self.symbol_intervals = {
            'BTCUSDT': ['1m', '5m', '1h'],
            'ETHUSDT': ['1m', '5m', '1h'],
            'BNBUSDT': ['5m', '1h'],
        }
        
        # Callbacks
        self.signal_callbacks: List = []
        self.position_callbacks: List = []
        
        logger.info("Trading Engine initialized successfully")
    
    async def start_engine(self, symbols: Optional[List[str]] = None):
        """Start the trading engine with real-time data feeds"""
        if self.is_running:
            logger.warning("Trading engine is already running")
            return
        
        try:
            logger.info("Starting FinGPT Trading Engine...")
            
            # Use default symbols if none provided
            if not symbols:
                symbols = list(self.symbol_intervals.keys())
            
            # Set up data callbacks
            self.data_manager.add_data_callback(self._on_market_data)
            
            # Start real-time data feeds
            intervals = ['1m', '5m']  # Start with these intervals
            self.data_manager.start_real_time_feeds(symbols, intervals)
            
            # Load historical data for analysis
            logger.info("Loading historical data for analysis...")
            historical_data = self.data_manager.fetch_historical_data(symbols, ['1h', '1d'])
            
            if historical_data:
                # Process historical data to warm up indicators
                for kline in historical_data:
                    self.ta_engine.update_data(kline)
                
                # Store historical data
                self.storage.store_kline_data(historical_data)
                logger.info(f"Processed {len(historical_data)} historical data points")
            
            # Update subscribed symbols
            self.subscribed_symbols.update(symbols)
            
            # Start background tasks
            asyncio.create_task(self._performance_monitor())
            asyncio.create_task(self._risk_monitor())
            asyncio.create_task(self._cleanup_task())
            
            self.is_running = True
            logger.info(f"Trading Engine started successfully with {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to start trading engine: {e}")
            raise
    
    async def stop_engine(self):
        """Stop the trading engine and cleanup resources"""
        logger.info("Stopping Trading Engine...")
        
        self.is_running = False
        
        # Stop data feeds
        self.data_manager.stop()
        
        # Save current state
        await self._save_engine_state()
        
        logger.info("Trading Engine stopped successfully")
    
    def _on_market_data(self, kline: KlineData):
        """Handle incoming real-time market data"""
        try:
            # Update technical analysis engine
            self.ta_engine.update_data(kline)
            
            # Store in MongoDB for real-time access
            asyncio.create_task(self.mongo_manager.store_kline(kline))
            
            # Generate trading signals
            asyncio.create_task(self._process_trading_signals(kline))
            
            # Update positions with current price
            self._update_positions_with_current_price(kline.symbol, kline.close_price)
            
        except Exception as e:
            logger.error(f"Error processing market data for {kline.symbol}: {e}")
    
    async def _process_trading_signals(self, kline: KlineData):
        """Process trading signals for the given market data"""
        try:
            symbol = kline.symbol
            interval = kline.interval
            current_price = kline.close_price
            
            # Only process signals for main intervals
            if interval not in ['1m', '5m']:
                return
            
            # Get signals from strategy manager
            signals = self.strategy_manager.get_signals(symbol, interval, current_price)
            
            for signal in signals:
                # Store signal in history
                self.signal_history.append(signal)
                
                # Store in MongoDB
                await self.mongo_manager.store_trading_signal(signal)
                
                # Execute signal if conditions are met
                if signal.confidence >= 0.7:  # High confidence threshold
                    await self._execute_signal(signal)
                
                # Notify callbacks
                for callback in self.signal_callbacks:
                    try:
                        callback(signal)
                    except Exception as e:
                        logger.error(f"Error in signal callback: {e}")
            
        except Exception as e:
            logger.error(f"Error processing trading signals: {e}")
    
    async def _execute_signal(self, signal: TradingSignal):
        """Execute a trading signal (paper trading or real)"""
        try:
            # For now, implement paper trading
            if not self.is_paper_trading:
                logger.info("Real trading not implemented yet - using paper trading")
            
            # Get default portfolio or create one
            portfolio_id = "default_portfolio"
            if portfolio_id not in self.active_portfolios:
                await self.create_portfolio("default_user", "Default Portfolio", 10000.0)
            
            portfolio = self.active_portfolios[portfolio_id]
            positions = self.active_positions.get(portfolio_id, [])
            
            # Calculate position size using risk management
            volatility_metrics = self.ta_engine.get_volatility_metrics(signal.symbol, "5m")
            volatility = volatility_metrics.get("volatility", 0.02)
            
            position_size_pct = self.risk_manager.calculate_optimal_position_size(
                portfolio, signal.price_target or 0, signal.stop_loss or 0,
                signal.confidence, volatility
            )
            
            # Validate position with risk manager
            can_trade, reason = self.risk_manager.validate_new_position(
                portfolio, positions, position_size_pct, signal.symbol
            )
            
            if not can_trade:
                logger.info(f"Trade rejected for {signal.symbol}: {reason}")
                return
            
            # Create and execute order (paper trading)
            order = Order(
                portfolio_id=portfolio_id,
                symbol=signal.symbol,
                side=signal.action,
                type="market",  # Market order for simplicity
                size=position_size_pct,
                price=signal.price_target,
                strategy_id=signal.strategy_used,
                signal_id=signal.id
            )
            
            # Simulate order execution (in paper trading)
            await self._execute_paper_order(order, signal)
            
        except Exception as e:
            logger.error(f"Error executing signal for {signal.symbol}: {e}")
    
    async def _execute_paper_order(self, order: Order, signal: TradingSignal):
        """Execute order in paper trading mode"""
        try:
            portfolio = self.active_portfolios[order.portfolio_id]
            
            # Calculate position details
            position_value = order.size * portfolio.current_balance
            shares = position_value / order.price if order.price and order.price > 0 else 0
            
            # Check if we have enough balance
            if position_value > portfolio.available_balance:
                logger.warning(f"Insufficient balance for {order.symbol}: needed {position_value}, available {portfolio.available_balance}")
                order.status = OrderStatus.REJECTED
                return
            
            # Create position
            position = Position(
                portfolio_id=order.portfolio_id,
                symbol=order.symbol,
                side=order.side,
                size=shares,
                entry_price=order.price or 0,
                current_price=order.price or 0,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            # Update portfolio
            portfolio.available_balance -= position_value
            portfolio.total_trades += 1
            
            # Add position to active positions
            if order.portfolio_id not in self.active_positions:
                self.active_positions[order.portfolio_id] = []
            self.active_positions[order.portfolio_id].append(position)
            
            # Update order status
            order.status = OrderStatus.FILLED
            order.filled_size = shares
            order.filled_price = order.price
            order.filled_at = datetime.utcnow()
            
            logger.info(f"Paper trade executed: {order.side} {shares:.4f} {order.symbol} at ${order.price:.4f}")
            
            # Notify callbacks
            for callback in self.position_callbacks:
                try:
                    callback(position)
                except Exception as e:
                    logger.error(f"Error in position callback: {e}")
            
        except Exception as e:
            logger.error(f"Error executing paper order: {e}")
    
    def _update_positions_with_current_price(self, symbol: str, current_price: float):
        """Update all positions for a symbol with current market price"""
        try:
            for portfolio_id, positions in self.active_positions.items():
                portfolio = self.active_portfolios.get(portfolio_id)
                if not portfolio:
                    continue
                
                positions_to_close = []
                
                for i, position in enumerate(positions):
                    if position.symbol == symbol:
                        # Update current price and unrealized P&L
                        old_price = position.current_price
                        position.current_price = current_price
                        
                        if position.side == OrderSide.BUY:
                            position.unrealized_pnl = (current_price - position.entry_price) * position.size
                        else:
                            position.unrealized_pnl = (position.entry_price - current_price) * position.size
                        
                        # Check if position should be closed
                        should_close, reason = self.risk_manager.should_close_position(
                            position, current_price, portfolio
                        )
                        
                        if should_close:
                            logger.info(f"Closing position {position.symbol}: {reason}")
                            positions_to_close.append((i, position, reason))
                
                # Close positions that need to be closed
                for i, position, reason in reversed(positions_to_close):
                    asyncio.create_task(self._close_position(portfolio_id, i, position, reason))
            
        except Exception as e:
            logger.error(f"Error updating positions for {symbol}: {e}")
    
    async def _close_position(self, portfolio_id: str, position_index: int, position: Position, reason: str):
        """Close a position and update portfolio"""
        try:
            portfolio = self.active_portfolios[portfolio_id]
            
            # Calculate realized P&L
            realized_pnl = position.unrealized_pnl
            
            # Update portfolio balance
            position_value = position.size * position.current_price
            portfolio.available_balance += position_value
            portfolio.total_pnl += realized_pnl
            
            if realized_pnl > 0:
                portfolio.winning_trades += 1
            
            # Update risk manager
            self.risk_manager.update_daily_pnl(realized_pnl)
            
            # Remove position from active positions
            self.active_positions[portfolio_id].pop(position_index)
            
            # Update strategy performance
            strategy_name = None
            for signal in reversed(self.signal_history):
                if signal.symbol == position.symbol:
                    strategy_name = signal.strategy_used
                    break
            
            if strategy_name:
                self.strategy_manager.update_strategy_performance(
                    strategy_name, realized_pnl > 0
                )
            
            logger.info(f"Closed position {position.symbol}: P&L ${realized_pnl:.2f} ({reason})")
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    async def create_portfolio(self, user_id: str, name: str, initial_balance: float) -> Portfolio:
        """Create a new trading portfolio"""
        portfolio = Portfolio(
            user_id=user_id,
            name=name,
            initial_balance=initial_balance,
            current_balance=initial_balance,
            available_balance=initial_balance
        )
        
        self.active_portfolios[portfolio.id] = portfolio
        self.active_positions[portfolio.id] = []
        self.pending_orders[portfolio.id] = []
        
        logger.info(f"Created portfolio '{name}' with ${initial_balance:,.2f}")
        return portfolio
    
    async def _performance_monitor(self):
        """Monitor and update performance metrics"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                for portfolio_id, portfolio in self.active_portfolios.items():
                    # Calculate current metrics
                    positions = self.active_positions.get(portfolio_id, [])
                    
                    total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
                    total_return = (portfolio.current_balance + total_unrealized_pnl - portfolio.initial_balance) / portfolio.initial_balance
                    
                    win_rate = (portfolio.winning_trades / portfolio.total_trades) if portfolio.total_trades > 0 else 0
                    
                    # Create performance metrics
                    metrics = PerformanceMetrics(
                        portfolio_id=portfolio_id,
                        date=datetime.utcnow(),
                        total_return=total_return,
                        daily_return=0,  # Would need historical data to calculate
                        cumulative_return=total_return,
                        volatility=0,  # Would need return series to calculate
                        sharpe_ratio=0,  # Would need risk-free rate and return series
                        max_drawdown=portfolio.max_drawdown,
                        win_rate=win_rate,
                        profit_factor=1.0,  # Simplified
                        total_trades=portfolio.total_trades
                    )
                    
                    self.performance_metrics[portfolio_id] = metrics
                
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
    
    async def _risk_monitor(self):
        """Monitor portfolio risk and trigger alerts if needed"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                for portfolio_id, portfolio in self.active_portfolios.items():
                    positions = self.active_positions.get(portfolio_id, [])
                    
                    # Check emergency shutdown conditions
                    if self.risk_manager.emergency_risk_shutdown(portfolio):
                        logger.critical(f"EMERGENCY SHUTDOWN triggered for portfolio {portfolio_id}")
                        # Close all positions
                        for i, position in enumerate(positions):
                            await self._close_position(portfolio_id, i, position, "Emergency shutdown")
                        
                        # Clear remaining positions
                        self.active_positions[portfolio_id] = []
                    
                    # Calculate and log risk metrics
                    risk_summary = self.risk_manager.get_risk_summary(portfolio, positions)
                    if risk_summary["risk_status"] != "LOW":
                        logger.warning(f"Portfolio {portfolio_id} risk status: {risk_summary['risk_status']}")
                
            except Exception as e:
                logger.error(f"Error in risk monitor: {e}")
    
    async def _cleanup_task(self):
        """Periodic cleanup of old data and resources"""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up old technical analysis data
                self.ta_engine.clear_old_data(24)  # Keep 24 hours
                
                # Clean up old storage data
                self.storage.cleanup_by_age(30)  # Keep 30 days
                
                # Reset daily counters at midnight
                current_hour = datetime.utcnow().hour
                if current_hour == 0:  # Midnight
                    self.strategy_manager.reset_daily_counters()
                
                logger.info("Completed periodic cleanup")
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    async def _save_engine_state(self):
        """Save current engine state to storage"""
        try:
            # In a production system, this would save to persistent storage
            logger.info("Engine state saved (placeholder implementation)")
            
        except Exception as e:
            logger.error(f"Error saving engine state: {e}")
    
    def add_signal_callback(self, callback):
        """Add callback for trading signals"""
        self.signal_callbacks.append(callback)
    
    def add_position_callback(self, callback):
        """Add callback for position updates"""
        self.position_callbacks.append(callback)
    
    def get_engine_status(self) -> Dict:
        """Get comprehensive engine status"""
        return {
            "is_running": self.is_running,
            "is_paper_trading": self.is_paper_trading,
            "subscribed_symbols": list(self.subscribed_symbols),
            "active_portfolios": len(self.active_portfolios),
            "total_positions": sum(len(positions) for positions in self.active_positions.values()),
            "signals_generated": len(self.signal_history),
            "data_points_processed": sum(
                len(data["timestamps"]) 
                for data in self.ta_engine.symbol_data.values()
            ),
            "strategy_status": self.strategy_manager.get_status(),
            "storage_stats": self.storage.get_storage_stats()
        }
    
    def get_portfolio_summary(self, portfolio_id: str) -> Optional[Dict]:
        """Get detailed portfolio summary"""
        if portfolio_id not in self.active_portfolios:
            return None
        
        portfolio = self.active_portfolios[portfolio_id]
        positions = self.active_positions.get(portfolio_id, [])
        
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
        total_position_value = sum(pos.size * pos.current_price for pos in positions)
        
        return {
            "portfolio": {
                "id": portfolio.id,
                "name": portfolio.name,
                "initial_balance": portfolio.initial_balance,
                "current_balance": portfolio.current_balance,
                "available_balance": portfolio.available_balance,
                "total_pnl": portfolio.total_pnl + total_unrealized_pnl,
                "unrealized_pnl": total_unrealized_pnl,
                "total_trades": portfolio.total_trades,
                "winning_trades": portfolio.winning_trades,
                "win_rate": (portfolio.winning_trades / portfolio.total_trades) if portfolio.total_trades > 0 else 0
            },
            "positions": [
                {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "size": pos.size,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "stop_loss": pos.stop_loss,
                    "take_profit": pos.take_profit
                }
                for pos in positions
            ],
            "risk_summary": self.risk_manager.get_risk_summary(portfolio, positions),
            "performance_metrics": self.performance_metrics.get(portfolio_id)
        }
    
    def get_recent_signals(self, limit: int = 20) -> List[Dict]:
        """Get recent trading signals"""
        recent_signals = list(self.signal_history)[-limit:]
        return [
            {
                "symbol": signal.symbol,
                "action": signal.action,
                "confidence": signal.confidence,
                "price_target": signal.price_target,
                "reasoning": signal.reasoning,
                "strategy_used": signal.strategy_used,
                "created_at": signal.created_at.isoformat()
            }
            for signal in recent_signals
        ]