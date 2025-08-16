"""
Enhanced Trading Engine with Full ML/RL Integration
Combining traditional strategies with advanced AI capabilities
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import concurrent.futures
import json
import os

# Import existing components
from .trading_engine import TradingEngine
from .data_models import TradingSignal, KlineData, Portfolio, Position
from .storage import MongoDBManager

# Import ML/RL components
from .ml_models.reinforcement_learning import (
    TradingRLAgent, MultiAgentRLSystem, AdvancedRLTrainer
)
from .ml_models.deep_learning import (
    LSTMPricePredictor, TransformerMarketAnalyzer, EnsembleMLPredictor
)
from .ml_models.sentiment_analysis import (
    MarketSentimentAnalyzer, NewsAnalyzer, SocialMediaAnalyzer
)
from .ml_models.optimization import (
    HyperparameterOptimizer, StrategyOptimizer, PortfolioOptimizer
)

logger = logging.getLogger(__name__)


class MLEnhancedTradingEngine(TradingEngine):
    """
    Enhanced Trading Engine with Full ML/RL Capabilities
    Extends the base trading engine with advanced AI features
    """
    
    def __init__(self, mongo_manager: MongoDBManager, storage_path: str = "./trading_data"):
        # Initialize base engine
        super().__init__(mongo_manager, storage_path)
        
        # Initialize ML/RL components
        self.ml_components = {}
        self.ml_enabled = True
        self.ml_predictions = {}
        self.sentiment_data = {}
        self.rl_agents = {}
        
        # Performance tracking
        self.ml_performance_history = deque(maxlen=1000)
        self.prediction_accuracy = {}
        
        logger.info("Initializing ML/RL Enhanced Trading Engine...")
        
        # Initialize ML models
        self._initialize_ml_models()
        
        # Initialize RL agents
        self._initialize_rl_agents()
        
        # Initialize sentiment analysis
        self._initialize_sentiment_analysis()
        
        # Initialize optimization systems
        self._initialize_optimization_systems()
        
        logger.info("ML Enhanced Trading Engine initialized successfully")
    
    def _initialize_ml_models(self):
        """Initialize deep learning models"""
        try:
            # LSTM Price Predictor
            self.ml_components['lstm_predictor'] = LSTMPricePredictor()
            
            # Transformer Market Analyzer
            self.ml_components['transformer_analyzer'] = TransformerMarketAnalyzer()
            
            # Ensemble ML Predictor
            self.ml_components['ensemble_predictor'] = EnsembleMLPredictor()
            
            logger.info("ML models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
            self.ml_enabled = False
    
    def _initialize_rl_agents(self):
        """Initialize reinforcement learning agents"""
        try:
            # Single RL Agent
            self.rl_agents['primary_agent'] = TradingRLAgent(
                data_manager=self.data_manager,
                ta_engine=self.ta_engine,
                risk_manager=self.risk_manager,
                algorithm='PPO',
                symbols=list(self.symbol_intervals.keys())
            )
            
            # Multi-Agent System
            self.rl_agents['multi_agent_system'] = MultiAgentRLSystem(
                data_manager=self.data_manager,
                ta_engine=self.ta_engine,
                risk_manager=self.risk_manager,
                symbols=list(self.symbol_intervals.keys())
            )
            
            # Advanced RL Trainer
            self.ml_components['rl_trainer'] = AdvancedRLTrainer(
                data_manager=self.data_manager,
                ta_engine=self.ta_engine,
                risk_manager=self.risk_manager,
                symbols=list(self.symbol_intervals.keys())
            )
            
            logger.info("RL agents initialized")
            
        except Exception as e:
            logger.error(f"Error initializing RL agents: {e}")
    
    def _initialize_sentiment_analysis(self):
        """Initialize sentiment analysis system"""
        try:
            # Market sentiment analyzer (works without API keys)
            self.ml_components['sentiment_analyzer'] = MarketSentimentAnalyzer()
            
            # News analyzer (optional with API key)
            self.ml_components['news_analyzer'] = NewsAnalyzer()
            
            # Social media analyzer (optional with Reddit API)
            self.ml_components['social_analyzer'] = SocialMediaAnalyzer()
            
            logger.info("Sentiment analysis systems initialized")
            
        except Exception as e:
            logger.error(f"Error initializing sentiment analysis: {e}")
    
    def _initialize_optimization_systems(self):
        """Initialize optimization systems"""
        try:
            # Hyperparameter optimizer
            self.ml_components['hyperopt'] = HyperparameterOptimizer()
            
            # Strategy optimizer
            self.ml_components['strategy_opt'] = StrategyOptimizer()
            
            # Portfolio optimizer
            self.ml_components['portfolio_opt'] = PortfolioOptimizer()
            
            logger.info("Optimization systems initialized")
            
        except Exception as e:
            logger.error(f"Error initializing optimization systems: {e}")
    
    async def start_enhanced_engine(self, symbols: Optional[List[str]] = None, enable_ml: bool = True):
        """Start the enhanced trading engine with ML/RL capabilities"""
        try:
            logger.info("Starting ML Enhanced Trading Engine...")
            
            # Start base engine
            await super().start_engine(symbols)
            
            if enable_ml and self.ml_enabled:
                # Start ML prediction tasks
                asyncio.create_task(self._ml_prediction_loop())
                
                # Start sentiment analysis loop
                asyncio.create_task(self._sentiment_analysis_loop())
                
                # Start RL agent training (background)
                asyncio.create_task(self._rl_training_loop())
                
                # Start ML performance monitoring
                asyncio.create_task(self._ml_performance_monitor())
                
                logger.info("ML/RL systems activated")
            
            logger.info("Enhanced Trading Engine started successfully")
            
        except Exception as e:
            logger.error(f"Error starting enhanced engine: {e}")
            raise
    
    async def _ml_prediction_loop(self):
        """Main ML prediction loop"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                for symbol in self.subscribed_symbols:
                    # Get recent market data
                    market_data = await self._get_market_data_for_ml(symbol)
                    
                    if market_data is not None and len(market_data) > 100:
                        # Generate ML predictions
                        predictions = await self._generate_ml_predictions(symbol, market_data)
                        
                        # Store predictions
                        self.ml_predictions[symbol] = predictions
                        
                        # Generate enhanced trading signals
                        enhanced_signals = await self._generate_enhanced_signals(symbol, predictions)
                        
                        # Process enhanced signals
                        for signal in enhanced_signals:
                            await self._process_enhanced_signal(signal)
            
            except Exception as e:
                logger.error(f"Error in ML prediction loop: {e}")
                await asyncio.sleep(30)  # Wait before retry
    
    async def _sentiment_analysis_loop(self):
        """Sentiment analysis loop"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Analyze market sentiment
                sentiment_analyzer = self.ml_components.get('sentiment_analyzer')
                if sentiment_analyzer:
                    symbols_to_analyze = [s.replace('USDT', '') for s in self.subscribed_symbols]
                    
                    sentiment_data = await sentiment_analyzer.analyze_market_sentiment(symbols_to_analyze)
                    self.sentiment_data = sentiment_data
                    
                    logger.info(f"Sentiment analysis completed: {sentiment_data.sentiment_trend}")
            
            except Exception as e:
                logger.error(f"Error in sentiment analysis loop: {e}")
                await asyncio.sleep(60)
    
    async def _rl_training_loop(self):
        """Background RL training loop"""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Check if we have enough data for training
                total_data_points = sum(
                    len(data["timestamps"]) for data in self.ta_engine.symbol_data.values()
                )
                
                if total_data_points > 1000:  # Minimum data threshold
                    # Train primary RL agent
                    primary_agent = self.rl_agents.get('primary_agent')
                    if primary_agent:
                        logger.info("Starting background RL training...")
                        primary_agent.train(total_timesteps=10000)  # Limited training
                        logger.info("Background RL training completed")
            
            except Exception as e:
                logger.error(f"Error in RL training loop: {e}")
    
    async def _ml_performance_monitor(self):
        """Monitor ML model performance"""
        while self.is_running:
            try:
                await asyncio.sleep(900)  # Run every 15 minutes
                
                # Evaluate prediction accuracy
                for symbol in self.subscribed_symbols:
                    if symbol in self.ml_predictions:
                        accuracy = await self._evaluate_prediction_accuracy(symbol)
                        self.prediction_accuracy[symbol] = accuracy
                
                # Store performance metrics
                performance_entry = {
                    'timestamp': datetime.utcnow(),
                    'prediction_accuracy': dict(self.prediction_accuracy),
                    'sentiment_confidence': getattr(self.sentiment_data, 'confidence_level', 0.0),
                    'ml_signals_generated': len(self.ml_predictions)
                }
                
                self.ml_performance_history.append(performance_entry)
            
            except Exception as e:
                logger.error(f"Error in ML performance monitoring: {e}")
    
    async def _get_market_data_for_ml(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get market data formatted for ML models"""
        try:
            # Get data from storage or TA engine
            if symbol in self.ta_engine.symbol_data:
                data = self.ta_engine.symbol_data[symbol]
                
                if len(data["timestamps"]) > 0:
                    df = pd.DataFrame({
                        'timestamp': data["timestamps"],
                        'open': data["opens"],
                        'high': data["highs"],
                        'low': data["lows"],
                        'close': data["closes"],
                        'volume': data["volumes"]
                    })
                    
                    return df
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting market data for ML: {e}")
            return None
    
    async def _generate_ml_predictions(self, symbol: str, market_data: pd.DataFrame) -> Dict:
        """Generate predictions from multiple ML models"""
        try:
            predictions = {}
            
            # LSTM Price Prediction
            lstm_predictor = self.ml_components.get('lstm_predictor')
            if lstm_predictor and lstm_predictor.is_trained:
                try:
                    lstm_pred, lstm_conf = lstm_predictor.predict(market_data, return_confidence=True)
                    predictions['lstm'] = {
                        'prediction': float(lstm_pred[0]) if isinstance(lstm_pred, np.ndarray) else float(lstm_pred),
                        'confidence': float(lstm_conf[0]) if isinstance(lstm_conf, np.ndarray) else float(lstm_conf)
                    }
                except Exception as e:
                    logger.debug(f"LSTM prediction failed for {symbol}: {e}")
            
            # Transformer Market Analysis
            transformer_analyzer = self.ml_components.get('transformer_analyzer')
            if transformer_analyzer and transformer_analyzer.is_trained:
                try:
                    market_state = transformer_analyzer.predict_market_state(market_data)
                    predictions['transformer'] = market_state
                except Exception as e:
                    logger.debug(f"Transformer analysis failed for {symbol}: {e}")
            
            # Ensemble Prediction
            ensemble_predictor = self.ml_components.get('ensemble_predictor')
            if ensemble_predictor and ensemble_predictor.is_trained:
                try:
                    ensemble_pred = ensemble_predictor.predict(market_data, return_individual=True)
                    predictions['ensemble'] = ensemble_pred
                except Exception as e:
                    logger.debug(f"Ensemble prediction failed for {symbol}: {e}")
            
            # RL Agent Decision
            primary_agent = self.rl_agents.get('primary_agent')
            if primary_agent:
                try:
                    # Get current market observation
                    action, _ = primary_agent.predict_action()
                    predictions['rl_action'] = {
                        'symbol_index': int(action[0]),
                        'action_type': int(action[1]),
                        'position_size': float(action[2])
                    }
                except Exception as e:
                    logger.debug(f"RL agent prediction failed: {e}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating ML predictions: {e}")
            return {}
    
    async def _generate_enhanced_signals(self, symbol: str, predictions: Dict) -> List[TradingSignal]:
        """Generate enhanced trading signals using ML predictions"""
        try:
            enhanced_signals = []
            current_price = await self._get_current_price(symbol)
            
            if not current_price:
                return enhanced_signals
            
            # Combine traditional and ML signals
            traditional_signals = self.strategy_manager.get_signals(symbol, '5m', current_price)
            
            for trad_signal in traditional_signals:
                # Enhance with ML predictions
                enhanced_signal = await self._enhance_signal_with_ml(trad_signal, predictions, symbol)
                if enhanced_signal:
                    enhanced_signals.append(enhanced_signal)
            
            # Generate pure ML signals
            ml_only_signals = await self._generate_ml_only_signals(symbol, predictions, current_price)
            enhanced_signals.extend(ml_only_signals)
            
            return enhanced_signals
            
        except Exception as e:
            logger.error(f"Error generating enhanced signals: {e}")
            return []
    
    async def _enhance_signal_with_ml(self, signal: TradingSignal, predictions: Dict, symbol: str) -> Optional[TradingSignal]:
        """Enhance traditional signal with ML predictions"""
        try:
            enhanced_confidence = signal.confidence
            enhanced_reasoning = [signal.reasoning]
            
            # Enhance with LSTM prediction
            if 'lstm' in predictions:
                lstm_pred = predictions['lstm']
                current_price = signal.price_target or 0
                
                if current_price > 0:
                    predicted_return = (lstm_pred['prediction'] - current_price) / current_price
                    
                    # Check if ML prediction agrees with signal direction
                    if signal.action.value == 'BUY' and predicted_return > 0.01:  # 1% threshold
                        enhanced_confidence += 0.1 * lstm_pred['confidence']
                        enhanced_reasoning.append(f"LSTM supports bullish outlook (+{predicted_return*100:.1f}%)")
                    elif signal.action.value == 'SELL' and predicted_return < -0.01:
                        enhanced_confidence += 0.1 * lstm_pred['confidence']
                        enhanced_reasoning.append(f"LSTM supports bearish outlook ({predicted_return*100:.1f}%)")
                    else:
                        enhanced_confidence -= 0.05  # ML disagrees, reduce confidence
                        enhanced_reasoning.append("LSTM prediction conflicts with signal")
            
            # Enhance with sentiment data
            if hasattr(self.sentiment_data, 'overall_sentiment'):
                sentiment = self.sentiment_data.overall_sentiment
                
                if signal.action.value == 'BUY' and sentiment > 0.1:
                    enhanced_confidence += 0.05
                    enhanced_reasoning.append(f"Positive market sentiment ({sentiment:.2f})")
                elif signal.action.value == 'SELL' and sentiment < -0.1:
                    enhanced_confidence += 0.05
                    enhanced_reasoning.append(f"Negative market sentiment ({sentiment:.2f})")
            
            # Enhance with transformer market state
            if 'transformer' in predictions:
                market_state = predictions['transformer']
                state = market_state.get('predicted_state', 'Unknown')
                
                if (signal.action.value == 'BUY' and state == 'Bullish') or \
                   (signal.action.value == 'SELL' and state == 'Bearish'):
                    enhanced_confidence += 0.08 * market_state.get('confidence', 0)
                    enhanced_reasoning.append(f"Market regime: {state}")
            
            # Cap confidence at 0.98
            enhanced_confidence = min(enhanced_confidence, 0.98)
            
            # Only return signal if confidence is still reasonable
            if enhanced_confidence > 0.4:
                # Create enhanced signal
                enhanced_signal = TradingSignal(
                    symbol=signal.symbol,
                    action=signal.action,
                    confidence=enhanced_confidence,
                    price_target=signal.price_target,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    reasoning=" | ".join(enhanced_reasoning),
                    strategy_used=f"{signal.strategy_used} + ML Enhanced"
                )
                
                return enhanced_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error enhancing signal with ML: {e}")
            return signal  # Return original signal if enhancement fails
    
    async def _generate_ml_only_signals(self, symbol: str, predictions: Dict, current_price: float) -> List[TradingSignal]:
        """Generate signals based purely on ML predictions"""
        try:
            ml_signals = []
            
            # RL Agent signal
            if 'rl_action' in predictions:
                rl_action = predictions['rl_action']
                action_type = rl_action['action_type']
                position_size = rl_action['position_size']
                
                if action_type == 1 and position_size > 0.1:  # Buy signal
                    from .data_models import OrderSide
                    
                    rl_signal = TradingSignal(
                        symbol=symbol,
                        action=OrderSide.BUY,
                        confidence=min(position_size, 0.8),
                        price_target=current_price,
                        stop_loss=current_price * 0.98,
                        take_profit=current_price * 1.04,
                        reasoning="RL Agent autonomous decision",
                        strategy_used="Deep RL Agent"
                    )
                    ml_signals.append(rl_signal)
                
                elif action_type == 2 and position_size > 0.1:  # Sell signal
                    rl_signal = TradingSignal(
                        symbol=symbol,
                        action=OrderSide.SELL,
                        confidence=min(position_size, 0.8),
                        price_target=current_price,
                        stop_loss=current_price * 1.02,
                        take_profit=current_price * 0.96,
                        reasoning="RL Agent autonomous decision",
                        strategy_used="Deep RL Agent"
                    )
                    ml_signals.append(rl_signal)
            
            # Ensemble ML signal
            if 'ensemble' in predictions:
                ensemble_pred = predictions['ensemble']
                
                if isinstance(ensemble_pred, dict) and 'ensemble_prediction' in ensemble_pred:
                    predicted_price = ensemble_pred['ensemble_prediction']
                    
                    if predicted_price > current_price * 1.01:  # 1% upside
                        ensemble_signal = TradingSignal(
                            symbol=symbol,
                            action=OrderSide.BUY,
                            confidence=0.7,
                            price_target=current_price,
                            stop_loss=current_price * 0.98,
                            take_profit=predicted_price * 0.98,
                            reasoning=f"Ensemble ML predicts {((predicted_price/current_price-1)*100):.1f}% upside",
                            strategy_used="Ensemble ML Model"
                        )
                        ml_signals.append(ensemble_signal)
                    
                    elif predicted_price < current_price * 0.99:  # 1% downside
                        ensemble_signal = TradingSignal(
                            symbol=symbol,
                            action=OrderSide.SELL,
                            confidence=0.7,
                            price_target=current_price,
                            stop_loss=current_price * 1.02,
                            take_profit=predicted_price * 1.02,
                            reasoning=f"Ensemble ML predicts {((predicted_price/current_price-1)*100):.1f}% downside",
                            strategy_used="Ensemble ML Model"
                        )
                        ml_signals.append(ensemble_signal)
            
            return ml_signals
            
        except Exception as e:
            logger.error(f"Error generating ML-only signals: {e}")
            return []
    
    async def _process_enhanced_signal(self, signal: TradingSignal):
        """Process enhanced trading signal"""
        try:
            # Store enhanced signal
            self.signal_history.append(signal)
            await self.mongo_manager.store_trading_signal(signal)
            
            # Execute if confidence is high enough
            if signal.confidence >= 0.75:  # Higher threshold for ML signals
                await self._execute_signal(signal)
            
            # Notify callbacks
            for callback in self.signal_callbacks:
                try:
                    callback(signal)
                except Exception as e:
                    logger.error(f"Error in enhanced signal callback: {e}")
        
        except Exception as e:
            logger.error(f"Error processing enhanced signal: {e}")
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            if symbol in self.ta_engine.symbol_data:
                data = self.ta_engine.symbol_data[symbol]
                if len(data["closes"]) > 0:
                    return float(data["closes"][-1])
            return None
        except:
            return None
    
    async def _evaluate_prediction_accuracy(self, symbol: str) -> Dict:
        """Evaluate ML prediction accuracy"""
        try:
            if symbol not in self.ml_predictions:
                return {'accuracy': 0.0, 'samples': 0}
            
            # This is a simplified accuracy calculation
            # In production, you'd track predictions over time and compare with actual outcomes
            
            predictions = self.ml_predictions[symbol]
            accuracy_metrics = {
                'lstm_available': 'lstm' in predictions,
                'transformer_available': 'transformer' in predictions,
                'ensemble_available': 'ensemble' in predictions,
                'rl_available': 'rl_action' in predictions,
                'prediction_count': len(predictions),
                'last_updated': datetime.utcnow().isoformat()
            }
            
            return accuracy_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating prediction accuracy: {e}")
            return {'accuracy': 0.0, 'samples': 0, 'error': str(e)}
    
    # Enhanced API methods for ML/RL features
    
    async def train_ml_models(self, model_type: str = 'all', symbols: Optional[List[str]] = None) -> Dict:
        """Train ML models with current market data"""
        try:
            logger.info(f"Starting ML model training for {model_type}")
            
            symbols = symbols or list(self.subscribed_symbols)
            training_results = {}
            
            # Prepare training data
            training_data = {}
            for symbol in symbols:
                market_data = await self._get_market_data_for_ml(symbol)
                if market_data is not None and len(market_data) > 200:
                    training_data[symbol] = market_data
            
            if not training_data:
                return {'error': 'Insufficient training data available'}
            
            # Train LSTM models
            if model_type in ['all', 'lstm']:
                for symbol, data in training_data.items():
                    try:
                        lstm_predictor = self.ml_components['lstm_predictor']
                        history = lstm_predictor.train(data, target_column='close')
                        training_results[f'lstm_{symbol}'] = history
                        logger.info(f"LSTM training completed for {symbol}")
                    except Exception as e:
                        logger.error(f"LSTM training failed for {symbol}: {e}")
                        training_results[f'lstm_{symbol}'] = {'error': str(e)}
            
            # Train Transformer models
            if model_type in ['all', 'transformer']:
                combined_data = pd.concat(training_data.values(), ignore_index=True)
                try:
                    transformer_analyzer = self.ml_components['transformer_analyzer']
                    history = transformer_analyzer.train(combined_data, target_type='trend')
                    training_results['transformer'] = history
                    logger.info("Transformer training completed")
                except Exception as e:
                    logger.error(f"Transformer training failed: {e}")
                    training_results['transformer'] = {'error': str(e)}
            
            # Train Ensemble models
            if model_type in ['all', 'ensemble']:
                combined_data = pd.concat(training_data.values(), ignore_index=True)
                try:
                    ensemble_predictor = self.ml_components['ensemble_predictor']
                    history = ensemble_predictor.train(combined_data, target_column='close')
                    training_results['ensemble'] = history
                    logger.info("Ensemble training completed")
                except Exception as e:
                    logger.error(f"Ensemble training failed: {e}")
                    training_results['ensemble'] = {'error': str(e)}
            
            # Train RL agents
            if model_type in ['all', 'rl']:
                try:
                    multi_agent_system = self.rl_agents['multi_agent_system']
                    multi_agent_system.train_all_agents(total_timesteps=20000)
                    training_results['rl_multi_agent'] = {'status': 'completed'}
                    logger.info("RL agents training completed")
                except Exception as e:
                    logger.error(f"RL training failed: {e}")
                    training_results['rl_multi_agent'] = {'error': str(e)}
            
            return {
                'training_completed': True,
                'results': training_results,
                'trained_models': list(training_results.keys()),
                'training_time': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"ML model training failed: {e}")
            return {'error': str(e)}
    
    async def optimize_hyperparameters(self, optimization_type: str = 'all') -> Dict:
        """Run hyperparameter optimization"""
        try:
            logger.info(f"Starting hyperparameter optimization for {optimization_type}")
            
            optimizer = self.ml_components['hyperopt']
            optimization_results = {}
            
            # Get training data
            combined_data = []
            for symbol in self.subscribed_symbols:
                market_data = await self._get_market_data_for_ml(symbol)
                if market_data is not None:
                    combined_data.append(market_data)
            
            if not combined_data:
                return {'error': 'No training data available for optimization'}
            
            training_data = pd.concat(combined_data, ignore_index=True)
            
            # Optimize LSTM
            if optimization_type in ['all', 'lstm']:
                try:
                    lstm_result = optimizer.optimize_lstm_parameters(training_data)
                    optimization_results['lstm'] = lstm_result
                    logger.info("LSTM hyperparameter optimization completed")
                except Exception as e:
                    logger.error(f"LSTM optimization failed: {e}")
                    optimization_results['lstm'] = {'error': str(e)}
            
            # Optimize Ensemble
            if optimization_type in ['all', 'ensemble']:
                try:
                    ensemble_result = optimizer.optimize_ensemble_parameters(training_data)
                    optimization_results['ensemble'] = ensemble_result
                    logger.info("Ensemble hyperparameter optimization completed")
                except Exception as e:
                    logger.error(f"Ensemble optimization failed: {e}")
                    optimization_results['ensemble'] = {'error': str(e)}
            
            # Optimize RL
            if optimization_type in ['all', 'rl']:
                try:
                    rl_result = optimizer.optimize_rl_parameters()
                    optimization_results['rl'] = rl_result
                    logger.info("RL hyperparameter optimization completed")
                except Exception as e:
                    logger.error(f"RL optimization failed: {e}")
                    optimization_results['rl'] = {'error': str(e)}
            
            return {
                'optimization_completed': True,
                'results': optimization_results,
                'optimized_components': list(optimization_results.keys())
            }
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {e}")
            return {'error': str(e)}
    
    def get_ml_engine_status(self) -> Dict:
        """Get comprehensive ML engine status"""
        try:
            base_status = super().get_engine_status()
            
            # Add ML-specific status
            ml_status = {
                'ml_enabled': self.ml_enabled,
                'ml_components': {
                    name: {
                        'available': component is not None,
                        'type': type(component).__name__,
                        'trained': getattr(component, 'is_trained', False)
                    }
                    for name, component in self.ml_components.items()
                },
                'rl_agents': {
                    name: {
                        'available': agent is not None,
                        'type': type(agent).__name__,
                        'status': agent.get_agent_status() if hasattr(agent, 'get_agent_status') else 'unknown'
                    }
                    for name, agent in self.rl_agents.items()
                },
                'ml_predictions': {
                    'symbols_with_predictions': list(self.ml_predictions.keys()),
                    'prediction_count': len(self.ml_predictions),
                    'last_prediction_time': max(
                        [datetime.utcnow() - timedelta(minutes=5)] +  # Default fallback
                        [entry['timestamp'] for entry in self.ml_performance_history[-1:]]
                    ).isoformat() if self.ml_performance_history else None
                },
                'sentiment_analysis': {
                    'available': hasattr(self.sentiment_data, 'overall_sentiment'),
                    'current_sentiment': getattr(self.sentiment_data, 'overall_sentiment', 0.0),
                    'sentiment_trend': getattr(self.sentiment_data, 'sentiment_trend', 'neutral'),
                    'confidence': getattr(self.sentiment_data, 'confidence_level', 0.0)
                },
                'ml_performance': {
                    'history_size': len(self.ml_performance_history),
                    'prediction_accuracy': dict(self.prediction_accuracy),
                    'recent_performance': self.ml_performance_history[-1] if self.ml_performance_history else None
                }
            }
            
            # Combine base and ML status
            enhanced_status = {**base_status, 'ml_systems': ml_status}
            
            return enhanced_status
            
        except Exception as e:
            logger.error(f"Error getting ML engine status: {e}")
            return {'error': str(e)}
    
    def get_ml_predictions(self, symbol: Optional[str] = None) -> Dict:
        """Get current ML predictions"""
        try:
            if symbol:
                return self.ml_predictions.get(symbol, {})
            else:
                return dict(self.ml_predictions)
        except Exception as e:
            return {'error': str(e)}
    
    def get_sentiment_analysis(self) -> Dict:
        """Get current sentiment analysis"""
        try:
            if hasattr(self.sentiment_data, '__dict__'):
                return self.sentiment_data.__dict__
            else:
                return {'sentiment_data': str(self.sentiment_data)}
        except Exception as e:
            return {'error': str(e)}
    
    async def run_ml_backtest(self, symbols: List[str], days: int = 30) -> Dict:
        """Run ML-enhanced backtest"""
        try:
            logger.info(f"Running ML backtest for {len(symbols)} symbols over {days} days")
            
            # This is a simplified backtest implementation
            # In production, you'd implement a comprehensive backtesting framework
            
            backtest_results = {
                'symbols': symbols,
                'days': days,
                'start_time': datetime.utcnow().isoformat(),
                'ml_strategies_tested': list(self.ml_components.keys()),
                'status': 'completed',
                'note': 'Simplified backtest - full implementation would include detailed performance metrics'
            }
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"ML backtest failed: {e}")
            return {'error': str(e)}