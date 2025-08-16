"""
Trading Strategies for FinGPT Trading System
Modular strategy framework with risk management integration
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

from .data_models import TradingSignal, OrderSide, KlineData
from .technical_analysis import TechnicalAnalysisEngine

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    WEAK = 0.3
    MODERATE = 0.6
    STRONG = 0.8
    VERY_STRONG = 0.95


@dataclass
class StrategyConfig:
    """Base configuration for trading strategies"""
    max_position_size: float = 0.05  # 5% max position
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    min_confidence: float = 0.6  # Minimum confidence for trade
    risk_reward_ratio: float = 2.0  # Risk:Reward ratio
    max_trades_per_day: int = 10
    cooldown_minutes: int = 30


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str, config: StrategyConfig, ta_engine: TechnicalAnalysisEngine):
        self.name = name
        self.config = config
        self.ta_engine = ta_engine
        self.last_signals = {}  # Track last signal per symbol
        self.trade_count_today = 0
        self.last_trade_time = None
    
    @abstractmethod
    def analyze(self, symbol: str, interval: str, current_price: float) -> Optional[TradingSignal]:
        """Analyze market data and generate trading signal"""
        pass
    
    def can_trade(self, symbol: str) -> bool:
        """Check if strategy can generate new signal for symbol"""
        current_time = datetime.utcnow()
        
        # Check daily trade limit
        if self.trade_count_today >= self.config.max_trades_per_day:
            return False
        
        # Check cooldown period
        if self.last_trade_time:
            time_diff = current_time - self.last_trade_time
            if time_diff < timedelta(minutes=self.config.cooldown_minutes):
                return False
        
        return True
    
    def calculate_position_size(self, confidence: float, volatility: float = 0.02) -> float:
        """Calculate position size based on confidence and volatility"""
        # Kelly Criterion inspired position sizing
        base_size = self.config.max_position_size
        
        # Adjust for confidence
        confidence_multiplier = min(confidence / 0.8, 1.0)
        
        # Adjust for volatility (reduce size in high volatility)
        volatility_adjustment = max(0.3, 1.0 - (volatility * 10))
        
        position_size = base_size * confidence_multiplier * volatility_adjustment
        
        return min(position_size, self.config.max_position_size)
    
    def calculate_stop_loss_take_profit(self, entry_price: float, side: OrderSide, atr: float = None) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        
        if atr and atr > 0:
            # Use ATR-based stops for better market adaptation
            stop_distance = atr * 2  # 2x ATR for stop loss
            profit_distance = atr * self.config.risk_reward_ratio * 2
        else:
            # Use percentage-based stops
            stop_distance = entry_price * self.config.stop_loss_pct
            profit_distance = entry_price * self.config.take_profit_pct
        
        if side == OrderSide.BUY:
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + profit_distance
        else:  # SELL
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - profit_distance
        
        return stop_loss, take_profit


class MomentumStrategy(BaseStrategy):
    """Momentum-based trading strategy using trend-following indicators"""
    
    def __init__(self, config: StrategyConfig, ta_engine: TechnicalAnalysisEngine):
        super().__init__("Momentum Strategy", config, ta_engine)
    
    def analyze(self, symbol: str, interval: str, current_price: float) -> Optional[TradingSignal]:
        """Analyze for momentum-based signals"""
        
        if not self.can_trade(symbol):
            return None
        
        # Get trend analysis
        trend_analysis = self.ta_engine.get_trend_analysis(symbol, interval)
        indicators = self.ta_engine.calculate_indicators(symbol, interval)
        volatility_metrics = self.ta_engine.get_volatility_metrics(symbol, interval)
        
        if not trend_analysis or not indicators:
            return None
        
        trend = trend_analysis["trend"]
        strength = trend_analysis["strength"]
        confidence = trend_analysis["confidence"]
        
        # Momentum conditions
        signal = None
        reasoning = []
        
        # Strong bullish momentum
        if (trend == "bullish" and 
            strength > 0.7 and 
            confidence > self.config.min_confidence and
            indicators.rsi and indicators.rsi < 70):  # Not overbought
            
            reasoning.append(f"Strong bullish trend (strength: {strength:.2f})")
            if indicators.macd and indicators.macd_signal and indicators.macd > indicators.macd_signal:
                reasoning.append("MACD bullish crossover")
                confidence += 0.1
            
            # Calculate position size and stops
            position_size = self.calculate_position_size(confidence, volatility_metrics["volatility"])
            stop_loss, take_profit = self.calculate_stop_loss_take_profit(
                current_price, OrderSide.BUY, volatility_metrics.get("atr")
            )
            
            signal = TradingSignal(
                symbol=symbol,
                action=OrderSide.BUY,
                confidence=min(confidence, 0.95),
                price_target=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=" | ".join(reasoning),
                strategy_used=self.name
            )
        
        # Strong bearish momentum
        elif (trend == "bearish" and 
              strength > 0.7 and 
              confidence > self.config.min_confidence and
              indicators.rsi and indicators.rsi > 30):  # Not oversold
            
            reasoning.append(f"Strong bearish trend (strength: {strength:.2f})")
            if indicators.macd and indicators.macd_signal and indicators.macd < indicators.macd_signal:
                reasoning.append("MACD bearish crossover")
                confidence += 0.1
            
            position_size = self.calculate_position_size(confidence, volatility_metrics["volatility"])
            stop_loss, take_profit = self.calculate_stop_loss_take_profit(
                current_price, OrderSide.SELL, volatility_metrics.get("atr")
            )
            
            signal = TradingSignal(
                symbol=symbol,
                action=OrderSide.SELL,
                confidence=min(confidence, 0.95),
                price_target=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=" | ".join(reasoning),
                strategy_used=self.name
            )
        
        # Update tracking
        if signal:
            self.last_signals[symbol] = signal
            self.trade_count_today += 1
            self.last_trade_time = datetime.utcnow()
        
        return signal


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy for oversold/overbought conditions"""
    
    def __init__(self, config: StrategyConfig, ta_engine: TechnicalAnalysisEngine):
        super().__init__("Mean Reversion Strategy", config, ta_engine)
        # Adjust config for mean reversion
        self.config.stop_loss_pct = 0.015  # Tighter stops
        self.config.take_profit_pct = 0.025  # Smaller profits
    
    def analyze(self, symbol: str, interval: str, current_price: float) -> Optional[TradingSignal]:
        """Analyze for mean reversion opportunities"""
        
        if not self.can_trade(symbol):
            return None
        
        indicators = self.ta_engine.calculate_indicators(symbol, interval)
        volatility_metrics = self.ta_engine.get_volatility_metrics(symbol, interval)
        
        if not indicators:
            return None
        
        signal = None
        reasoning = []
        confidence = 0.6
        
        # Oversold condition (potential buy)
        if (indicators.rsi and indicators.rsi < 30 and
            indicators.bollinger_lower and current_price <= indicators.bollinger_lower * 1.005):
            
            reasoning.append(f"Oversold conditions (RSI: {indicators.rsi:.1f})")
            reasoning.append("Price at/below Bollinger Band lower")
            
            # Additional confirmation
            if indicators.rsi < 25:
                reasoning.append("Extreme oversold (RSI < 25)")
                confidence += 0.15
            
            if volatility_metrics["volatility"] < 0.03:  # Low volatility
                reasoning.append("Low volatility environment")
                confidence += 0.1
            
            if confidence >= self.config.min_confidence:
                stop_loss, take_profit = self.calculate_stop_loss_take_profit(
                    current_price, OrderSide.BUY, volatility_metrics.get("atr")
                )
                
                signal = TradingSignal(
                    symbol=symbol,
                    action=OrderSide.BUY,
                    confidence=min(confidence, 0.9),
                    price_target=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reasoning=" | ".join(reasoning),
                    strategy_used=self.name
                )
        
        # Overbought condition (potential sell)
        elif (indicators.rsi and indicators.rsi > 70 and
              indicators.bollinger_upper and current_price >= indicators.bollinger_upper * 0.995):
            
            reasoning.append(f"Overbought conditions (RSI: {indicators.rsi:.1f})")
            reasoning.append("Price at/above Bollinger Band upper")
            
            if indicators.rsi > 75:
                reasoning.append("Extreme overbought (RSI > 75)")
                confidence += 0.15
            
            if volatility_metrics["volatility"] < 0.03:
                reasoning.append("Low volatility environment")
                confidence += 0.1
            
            if confidence >= self.config.min_confidence:
                stop_loss, take_profit = self.calculate_stop_loss_take_profit(
                    current_price, OrderSide.SELL, volatility_metrics.get("atr")
                )
                
                signal = TradingSignal(
                    symbol=symbol,
                    action=OrderSide.SELL,
                    confidence=min(confidence, 0.9),
                    price_target=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reasoning=" | ".join(reasoning),
                    strategy_used=self.name
                )
        
        if signal:
            self.last_signals[symbol] = signal
            self.trade_count_today += 1
            self.last_trade_time = datetime.utcnow()
        
        return signal


class BreakoutStrategy(BaseStrategy):
    """Breakout strategy for support/resistance levels"""
    
    def __init__(self, config: StrategyConfig, ta_engine: TechnicalAnalysisEngine):
        super().__init__("Breakout Strategy", config, ta_engine)
    
    def analyze(self, symbol: str, interval: str, current_price: float) -> Optional[TradingSignal]:
        """Analyze for breakout opportunities"""
        
        if not self.can_trade(symbol):
            return None
        
        indicators = self.ta_engine.calculate_indicators(symbol, interval)
        support_resistance = self.ta_engine.get_support_resistance(symbol, interval)
        volatility_metrics = self.ta_engine.get_volatility_metrics(symbol, interval)
        
        if not indicators or not support_resistance:
            return None
        
        signal = None
        reasoning = []
        confidence = 0.6
        
        resistance_levels = support_resistance.get("resistance", [])
        support_levels = support_resistance.get("support", [])
        
        # Bullish breakout above resistance
        if resistance_levels:
            nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
            if current_price > nearest_resistance * 1.002:  # 0.2% above resistance
                reasoning.append(f"Breakout above resistance at {nearest_resistance:.4f}")
                
                # Volume confirmation (if available)
                if (volatility_metrics.get("atr", 0) > 0 and 
                    volatility_metrics["volatility"] > 0.02):
                    reasoning.append("High volume/volatility confirmation")
                    confidence += 0.15
                
                # RSI momentum confirmation
                if indicators.rsi and 50 < indicators.rsi < 70:
                    reasoning.append("RSI shows good momentum")
                    confidence += 0.1
                
                if confidence >= self.config.min_confidence:
                    stop_loss, take_profit = self.calculate_stop_loss_take_profit(
                        current_price, OrderSide.BUY, volatility_metrics.get("atr")
                    )
                    
                    signal = TradingSignal(
                        symbol=symbol,
                        action=OrderSide.BUY,
                        confidence=min(confidence, 0.9),
                        price_target=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        reasoning=" | ".join(reasoning),
                        strategy_used=self.name
                    )
        
        # Bearish breakdown below support
        elif support_levels:
            nearest_support = max(support_levels, key=lambda x: -abs(x - current_price))
            if current_price < nearest_support * 0.998:  # 0.2% below support
                reasoning.append(f"Breakdown below support at {nearest_support:.4f}")
                
                if (volatility_metrics.get("atr", 0) > 0 and 
                    volatility_metrics["volatility"] > 0.02):
                    reasoning.append("High volume/volatility confirmation")
                    confidence += 0.15
                
                if indicators.rsi and 30 < indicators.rsi < 50:
                    reasoning.append("RSI shows downward momentum")
                    confidence += 0.1
                
                if confidence >= self.config.min_confidence:
                    stop_loss, take_profit = self.calculate_stop_loss_take_profit(
                        current_price, OrderSide.SELL, volatility_metrics.get("atr")
                    )
                    
                    signal = TradingSignal(
                        symbol=symbol,
                        action=OrderSide.SELL,
                        confidence=min(confidence, 0.9),
                        price_target=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        reasoning=" | ".join(reasoning),
                        strategy_used=self.name
                    )
        
        if signal:
            self.last_signals[symbol] = signal
            self.trade_count_today += 1
            self.last_trade_time = datetime.utcnow()
        
        return signal


class StrategyManager:
    """Manager for coordinating multiple trading strategies"""
    
    def __init__(self, ta_engine: TechnicalAnalysisEngine):
        self.ta_engine = ta_engine
        self.strategies = {}
        self.strategy_performance = {}
        
        # Initialize default strategies
        config = StrategyConfig()
        self.add_strategy("momentum", MomentumStrategy(config, ta_engine))
        self.add_strategy("mean_reversion", MeanReversionStrategy(config, ta_engine))
        self.add_strategy("breakout", BreakoutStrategy(config, ta_engine))
    
    def add_strategy(self, name: str, strategy: BaseStrategy):
        """Add a new strategy to the manager"""
        self.strategies[name] = strategy
        self.strategy_performance[name] = {
            "total_signals": 0,
            "successful_signals": 0,
            "win_rate": 0.0,
            "avg_confidence": 0.0
        }
    
    def get_signals(self, symbol: str, interval: str, current_price: float) -> List[TradingSignal]:
        """Get signals from all active strategies"""
        signals = []
        
        for name, strategy in self.strategies.items():
            try:
                signal = strategy.analyze(symbol, interval, current_price)
                if signal:
                    signals.append(signal)
                    
                    # Update performance tracking
                    perf = self.strategy_performance[name]
                    perf["total_signals"] += 1
                    perf["avg_confidence"] = (
                        (perf["avg_confidence"] * (perf["total_signals"] - 1) + signal.confidence) /
                        perf["total_signals"]
                    )
                    
            except Exception as e:
                logger.error(f"Error in strategy {name} for {symbol}: {e}")
        
        return signals
    
    def get_best_signal(self, symbol: str, interval: str, current_price: float) -> Optional[TradingSignal]:
        """Get the highest confidence signal from all strategies"""
        signals = self.get_signals(symbol, interval, current_price)
        
        if not signals:
            return None
        
        # Return signal with highest confidence
        return max(signals, key=lambda s: s.confidence)
    
    def update_strategy_performance(self, strategy_name: str, was_successful: bool):
        """Update performance metrics for a strategy"""
        if strategy_name in self.strategy_performance:
            perf = self.strategy_performance[strategy_name]
            if was_successful:
                perf["successful_signals"] += 1
            
            # Recalculate win rate
            if perf["total_signals"] > 0:
                perf["win_rate"] = perf["successful_signals"] / perf["total_signals"]
    
    def get_strategy_rankings(self) -> List[Dict]:
        """Get strategies ranked by performance"""
        rankings = []
        
        for name, perf in self.strategy_performance.items():
            rankings.append({
                "strategy": name,
                "win_rate": perf["win_rate"],
                "total_signals": perf["total_signals"],
                "avg_confidence": perf["avg_confidence"],
                "score": perf["win_rate"] * 0.6 + perf["avg_confidence"] * 0.4
            })
        
        return sorted(rankings, key=lambda x: x["score"], reverse=True)
    
    def reset_daily_counters(self):
        """Reset daily trade counters for all strategies"""
        for strategy in self.strategies.values():
            strategy.trade_count_today = 0
        
        logger.info("Reset daily trade counters for all strategies")
    
    def get_status(self) -> Dict:
        """Get current status of all strategies"""
        status = {}
        
        for name, strategy in self.strategies.items():
            status[name] = {
                "trades_today": strategy.trade_count_today,
                "max_trades": strategy.config.max_trades_per_day,
                "can_trade": strategy.can_trade("test"),
                "last_trade_time": strategy.last_trade_time.isoformat() if strategy.last_trade_time else None,
                "performance": self.strategy_performance[name]
            }
        
        return status