"""
Technical Analysis Engine for FinGPT Trading System
Real-time calculation of technical indicators optimized for memory efficiency
"""
import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime

from .data_models import KlineData, TechnicalIndicators

logger = logging.getLogger(__name__)


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators"""
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    sma_short: int = 20
    sma_long: int = 50
    ema_short: int = 12
    ema_long: int = 26
    volume_period: int = 20


class TechnicalAnalysisEngine:
    """
    Memory-efficient technical analysis engine
    Optimized for real-time processing with limited resources
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        self.config = config or IndicatorConfig()
        self.symbol_data = {}  # Store recent data per symbol
        self.max_history = 200  # Keep only last 200 candles per symbol
        
    def update_data(self, kline: KlineData):
        """Update internal data buffer with new kline"""
        symbol_key = f"{kline.symbol}_{kline.interval}"
        
        # Initialize if new symbol
        if symbol_key not in self.symbol_data:
            self.symbol_data[symbol_key] = {
                'timestamps': [],
                'open': [],
                'high': [],
                'low': [],
                'close': [],
                'volume': []
            }
        
        data = self.symbol_data[symbol_key]
        
        # Check if this timestamp already exists (update case)
        if kline.timestamp in data['timestamps']:
            idx = data['timestamps'].index(kline.timestamp)
            data['open'][idx] = kline.open_price
            data['high'][idx] = kline.high_price
            data['low'][idx] = kline.low_price
            data['close'][idx] = kline.close_price
            data['volume'][idx] = kline.volume
        else:
            # Append new data
            data['timestamps'].append(kline.timestamp)
            data['open'].append(kline.open_price)
            data['high'].append(kline.high_price)
            data['low'].append(kline.low_price)
            data['close'].append(kline.close_price)
            data['volume'].append(kline.volume)
            
            # Maintain memory limit
            if len(data['timestamps']) > self.max_history:
                for key in data:
                    data[key] = data[key][-self.max_history:]
    
    def calculate_indicators(self, symbol: str, interval: str) -> Optional[TechnicalIndicators]:
        """Calculate all technical indicators for a symbol"""
        symbol_key = f"{symbol}_{interval}"
        
        if symbol_key not in self.symbol_data:
            logger.warning(f"No data available for {symbol_key}")
            return None
        
        data = self.symbol_data[symbol_key]
        
        # Need at least 50 periods for most indicators
        if len(data['close']) < 50:
            logger.debug(f"Insufficient data for {symbol_key}: {len(data['close'])} periods")
            return None
        
        try:
            # Convert to numpy arrays for TA-Lib
            close = np.array(data['close'], dtype=np.float64)
            high = np.array(data['high'], dtype=np.float64)
            low = np.array(data['low'], dtype=np.float64)
            volume = np.array(data['volume'], dtype=np.float64)
            
            # Calculate indicators
            indicators = TechnicalIndicators(
                symbol=symbol,
                timestamp=data['timestamps'][-1]
            )
            
            # RSI
            if len(close) >= self.config.rsi_period:
                rsi = talib.RSI(close, timeperiod=self.config.rsi_period)
                indicators.rsi = float(rsi[-1]) if not np.isnan(rsi[-1]) else None
            
            # MACD
            if len(close) >= self.config.macd_slow:
                macd, macd_signal, macd_hist = talib.MACD(
                    close,
                    fastperiod=self.config.macd_fast,
                    slowperiod=self.config.macd_slow,
                    signalperiod=self.config.macd_signal
                )
                indicators.macd = float(macd[-1]) if not np.isnan(macd[-1]) else None
                indicators.macd_signal = float(macd_signal[-1]) if not np.isnan(macd_signal[-1]) else None
            
            # Bollinger Bands
            if len(close) >= self.config.bollinger_period:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(
                    close,
                    timeperiod=self.config.bollinger_period,
                    nbdevup=self.config.bollinger_std,
                    nbdevdn=self.config.bollinger_std
                )
                indicators.bollinger_upper = float(bb_upper[-1]) if not np.isnan(bb_upper[-1]) else None
                indicators.bollinger_middle = float(bb_middle[-1]) if not np.isnan(bb_middle[-1]) else None
                indicators.bollinger_lower = float(bb_lower[-1]) if not np.isnan(bb_lower[-1]) else None
            
            # Moving Averages
            if len(close) >= self.config.sma_short:
                sma_20 = talib.SMA(close, timeperiod=self.config.sma_short)
                indicators.sma_20 = float(sma_20[-1]) if not np.isnan(sma_20[-1]) else None
            
            if len(close) >= self.config.sma_long:
                sma_50 = talib.SMA(close, timeperiod=self.config.sma_long)
                indicators.sma_50 = float(sma_50[-1]) if not np.isnan(sma_50[-1]) else None
            
            if len(close) >= self.config.ema_short:
                ema_12 = talib.EMA(close, timeperiod=self.config.ema_short)
                indicators.ema_12 = float(ema_12[-1]) if not np.isnan(ema_12[-1]) else None
            
            if len(close) >= self.config.ema_long:
                ema_26 = talib.EMA(close, timeperiod=self.config.ema_long)
                indicators.ema_26 = float(ema_26[-1]) if not np.isnan(ema_26[-1]) else None
            
            # Volume SMA
            if len(volume) >= self.config.volume_period:
                volume_sma = talib.SMA(volume, timeperiod=self.config.volume_period)
                indicators.volume_sma = float(volume_sma[-1]) if not np.isnan(volume_sma[-1]) else None
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol_key}: {e}")
            return None
    
    def get_trend_analysis(self, symbol: str, interval: str) -> Dict:
        """Analyze current trend using multiple indicators"""
        indicators = self.calculate_indicators(symbol, interval)
        
        if not indicators:
            return {"trend": "unknown", "strength": 0, "confidence": 0}
        
        trend_signals = []
        
        # Price vs Moving Averages
        symbol_key = f"{symbol}_{interval}"
        if symbol_key in self.symbol_data:
            current_price = self.symbol_data[symbol_key]['close'][-1]
            
            # SMA trend
            if indicators.sma_20 and indicators.sma_50:
                if current_price > indicators.sma_20 > indicators.sma_50:
                    trend_signals.append(("bullish", 0.8))
                elif current_price < indicators.sma_20 < indicators.sma_50:
                    trend_signals.append(("bearish", 0.8))
                else:
                    trend_signals.append(("sideways", 0.5))
            
            # Bollinger Bands
            if indicators.bollinger_upper and indicators.bollinger_lower:
                bb_position = (current_price - indicators.bollinger_lower) / (indicators.bollinger_upper - indicators.bollinger_lower)
                if bb_position > 0.8:
                    trend_signals.append(("overbought", 0.7))
                elif bb_position < 0.2:
                    trend_signals.append(("oversold", 0.7))
        
        # RSI analysis
        if indicators.rsi:
            if indicators.rsi > 70:
                trend_signals.append(("overbought", 0.6))
            elif indicators.rsi < 30:
                trend_signals.append(("oversold", 0.6))
            elif 40 < indicators.rsi < 60:
                trend_signals.append(("sideways", 0.4))
        
        # MACD analysis
        if indicators.macd and indicators.macd_signal:
            if indicators.macd > indicators.macd_signal:
                trend_signals.append(("bullish", 0.6))
            else:
                trend_signals.append(("bearish", 0.6))
        
        # Aggregate signals
        if not trend_signals:
            return {"trend": "unknown", "strength": 0, "confidence": 0}
        
        # Count signal types
        signal_counts = {}
        total_confidence = 0
        
        for signal, confidence in trend_signals:
            if signal not in signal_counts:
                signal_counts[signal] = {"count": 0, "confidence": 0}
            signal_counts[signal]["count"] += 1
            signal_counts[signal]["confidence"] += confidence
            total_confidence += confidence
        
        # Determine dominant trend
        dominant_signal = max(signal_counts.items(), key=lambda x: x[1]["confidence"])
        trend = dominant_signal[0]
        strength = dominant_signal[1]["confidence"] / len(trend_signals)
        overall_confidence = total_confidence / len(trend_signals)
        
        return {
            "trend": trend,
            "strength": strength,
            "confidence": overall_confidence,
            "signals": trend_signals,
            "indicators": {
                "rsi": indicators.rsi,
                "macd": indicators.macd,
                "macd_signal": indicators.macd_signal,
                "current_price": self.symbol_data[symbol_key]['close'][-1] if symbol_key in self.symbol_data else None
            }
        }
    
    def get_support_resistance(self, symbol: str, interval: str, lookback: int = 50) -> Dict:
        """Calculate support and resistance levels"""
        symbol_key = f"{symbol}_{interval}"
        
        if symbol_key not in self.symbol_data:
            return {"support": [], "resistance": []}
        
        data = self.symbol_data[symbol_key]
        
        if len(data['close']) < lookback:
            lookback = len(data['close'])
        
        highs = data['high'][-lookback:]
        lows = data['low'][-lookback:]
        closes = data['close'][-lookback:]
        
        # Find local peaks and troughs
        resistance_levels = []
        support_levels = []
        
        try:
            # Simple method: use recent highs/lows as support/resistance
            recent_high = max(highs)
            recent_low = min(lows)
            
            # Calculate levels based on price action
            price_range = recent_high - recent_low
            current_price = closes[-1]
            
            # Resistance levels (above current price)
            resistance_levels = [
                current_price + (price_range * 0.1),
                current_price + (price_range * 0.2),
                recent_high
            ]
            
            # Support levels (below current price)
            support_levels = [
                current_price - (price_range * 0.1),
                current_price - (price_range * 0.2),
                recent_low
            ]
            
            return {
                "support": sorted(support_levels, reverse=True),
                "resistance": sorted(resistance_levels),
                "current_price": current_price,
                "range_high": recent_high,
                "range_low": recent_low
            }
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance for {symbol_key}: {e}")
            return {"support": [], "resistance": []}
    
    def get_volatility_metrics(self, symbol: str, interval: str, period: int = 20) -> Dict:
        """Calculate volatility metrics"""
        symbol_key = f"{symbol}_{interval}"
        
        if symbol_key not in self.symbol_data:
            return {"volatility": 0, "atr": 0}
        
        data = self.symbol_data[symbol_key]
        
        if len(data['close']) < period:
            return {"volatility": 0, "atr": 0}
        
        try:
            closes = np.array(data['close'][-period:])
            highs = np.array(data['high'][-period:])
            lows = np.array(data['low'][-period:])
            
            # Calculate price volatility (standard deviation of returns)
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns) if len(returns) > 1 else 0
            
            # Calculate Average True Range (ATR)
            if len(highs) >= 14:
                atr = talib.ATR(highs, lows, closes, timeperiod=min(14, len(closes)))
                atr_value = float(atr[-1]) if not np.isnan(atr[-1]) else 0
            else:
                atr_value = 0
            
            return {
                "volatility": float(volatility),
                "atr": atr_value,
                "price_range": float(np.max(closes) - np.min(closes)),
                "relative_volatility": float(volatility / np.mean(closes)) if np.mean(closes) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol_key}: {e}")
            return {"volatility": 0, "atr": 0}
    
    def clear_old_data(self, max_age_hours: int = 24):
        """Clean up old data to manage memory"""
        current_time = int(datetime.utcnow().timestamp() * 1000)
        max_age_ms = max_age_hours * 60 * 60 * 1000
        
        symbols_to_remove = []
        
        for symbol_key, data in self.symbol_data.items():
            if not data['timestamps']:
                symbols_to_remove.append(symbol_key)
                continue
            
            # Keep only recent data
            cutoff_time = current_time - max_age_ms
            valid_indices = [i for i, ts in enumerate(data['timestamps']) if ts >= cutoff_time]
            
            if not valid_indices:
                symbols_to_remove.append(symbol_key)
            else:
                # Keep only recent data points
                for key in data:
                    data[key] = [data[key][i] for i in valid_indices]
        
        # Remove empty symbols
        for symbol_key in symbols_to_remove:
            del self.symbol_data[symbol_key]
        
        if symbols_to_remove:
            logger.info(f"Cleaned up {len(symbols_to_remove)} old symbol datasets")
    
    def get_data_summary(self) -> Dict:
        """Get summary of stored data"""
        summary = {}
        
        for symbol_key, data in self.symbol_data.items():
            summary[symbol_key] = {
                "data_points": len(data['timestamps']),
                "latest_timestamp": max(data['timestamps']) if data['timestamps'] else None,
                "price_range": {
                    "high": max(data['high']) if data['high'] else None,
                    "low": min(data['low']) if data['low'] else None,
                    "current": data['close'][-1] if data['close'] else None
                }
            }
        
        return summary