"""
Risk Management System for FinGPT Trading System
Real-time risk monitoring and position management
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from .data_models import Position, Portfolio, Order, RiskMetrics, OrderSide

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk management limits and parameters"""
    max_portfolio_risk: float = 0.10  # 10% max drawdown
    max_position_size: float = 0.05  # 5% per position
    max_sector_exposure: float = 0.25  # 25% per sector
    max_correlation: float = 0.7  # Max correlation between positions
    daily_loss_limit: float = 0.02  # 2% daily loss limit
    max_leverage: float = 1.0  # No leverage initially
    var_limit_1d: float = 0.05  # 5% Value at Risk (1 day)
    concentration_limit: float = 0.15  # 15% max in single asset
    
    # Position specific
    stop_loss_buffer: float = 0.001  # 0.1% buffer for stop loss
    min_risk_reward: float = 1.5  # Minimum risk:reward ratio


class PositionSizer:
    """Calculate optimal position sizes using various methods"""
    
    def __init__(self, risk_limits: RiskLimits):
        self.risk_limits = risk_limits
    
    def kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float, 
                       portfolio_balance: float) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        if avg_loss <= 0 or win_rate <= 0:
            return 0
        
        # Kelly percentage: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1 - win_rate
        
        kelly_pct = (b * p - q) / b
        
        # Apply conservative scaling (25% of Kelly for safety)
        conservative_kelly = kelly_pct * 0.25
        
        # Cap at maximum position size
        max_pct = self.risk_limits.max_position_size
        final_pct = min(conservative_kelly, max_pct)
        
        return max(0, final_pct)
    
    def fixed_fractional(self, risk_per_trade: float, stop_loss_distance: float,
                        entry_price: float, portfolio_balance: float) -> float:
        """Calculate position size using fixed fractional method"""
        if stop_loss_distance <= 0 or entry_price <= 0:
            return 0
        
        # Risk per share/unit
        risk_per_unit = stop_loss_distance
        
        # Maximum dollar risk
        max_risk = portfolio_balance * risk_per_trade
        
        # Position size in units
        position_units = max_risk / risk_per_unit
        
        # Position size as percentage of portfolio
        position_value = position_units * entry_price
        position_pct = position_value / portfolio_balance
        
        # Cap at maximum position size
        return min(position_pct, self.risk_limits.max_position_size)
    
    def volatility_adjusted(self, volatility: float, base_position_size: float) -> float:
        """Adjust position size based on asset volatility"""
        # Target volatility (2% daily)
        target_vol = 0.02
        
        if volatility <= 0:
            return base_position_size
        
        # Scale position size inversely with volatility
        vol_adjustment = target_vol / volatility
        
        # Cap the adjustment (don't go beyond 2x or below 0.3x)
        vol_adjustment = max(0.3, min(2.0, vol_adjustment))
        
        adjusted_size = base_position_size * vol_adjustment
        
        return min(adjusted_size, self.risk_limits.max_position_size)


class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, risk_limits: Optional[RiskLimits] = None):
        self.risk_limits = risk_limits or RiskLimits()
        self.position_sizer = PositionSizer(self.risk_limits)
        self.daily_pnl_tracker = {}
        self.risk_metrics_history = []
    
    def calculate_portfolio_risk(self, portfolio: Portfolio, positions: List[Position]) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        
        if not positions:
            return RiskMetrics(
                portfolio_id=portfolio.id,
                var_1d=0.0,
                var_1w=0.0,
                portfolio_beta=0.0,
                concentration_risk=0.0,
                correlation_risk=0.0,
                leverage_ratio=0.0,
                margin_usage=0.0
            )
        
        total_exposure = sum(abs(pos.size * pos.current_price) for pos in positions)
        portfolio_value = portfolio.current_balance
        
        # Calculate Value at Risk (simplified)
        position_weights = []
        position_returns = []
        
        for pos in positions:
            weight = (pos.size * pos.current_price) / portfolio_value
            position_weights.append(weight)
            
            # Simplified return estimation (would need historical data for real calculation)
            daily_return_estimate = pos.unrealized_pnl / (pos.size * pos.entry_price)
            position_returns.append(daily_return_estimate)
        
        # Portfolio variance (simplified)
        if len(position_weights) > 1:
            portfolio_variance = np.var(position_returns) if position_returns else 0
        else:
            portfolio_variance = position_returns[0]**2 if position_returns else 0
        
        # VaR calculation (95% confidence, normal distribution assumption)
        var_1d = 1.645 * np.sqrt(portfolio_variance) * portfolio_value
        var_1w = var_1d * np.sqrt(5)  # Assuming 5 trading days per week
        
        # Concentration risk (largest position weight)
        concentration_risk = max(abs(w) for w in position_weights) if position_weights else 0
        
        # Average correlation (simplified - would need price correlation matrix)
        correlation_risk = 0.5  # Placeholder - would calculate from actual correlations
        
        # Portfolio beta (simplified)
        portfolio_beta = 1.0  # Placeholder - would calculate vs benchmark
        
        # Leverage ratio
        leverage_ratio = total_exposure / portfolio_value if portfolio_value > 0 else 0
        
        # Margin usage (assuming no margin for now)
        margin_usage = 0.0
        
        risk_metrics = RiskMetrics(
            portfolio_id=portfolio.id,
            var_1d=var_1d,
            var_1w=var_1w,
            portfolio_beta=portfolio_beta,
            concentration_risk=concentration_risk,
            correlation_risk=correlation_risk,
            leverage_ratio=leverage_ratio,
            margin_usage=margin_usage
        )
        
        # Store in history
        self.risk_metrics_history.append(risk_metrics)
        
        # Keep only last 30 days
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        self.risk_metrics_history = [
            rm for rm in self.risk_metrics_history 
            if rm.created_at >= cutoff_date
        ]
        
        return risk_metrics
    
    def validate_new_position(self, portfolio: Portfolio, positions: List[Position], 
                            new_position_size: float, symbol: str) -> Tuple[bool, str]:
        """Validate if a new position can be opened"""
        
        # Check position size limit
        if abs(new_position_size) > self.risk_limits.max_position_size:
            return False, f"Position size {new_position_size:.2%} exceeds limit {self.risk_limits.max_position_size:.2%}"
        
        # Check concentration risk
        current_exposure = {}
        for pos in positions:
            current_exposure[pos.symbol] = current_exposure.get(pos.symbol, 0) + abs(pos.size * pos.current_price)
        
        new_exposure = current_exposure.get(symbol, 0) + abs(new_position_size * portfolio.current_balance)
        concentration_pct = new_exposure / portfolio.current_balance
        
        if concentration_pct > self.risk_limits.concentration_limit:
            return False, f"Concentration in {symbol} would be {concentration_pct:.2%}, exceeds limit {self.risk_limits.concentration_limit:.2%}"
        
        # Check daily loss limit
        today = datetime.utcnow().date()
        daily_pnl = self.daily_pnl_tracker.get(today, 0)
        daily_loss_pct = daily_pnl / portfolio.initial_balance
        
        if daily_loss_pct < -self.risk_limits.daily_loss_limit:
            return False, f"Daily loss limit reached: {daily_loss_pct:.2%}"
        
        # Check total portfolio exposure
        total_current_exposure = sum(current_exposure.values())
        new_total_exposure = total_current_exposure + abs(new_position_size * portfolio.current_balance)
        total_exposure_pct = new_total_exposure / portfolio.current_balance
        
        if total_exposure_pct > 1.0:  # 100% invested
            return False, f"Total portfolio exposure would be {total_exposure_pct:.2%}, exceeds 100%"
        
        return True, "Position approved"
    
    def calculate_optimal_position_size(self, portfolio: Portfolio, entry_price: float, 
                                      stop_loss: float, confidence: float, 
                                      volatility: float = 0.02) -> float:
        """Calculate optimal position size considering multiple factors"""
        
        # Base risk per trade (1-2% of portfolio)
        base_risk = 0.015  # 1.5%
        
        # Adjust for confidence
        confidence_adjusted_risk = base_risk * min(confidence / 0.7, 1.5)
        
        # Calculate stop loss distance
        stop_loss_distance = abs(entry_price - stop_loss)
        
        # Use fixed fractional method
        position_size = self.position_sizer.fixed_fractional(
            confidence_adjusted_risk, stop_loss_distance, entry_price, portfolio.current_balance
        )
        
        # Adjust for volatility
        position_size = self.position_sizer.volatility_adjusted(volatility, position_size)
        
        # Final safety cap
        return min(position_size, self.risk_limits.max_position_size)
    
    def should_close_position(self, position: Position, current_price: float, 
                            portfolio: Portfolio) -> Tuple[bool, str]:
        """Check if a position should be closed due to risk management"""
        
        # Update unrealized P&L
        if position.side == OrderSide.BUY:
            unrealized_pnl = (current_price - position.entry_price) * position.size
        else:
            unrealized_pnl = (position.entry_price - current_price) * position.size
        
        # Check stop loss
        if position.stop_loss:
            if position.side == OrderSide.BUY and current_price <= position.stop_loss:
                return True, "Stop loss triggered"
            elif position.side == OrderSide.SELL and current_price >= position.stop_loss:
                return True, "Stop loss triggered"
        
        # Check take profit
        if position.take_profit:
            if position.side == OrderSide.BUY and current_price >= position.take_profit:
                return True, "Take profit triggered"
            elif position.side == OrderSide.SELL and current_price <= position.take_profit:
                return True, "Take profit triggered"
        
        # Check position loss limit (5% of position value)
        position_value = position.size * position.entry_price
        loss_limit = position_value * 0.05
        
        if unrealized_pnl < -loss_limit:
            return True, f"Position loss limit exceeded: {unrealized_pnl:.2f}"
        
        # Check portfolio drawdown
        total_unrealized = portfolio.total_pnl + unrealized_pnl
        drawdown = (portfolio.initial_balance + total_unrealized - portfolio.initial_balance) / portfolio.initial_balance
        
        if drawdown < -self.risk_limits.max_portfolio_risk:
            return True, f"Portfolio drawdown limit exceeded: {drawdown:.2%}"
        
        return False, "Position within risk limits"
    
    def update_daily_pnl(self, pnl_change: float):
        """Update daily P&L tracking"""
        today = datetime.utcnow().date()
        self.daily_pnl_tracker[today] = self.daily_pnl_tracker.get(today, 0) + pnl_change
        
        # Clean up old entries (keep last 30 days)
        cutoff_date = today - timedelta(days=30)
        self.daily_pnl_tracker = {
            date: pnl for date, pnl in self.daily_pnl_tracker.items()
            if date >= cutoff_date
        }
    
    def get_risk_summary(self, portfolio: Portfolio, positions: List[Position]) -> Dict:
        """Get comprehensive risk summary"""
        
        risk_metrics = self.calculate_portfolio_risk(portfolio, positions)
        
        # Calculate current exposure by symbol
        exposures = {}
        for pos in positions:
            symbol = pos.symbol
            exposure = abs(pos.size * pos.current_price)
            exposures[symbol] = exposures.get(symbol, 0) + exposure
        
        # Sort by exposure size
        sorted_exposures = sorted(exposures.items(), key=lambda x: x[1], reverse=True)
        
        # Risk status
        risk_status = "LOW"
        if risk_metrics.concentration_risk > 0.10:
            risk_status = "MEDIUM"
        if risk_metrics.concentration_risk > 0.20 or risk_metrics.var_1d / portfolio.current_balance > 0.03:
            risk_status = "HIGH"
        
        # Daily P&L
        today = datetime.utcnow().date()
        daily_pnl = self.daily_pnl_tracker.get(today, 0)
        
        return {
            "risk_status": risk_status,
            "portfolio_value": portfolio.current_balance,
            "total_exposure": sum(exposures.values()),
            "concentration_risk": risk_metrics.concentration_risk,
            "largest_position": sorted_exposures[0] if sorted_exposures else None,
            "daily_pnl": daily_pnl,
            "daily_pnl_pct": daily_pnl / portfolio.initial_balance if portfolio.initial_balance > 0 else 0,
            "var_1d": risk_metrics.var_1d,
            "var_1d_pct": risk_metrics.var_1d / portfolio.current_balance if portfolio.current_balance > 0 else 0,
            "leverage_ratio": risk_metrics.leverage_ratio,
            "positions_count": len(positions),
            "risk_limits": {
                "max_position_size": self.risk_limits.max_position_size,
                "daily_loss_limit": self.risk_limits.daily_loss_limit,
                "max_portfolio_risk": self.risk_limits.max_portfolio_risk,
                "concentration_limit": self.risk_limits.concentration_limit
            },
            "risk_utilization": {
                "position_size": risk_metrics.concentration_risk / self.risk_limits.max_position_size,
                "daily_loss": abs(daily_pnl / portfolio.initial_balance) / self.risk_limits.daily_loss_limit if portfolio.initial_balance > 0 else 0,
                "portfolio_risk": abs(portfolio.total_pnl / portfolio.initial_balance) / self.risk_limits.max_portfolio_risk if portfolio.initial_balance > 0 else 0
            }
        }
    
    def adjust_risk_limits(self, performance_metrics: Dict):
        """Dynamically adjust risk limits based on performance"""
        
        win_rate = performance_metrics.get("win_rate", 0.5)
        sharpe_ratio = performance_metrics.get("sharpe_ratio", 0.0)
        max_drawdown = performance_metrics.get("max_drawdown", 0.0)
        
        # Increase limits for good performance
        if win_rate > 0.65 and sharpe_ratio > 1.2 and max_drawdown < 0.05:
            self.risk_limits.max_position_size = min(0.08, self.risk_limits.max_position_size * 1.1)
            logger.info(f"Increased max position size to {self.risk_limits.max_position_size:.2%}")
        
        # Decrease limits for poor performance
        elif win_rate < 0.45 or sharpe_ratio < 0.5 or max_drawdown > 0.10:
            self.risk_limits.max_position_size = max(0.02, self.risk_limits.max_position_size * 0.9)
            logger.info(f"Decreased max position size to {self.risk_limits.max_position_size:.2%}")
    
    def emergency_risk_shutdown(self, portfolio: Portfolio) -> bool:
        """Check if emergency risk shutdown is needed"""
        
        # Portfolio drawdown exceeds emergency limit (15%)
        drawdown = (portfolio.initial_balance - portfolio.current_balance) / portfolio.initial_balance
        if drawdown > 0.15:
            logger.critical(f"EMERGENCY SHUTDOWN: Portfolio drawdown {drawdown:.2%} exceeds emergency limit")
            return True
        
        # Daily loss exceeds emergency limit (5%)
        today = datetime.utcnow().date()
        daily_pnl = self.daily_pnl_tracker.get(today, 0)
        daily_loss_pct = abs(daily_pnl) / portfolio.initial_balance
        
        if daily_loss_pct > 0.05:
            logger.critical(f"EMERGENCY SHUTDOWN: Daily loss {daily_loss_pct:.2%} exceeds emergency limit")
            return True
        
        return False