"""
Advanced Reinforcement Learning Agents for Autonomous Trading
Implementing PPO, A3C, DQN algorithms for adaptive trading strategies
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from collections import deque
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """Configuration for RL agents"""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    gae_lambda: float = 0.95
    target_kl: float = 0.01
    normalize_advantage: bool = True
    use_sde: bool = False
    sde_sample_freq: int = -1


class TradingEnvironment(gym.Env):
    """
    Advanced Trading Environment for RL Agents
    Optimized for high-frequency trading decisions
    """
    
    def __init__(self, 
                 data_manager,
                 ta_engine, 
                 risk_manager,
                 symbols: List[str] = ['BTCUSDT', 'ETHUSDT'],
                 lookback_window: int = 100,
                 initial_balance: float = 10000.0,
                 transaction_cost: float = 0.001,
                 max_positions: int = 5):
        
        super().__init__()
        
        self.data_manager = data_manager
        self.ta_engine = ta_engine
        self.risk_manager = risk_manager
        self.symbols = symbols
        self.lookback_window = lookback_window
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_positions = max_positions
        
        # State space: [prices, indicators, portfolio, positions]
        # Price features: OHLCV for each symbol (5 * len(symbols))
        # Technical indicators: 20 indicators per symbol
        # Portfolio: balance, equity, positions (3 + len(symbols) * 2)
        state_size = (
            len(symbols) * (5 + 20) +  # OHLCV + indicators per symbol
            3 +  # portfolio metrics
            len(symbols) * 2  # position info per symbol
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
        )
        
        # Action space: [symbol_index, action_type, position_size]
        # symbol_index: which symbol to trade (0 to len(symbols)-1)
        # action_type: 0=hold, 1=buy, 2=sell
        # position_size: 0.0 to 1.0 (fraction of available capital)
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0.0]), 
            high=np.array([len(symbols)-1, 2, 1.0]), 
            dtype=np.float32
        )
        
        # Environment state
        self.reset()
        
        logger.info(f"Trading Environment initialized for {len(symbols)} symbols")
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.position_values = {symbol: 0.0 for symbol in self.symbols}
        self.entry_prices = {symbol: 0.0 for symbol in self.symbols}
        
        self.current_step = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_equity = self.initial_balance
        self.max_drawdown = 0.0
        
        # History tracking
        self.price_history = deque(maxlen=self.lookback_window)
        self.indicator_history = deque(maxlen=self.lookback_window)
        self.action_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=1000)
        
        # Load initial market data
        self._load_market_data()
        
        return self._get_observation(), {}
    
    def _load_market_data(self):
        """Load market data for simulation"""
        try:
            # Get historical data for all symbols
            self.market_data = {}
            for symbol in self.symbols:
                # In real implementation, this would fetch actual market data
                # For now, simulate realistic price movements
                data = self._simulate_market_data(symbol, 10000)  # 10k data points
                self.market_data[symbol] = data
            
            self.max_steps = min(len(data) for data in self.market_data.values()) - self.lookback_window
            
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            # Fallback to simulated data
            self._create_simulated_data()
    
    def _simulate_market_data(self, symbol: str, length: int) -> pd.DataFrame:
        """Simulate realistic market data for testing"""
        np.random.seed(42)  # For reproducible results
        
        # Generate price series with trend and volatility
        initial_price = 50000 if 'BTC' in symbol else 3000
        returns = np.random.normal(0.0005, 0.02, length)  # Small positive drift
        
        prices = [initial_price]
        for r in returns:
            new_price = prices[-1] * (1 + r)
            prices.append(max(new_price, prices[-1] * 0.95))  # Prevent extreme crashes
        
        prices = np.array(prices[1:])
        
        # Generate OHLCV data
        highs = prices * (1 + np.abs(np.random.normal(0, 0.01, length)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.01, length)))
        volumes = np.random.lognormal(10, 1, length)
        
        return pd.DataFrame({
            'open': prices,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes,
            'timestamp': pd.date_range(start='2023-01-01', periods=length, freq='5T')
        })
    
    def _create_simulated_data(self):
        """Fallback simulation if data loading fails"""
        self.market_data = {}
        for symbol in self.symbols:
            self.market_data[symbol] = self._simulate_market_data(symbol, 5000)
        self.max_steps = 4900  # 5000 - 100 lookback
    
    def step(self, action):
        """Execute action in environment"""
        try:
            # Parse action
            symbol_idx = int(np.clip(action[0], 0, len(self.symbols)-1))
            action_type = int(np.clip(action[1], 0, 2))
            position_size = float(np.clip(action[2], 0.0, 1.0))
            
            symbol = self.symbols[symbol_idx]
            current_price = self._get_current_price(symbol)
            
            # Execute trade
            reward = self._execute_trade(symbol, action_type, position_size, current_price)
            
            # Update environment state
            self.current_step += 1
            self._update_portfolio_metrics()
            
            # Check if episode is done
            done = (
                self.current_step >= self.max_steps or
                self.equity <= self.initial_balance * 0.5 or  # Stop loss
                self.equity >= self.initial_balance * 3.0     # Take profit
            )
            
            # Additional info
            info = {
                'equity': self.equity,
                'balance': self.balance,
                'total_pnl': self.total_pnl,
                'total_trades': self.total_trades,
                'win_rate': self.winning_trades / max(1, self.total_trades),
                'max_drawdown': self.max_drawdown,
                'positions': dict(self.positions)
            }
            
            # Store action and reward
            self.action_history.append({
                'step': self.current_step,
                'symbol': symbol,
                'action_type': action_type,
                'position_size': position_size,
                'price': current_price,
                'reward': reward
            })
            self.reward_history.append(reward)
            
            return self._get_observation(), reward, done, False, info
            
        except Exception as e:
            logger.error(f"Error in step: {e}")
            return self._get_observation(), -1.0, True, False, {}
    
    def _execute_trade(self, symbol: str, action_type: int, position_size: float, current_price: float) -> float:
        """Execute trading action and return reward"""
        reward = 0.0
        
        try:
            if action_type == 0:  # Hold
                # Small penalty for inaction in volatile markets
                volatility = self._get_volatility(symbol)
                if volatility > 0.03:  # High volatility
                    reward = -0.001
                else:
                    reward = 0.0
                    
            elif action_type == 1:  # Buy
                if position_size > 0 and self.positions[symbol] <= 0:  # Only if not already long
                    trade_value = self.balance * position_size * 0.95  # Leave some cash
                    shares = trade_value / current_price
                    cost = trade_value * self.transaction_cost
                    
                    if trade_value + cost <= self.balance:
                        # Execute buy order
                        old_position = self.positions[symbol]
                        self.positions[symbol] += shares
                        self.position_values[symbol] += trade_value
                        self.entry_prices[symbol] = current_price if old_position == 0 else \
                            (self.entry_prices[symbol] * old_position + current_price * shares) / self.positions[symbol]
                        
                        self.balance -= (trade_value + cost)
                        self.total_trades += 1
                        
                        # Reward based on expected profitability
                        momentum = self._get_momentum(symbol)
                        reward = momentum * 0.1  # Reward buying in upward momentum
            
            elif action_type == 2:  # Sell
                if position_size > 0 and self.positions[symbol] > 0:  # Only if holding position
                    shares_to_sell = min(self.positions[symbol], self.positions[symbol] * position_size)
                    trade_value = shares_to_sell * current_price
                    cost = trade_value * self.transaction_cost
                    
                    # Calculate P&L for this trade
                    entry_price = self.entry_prices[symbol]
                    pnl = (current_price - entry_price) * shares_to_sell
                    
                    # Execute sell order
                    self.positions[symbol] -= shares_to_sell
                    self.position_values[symbol] -= (shares_to_sell / max(0.001, self.positions[symbol] + shares_to_sell)) * self.position_values[symbol]
                    self.balance += (trade_value - cost)
                    self.total_pnl += pnl
                    
                    if pnl > 0:
                        self.winning_trades += 1
                        reward = pnl / self.initial_balance * 10  # Scale reward
                    else:
                        reward = pnl / self.initial_balance * 10  # Negative reward for losses
            
            # Risk penalty
            portfolio_risk = self._calculate_portfolio_risk()
            if portfolio_risk > 0.3:  # High risk threshold
                reward -= 0.01
                
            # Diversification reward
            active_positions = sum(1 for pos in self.positions.values() if abs(pos) > 0.001)
            if active_positions > 1:
                reward += 0.001  # Small diversification bonus
            
            return reward
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return -0.1  # Penalty for errors
    
    def _get_observation(self) -> np.ndarray:
        """Get current environment observation"""
        try:
            obs = []
            
            # Market data for each symbol
            for symbol in self.symbols:
                current_data = self.market_data[symbol].iloc[self.current_step]
                
                # Price features (normalized)
                base_price = self.market_data[symbol]['close'].iloc[max(0, self.current_step-20):self.current_step].mean()
                price_features = [
                    current_data['open'] / base_price - 1,
                    current_data['high'] / base_price - 1,
                    current_data['low'] / base_price - 1,
                    current_data['close'] / base_price - 1,
                    np.log(current_data['volume'] + 1) / 20  # Log-normalized volume
                ]
                obs.extend(price_features)
                
                # Technical indicators (mock implementation)
                indicators = self._get_indicators(symbol)
                obs.extend(indicators[:20])  # First 20 indicators
            
            # Portfolio metrics
            portfolio_features = [
                self.balance / self.initial_balance - 1,  # Balance ratio
                self.equity / self.initial_balance - 1,   # Equity ratio
                self.total_pnl / self.initial_balance     # P&L ratio
            ]
            obs.extend(portfolio_features)
            
            # Position features for each symbol
            for symbol in self.symbols:
                current_price = self._get_current_price(symbol)
                position_value = self.positions[symbol] * current_price
                obs.extend([
                    self.positions[symbol] / 1000,  # Normalized position size
                    position_value / self.initial_balance  # Position value ratio
                ])
            
            return np.array(obs, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error getting observation: {e}")
            # Return zero observation as fallback
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        try:
            return float(self.market_data[symbol]['close'].iloc[self.current_step])
        except:
            return 50000.0 if 'BTC' in symbol else 3000.0
    
    def _get_indicators(self, symbol: str) -> List[float]:
        """Get technical indicators for symbol (mock implementation)"""
        try:
            data = self.market_data[symbol].iloc[max(0, self.current_step-50):self.current_step+1]
            
            # Simple moving averages
            sma_5 = data['close'].rolling(5).mean().iloc[-1] if len(data) >= 5 else data['close'].iloc[-1]
            sma_20 = data['close'].rolling(20).mean().iloc[-1] if len(data) >= 20 else data['close'].iloc[-1]
            sma_50 = data['close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else data['close'].iloc[-1]
            
            current_price = data['close'].iloc[-1]
            
            indicators = [
                (current_price / sma_5 - 1) if sma_5 > 0 else 0,     # SMA5 ratio
                (current_price / sma_20 - 1) if sma_20 > 0 else 0,   # SMA20 ratio
                (current_price / sma_50 - 1) if sma_50 > 0 else 0,   # SMA50 ratio
                (sma_5 / sma_20 - 1) if sma_20 > 0 else 0,           # Trend strength
                self._get_rsi(data['close']) / 100 - 0.5,             # RSI (centered)
                self._get_momentum(symbol),                            # Momentum
                self._get_volatility(symbol),                          # Volatility
                np.random.normal(0, 0.01),  # Additional features (placeholder)
                np.random.normal(0, 0.01),
                np.random.normal(0, 0.01),
                np.random.normal(0, 0.01),
                np.random.normal(0, 0.01),
                np.random.normal(0, 0.01),
                np.random.normal(0, 0.01),
                np.random.normal(0, 0.01),
                np.random.normal(0, 0.01),
                np.random.normal(0, 0.01),
                np.random.normal(0, 0.01),
                np.random.normal(0, 0.01),
                np.random.normal(0, 0.01),
            ]
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return [0.0] * 20
    
    def _get_rsi(self, prices: pd.Series, window: int = 14) -> float:
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    def _get_momentum(self, symbol: str) -> float:
        """Calculate price momentum"""
        try:
            data = self.market_data[symbol].iloc[max(0, self.current_step-10):self.current_step+1]
            if len(data) >= 2:
                return float((data['close'].iloc[-1] / data['close'].iloc[0] - 1))
            return 0.0
        except:
            return 0.0
    
    def _get_volatility(self, symbol: str) -> float:
        """Calculate price volatility"""
        try:
            data = self.market_data[symbol].iloc[max(0, self.current_step-20):self.current_step+1]
            if len(data) >= 2:
                returns = data['close'].pct_change().dropna()
                return float(returns.std()) if len(returns) > 1 else 0.02
            return 0.02
        except:
            return 0.02
    
    def _update_portfolio_metrics(self):
        """Update portfolio metrics"""
        try:
            # Calculate current equity
            position_value = 0
            for symbol in self.symbols:
                current_price = self._get_current_price(symbol)
                position_value += self.positions[symbol] * current_price
            
            self.equity = self.balance + position_value
            
            # Update max drawdown
            self.max_equity = max(self.max_equity, self.equity)
            current_drawdown = (self.max_equity - self.equity) / self.max_equity
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")
    
    def _calculate_portfolio_risk(self) -> float:
        """Calculate portfolio risk level"""
        try:
            total_position_value = sum(abs(self.positions[symbol] * self._get_current_price(symbol)) 
                                     for symbol in self.symbols)
            return total_position_value / self.equity if self.equity > 0 else 0.0
        except:
            return 0.0


class TradingRLAgent:
    """
    Advanced RL Agent for Autonomous Trading
    Supports multiple algorithms (PPO, A2C, DQN, SAC)
    """
    
    def __init__(self, 
                 data_manager,
                 ta_engine, 
                 risk_manager,
                 algorithm: str = 'PPO',
                 symbols: List[str] = ['BTCUSDT', 'ETHUSDT'],
                 config: Optional[RLConfig] = None):
        
        self.data_manager = data_manager
        self.ta_engine = ta_engine
        self.risk_manager = risk_manager
        self.algorithm = algorithm
        self.symbols = symbols
        self.config = config or RLConfig()
        
        # Create trading environment
        self.env = TradingEnvironment(
            data_manager=data_manager,
            ta_engine=ta_engine,
            risk_manager=risk_manager,
            symbols=symbols
        )
        
        # Initialize RL model
        self.model = self._create_model()
        
        # Training metrics
        self.training_history = []
        self.performance_metrics = {}
        
        logger.info(f"RL Agent initialized with {algorithm} algorithm")
    
    def _create_model(self):
        """Create RL model based on algorithm"""
        try:
            if self.algorithm == 'PPO':
                return PPO(
                    'MlpPolicy',
                    self.env,
                    learning_rate=self.config.learning_rate,
                    gamma=self.config.gamma,
                    n_steps=self.config.n_steps,
                    batch_size=self.config.batch_size,
                    n_epochs=self.config.n_epochs,
                    clip_range=self.config.clip_range,
                    vf_coef=self.config.vf_coef,
                    ent_coef=self.config.ent_coef,
                    max_grad_norm=self.config.max_grad_norm,
                    gae_lambda=self.config.gae_lambda,
                    target_kl=self.config.target_kl,
                    verbose=1,
                    tensorboard_log="./tensorboard_logs/"
                )
            
            elif self.algorithm == 'A2C':
                return A2C(
                    'MlpPolicy',
                    self.env,
                    learning_rate=self.config.learning_rate,
                    gamma=self.config.gamma,
                    n_steps=self.config.n_steps,
                    vf_coef=self.config.vf_coef,
                    ent_coef=self.config.ent_coef,
                    max_grad_norm=self.config.max_grad_norm,
                    gae_lambda=self.config.gae_lambda,
                    normalize_advantage=self.config.normalize_advantage,
                    use_sde=self.config.use_sde,
                    verbose=1,
                    tensorboard_log="./tensorboard_logs/"
                )
            
            elif self.algorithm == 'DQN':
                return DQN(
                    'MlpPolicy',
                    self.env,
                    learning_rate=self.config.learning_rate,
                    gamma=self.config.gamma,
                    batch_size=self.config.batch_size,
                    buffer_size=100000,
                    learning_starts=1000,
                    target_update_interval=1000,
                    train_freq=1,
                    verbose=1,
                    tensorboard_log="./tensorboard_logs/"
                )
            
            elif self.algorithm == 'SAC':
                return SAC(
                    'MlpPolicy',
                    self.env,
                    learning_rate=self.config.learning_rate,
                    gamma=self.config.gamma,
                    batch_size=self.config.batch_size,
                    buffer_size=100000,
                    learning_starts=1000,
                    train_freq=1,
                    verbose=1,
                    tensorboard_log="./tensorboard_logs/"
                )
            
            else:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}")
                
        except Exception as e:
            logger.error(f"Error creating RL model: {e}")
            # Fallback to PPO
            return PPO('MlpPolicy', self.env, verbose=1)
    
    def train(self, total_timesteps: int = 100000, eval_freq: int = 10000):
        """Train the RL agent"""
        try:
            logger.info(f"Starting RL training for {total_timesteps} timesteps...")
            
            # Custom callback for evaluation
            class EvalCallback(BaseCallback):
                def __init__(self, eval_env, eval_freq, agent):
                    super().__init__()
                    self.eval_env = eval_env
                    self.eval_freq = eval_freq
                    self.agent = agent
                    self.best_mean_reward = -np.inf
                
                def _on_step(self) -> bool:
                    if self.n_calls % self.eval_freq == 0:
                        # Evaluate performance
                        mean_reward, std_reward = self.agent.evaluate_performance(n_episodes=5)
                        
                        # Log metrics
                        self.logger.record('eval/mean_reward', mean_reward)
                        self.logger.record('eval/std_reward', std_reward)
                        
                        # Save best model
                        if mean_reward > self.best_mean_reward:
                            self.best_mean_reward = mean_reward
                            self.agent.save_model(f"./models/best_{self.agent.algorithm}_model")
                    
                    return True
            
            # Train model with callback
            callback = EvalCallback(self.env, eval_freq, self)
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=True
            )
            
            # Save final model
            self.save_model(f"./models/{self.algorithm}_final_model")
            
            logger.info("RL training completed successfully")
            
        except Exception as e:
            logger.error(f"Error during RL training: {e}")
    
    def predict_action(self, observation=None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict trading action using trained model"""
        try:
            if observation is None:
                observation = self.env._get_observation()
            
            action, _states = self.model.predict(observation, deterministic=True)
            return action, _states
            
        except Exception as e:
            logger.error(f"Error predicting action: {e}")
            # Return neutral action as fallback
            return np.array([0, 0, 0.0]), None
    
    def evaluate_performance(self, n_episodes: int = 10) -> Tuple[float, float]:
        """Evaluate agent performance"""
        try:
            episode_rewards = []
            
            for episode in range(n_episodes):
                obs, _ = self.env.reset()
                total_reward = 0
                done = False
                
                while not done:
                    action, _states = self.predict_action(obs)
                    obs, reward, done, truncated, info = self.env.step(action)
                    total_reward += reward
                    
                    if truncated:
                        done = True
                
                episode_rewards.append(total_reward)
            
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            
            # Update performance metrics
            self.performance_metrics.update({
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'episodes_evaluated': n_episodes,
                'evaluation_time': datetime.utcnow().isoformat()
            })
            
            logger.info(f"Performance evaluation: Mean reward = {mean_reward:.4f} ± {std_reward:.4f}")
            
            return mean_reward, std_reward
            
        except Exception as e:
            logger.error(f"Error evaluating performance: {e}")
            return 0.0, 0.0
    
    def save_model(self, path: str):
        """Save trained model"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)
            
            # Save additional metadata
            metadata = {
                'algorithm': self.algorithm,
                'symbols': self.symbols,
                'config': self.config.__dict__,
                'performance_metrics': self.performance_metrics,
                'saved_at': datetime.utcnow().isoformat()
            }
            
            with open(f"{path}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, path: str):
        """Load trained model"""
        try:
            if self.algorithm == 'PPO':
                self.model = PPO.load(path)
            elif self.algorithm == 'A2C':
                self.model = A2C.load(path)
            elif self.algorithm == 'DQN':
                self.model = DQN.load(path)
            elif self.algorithm == 'SAC':
                self.model = SAC.load(path)
            
            # Load metadata if available
            try:
                with open(f"{path}_metadata.json", 'r') as f:
                    metadata = json.load(f)
                    self.performance_metrics = metadata.get('performance_metrics', {})
            except FileNotFoundError:
                pass
            
            logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def get_agent_status(self) -> Dict:
        """Get current agent status"""
        return {
            'algorithm': self.algorithm,
            'symbols': self.symbols,
            'model_trained': hasattr(self.model, 'policy'),
            'performance_metrics': self.performance_metrics,
            'environment_info': {
                'observation_space': str(self.env.observation_space),
                'action_space': str(self.env.action_space),
                'max_steps': getattr(self.env, 'max_steps', 0)
            }
        }


class MultiAgentRLSystem:
    """
    Multi-Agent RL System with competing strategies
    Each agent specializes in different market conditions
    """
    
    def __init__(self, 
                 data_manager,
                 ta_engine, 
                 risk_manager,
                 symbols: List[str] = ['BTCUSDT', 'ETHUSDT']):
        
        self.data_manager = data_manager
        self.ta_engine = ta_engine
        self.risk_manager = risk_manager
        self.symbols = symbols
        
        # Create specialized agents
        self.agents = {
            'momentum_agent': TradingRLAgent(
                data_manager, ta_engine, risk_manager, 
                algorithm='PPO', symbols=symbols
            ),
            'mean_reversion_agent': TradingRLAgent(
                data_manager, ta_engine, risk_manager, 
                algorithm='A2C', symbols=symbols
            ),
            'breakout_agent': TradingRLAgent(
                data_manager, ta_engine, risk_manager, 
                algorithm='SAC', symbols=symbols
            ),
            'volatility_agent': TradingRLAgent(
                data_manager, ta_engine, risk_manager, 
                algorithm='DQN', symbols=symbols
            )
        }
        
        # Agent performance tracking
        self.agent_performance = {name: {'wins': 0, 'total': 0, 'avg_reward': 0.0} 
                                for name in self.agents.keys()}
        
        # Voting system for action selection
        self.voting_weights = {name: 1.0 for name in self.agents.keys()}
        
        logger.info(f"Multi-Agent RL System initialized with {len(self.agents)} agents")
    
    def train_all_agents(self, total_timesteps: int = 50000):
        """Train all agents in parallel"""
        try:
            import concurrent.futures
            
            def train_agent(name_agent_pair):
                name, agent = name_agent_pair
                logger.info(f"Training {name}...")
                agent.train(total_timesteps // 2)  # Reduced timesteps per agent
                return name, agent.performance_metrics
            
            # Train agents in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(train_agent, item) for item in self.agents.items()]
                
                for future in concurrent.futures.as_completed(futures):
                    name, metrics = future.result()
                    logger.info(f"Completed training for {name}: {metrics}")
            
            logger.info("All agents trained successfully")
            
        except Exception as e:
            logger.error(f"Error training agents: {e}")
    
    def get_ensemble_action(self, market_conditions: Dict) -> Tuple[str, np.ndarray]:
        """Get action from ensemble of agents based on market conditions"""
        try:
            agent_predictions = {}
            agent_confidences = {}
            
            # Get predictions from all agents
            for name, agent in self.agents.items():
                action, _states = agent.predict_action()
                confidence = self._calculate_agent_confidence(name, market_conditions)
                
                agent_predictions[name] = action
                agent_confidences[name] = confidence
            
            # Select best agent based on market conditions and performance
            best_agent = self._select_best_agent(market_conditions, agent_confidences)
            best_action = agent_predictions[best_agent]
            
            return best_agent, best_action
            
        except Exception as e:
            logger.error(f"Error getting ensemble action: {e}")
            return 'momentum_agent', np.array([0, 0, 0.0])
    
    def _calculate_agent_confidence(self, agent_name: str, market_conditions: Dict) -> float:
        """Calculate agent confidence based on market conditions"""
        try:
            base_confidence = 0.5
            
            # Adjust confidence based on market conditions and agent specialty
            volatility = market_conditions.get('volatility', 0.02)
            trend_strength = market_conditions.get('trend_strength', 0.0)
            
            if agent_name == 'momentum_agent':
                # High confidence in trending markets
                base_confidence += abs(trend_strength) * 0.3
                
            elif agent_name == 'mean_reversion_agent':
                # High confidence in sideways markets
                base_confidence += (1 - abs(trend_strength)) * 0.3
                
            elif agent_name == 'breakout_agent':
                # High confidence in volatile markets
                base_confidence += min(volatility * 10, 0.3)
                
            elif agent_name == 'volatility_agent':
                # High confidence in highly volatile markets
                base_confidence += min(volatility * 15, 0.4)
            
            # Adjust for historical performance
            performance = self.agent_performance[agent_name]
            if performance['total'] > 0:
                win_rate = performance['wins'] / performance['total']
                base_confidence *= (0.5 + win_rate)
            
            return min(max(base_confidence, 0.1), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _select_best_agent(self, market_conditions: Dict, confidences: Dict) -> str:
        """Select the best agent for current market conditions"""
        try:
            # Weight confidences by recent performance
            weighted_scores = {}
            
            for name, confidence in confidences.items():
                performance_weight = self.voting_weights[name]
                weighted_scores[name] = confidence * performance_weight
            
            # Select agent with highest weighted score
            best_agent = max(weighted_scores, key=weighted_scores.get)
            
            return best_agent
            
        except Exception as e:
            logger.error(f"Error selecting best agent: {e}")
            return 'momentum_agent'
    
    def update_agent_performance(self, agent_name: str, success: bool, reward: float):
        """Update agent performance metrics"""
        try:
            performance = self.agent_performance[agent_name]
            performance['total'] += 1
            
            if success:
                performance['wins'] += 1
            
            # Update average reward (exponential moving average)
            alpha = 0.1
            performance['avg_reward'] = (1 - alpha) * performance['avg_reward'] + alpha * reward
            
            # Update voting weights based on performance
            if performance['total'] >= 10:  # Minimum samples for reliable statistics
                win_rate = performance['wins'] / performance['total']
                self.voting_weights[agent_name] = 0.5 + win_rate
            
        except Exception as e:
            logger.error(f"Error updating agent performance: {e}")
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            'agents': {name: agent.get_agent_status() for name, agent in self.agents.items()},
            'performance': dict(self.agent_performance),
            'voting_weights': dict(self.voting_weights),
            'total_agents': len(self.agents)
        }


class AdvancedRLTrainer:
    """
    Advanced trainer with hyperparameter optimization and automated model selection
    """
    
    def __init__(self, 
                 data_manager,
                 ta_engine, 
                 risk_manager,
                 symbols: List[str] = ['BTCUSDT', 'ETHUSDT']):
        
        self.data_manager = data_manager
        self.ta_engine = ta_engine
        self.risk_manager = risk_manager
        self.symbols = symbols
        
        # Hyperparameter optimization
        self.study = None
        self.best_params = {}
        self.training_results = {}
        
        logger.info("Advanced RL Trainer initialized")
    
    def optimize_hyperparameters(self, n_trials: int = 20) -> Dict:
        """Optimize hyperparameters using Optuna"""
        try:
            import optuna
            
            def objective(trial):
                # Suggest hyperparameters
                config = RLConfig(
                    learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                    gamma=trial.suggest_float('gamma', 0.9, 0.999),
                    n_steps=trial.suggest_int('n_steps', 512, 4096),
                    batch_size=trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                    n_epochs=trial.suggest_int('n_epochs', 3, 20),
                    clip_range=trial.suggest_float('clip_range', 0.1, 0.4),
                    ent_coef=trial.suggest_float('ent_coef', 0.001, 0.1, log=True)
                )
                
                # Create and train agent
                agent = TradingRLAgent(
                    self.data_manager, self.ta_engine, self.risk_manager,
                    algorithm='PPO', symbols=self.symbols, config=config
                )
                
                # Quick training for optimization
                agent.train(total_timesteps=10000)
                
                # Evaluate performance
                mean_reward, _ = agent.evaluate_performance(n_episodes=3)
                
                return mean_reward
            
            # Create and run study
            self.study = optuna.create_study(direction='maximize')
            self.study.optimize(objective, n_trials=n_trials)
            
            self.best_params = self.study.best_params
            
            logger.info(f"Hyperparameter optimization completed. Best params: {self.best_params}")
            
            return self.best_params
            
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {e}")
            return {}
    
    def train_optimal_model(self, total_timesteps: int = 100000) -> TradingRLAgent:
        """Train model with optimized hyperparameters"""
        try:
            if not self.best_params:
                logger.info("No optimized parameters found, using default config")
                config = RLConfig()
            else:
                config = RLConfig(**self.best_params)
            
            # Create and train agent with optimal configuration
            agent = TradingRLAgent(
                self.data_manager, self.ta_engine, self.risk_manager,
                algorithm='PPO', symbols=self.symbols, config=config
            )
            
            agent.train(total_timesteps=total_timesteps)
            
            # Final evaluation
            final_performance = agent.evaluate_performance(n_episodes=10)
            self.training_results['final_performance'] = final_performance
            
            logger.info(f"Optimal model training completed with performance: {final_performance}")
            
            return agent
            
        except Exception as e:
            logger.error(f"Error training optimal model: {e}")
            return None
    
    def get_training_status(self) -> Dict:
        """Get comprehensive training status"""
        return {
            'optimization_completed': self.study is not None,
            'best_parameters': self.best_params,
            'training_results': self.training_results,
            'study_trials': len(self.study.trials) if self.study else 0
        }