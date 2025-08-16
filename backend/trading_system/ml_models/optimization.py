"""
Advanced Optimization System for Hyperparameters, Strategies, and Portfolios
Using Optuna, Genetic Algorithms, and Bayesian Optimization
"""

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Optimization libraries
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from collections import defaultdict

# Portfolio optimization
try:
    import cvxpy as cp
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    PORTFOLIO_OPT_AVAILABLE = True
except ImportError:
    PORTFOLIO_OPT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for optimization processes"""
    n_trials: int = 100
    timeout: int = 3600  # 1 hour
    n_jobs: int = 4
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    direction: str = 'maximize'  # or 'minimize'
    pruner_patience: int = 10
    sampler_type: str = 'tpe'  # 'tpe', 'cmaes', 'random'


@dataclass
class OptimizationResult:
    """Results from optimization process"""
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    optimization_time: float
    convergence_history: List[float]
    parameter_importance: Dict[str, float]
    study_metadata: Dict[str, Any]
    timestamp: datetime


class HyperparameterOptimizer:
    """
    Advanced hyperparameter optimization for ML models
    Supports multiple optimization algorithms and intelligent search strategies
    """
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.study = None
        self.optimization_history = []
        self.best_results = {}
        
        # Set up Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        logger.info("Hyperparameter Optimizer initialized")
    
    def optimize_lstm_parameters(self, 
                                train_data: pd.DataFrame, 
                                target_column: str = 'close',
                                model_class=None) -> OptimizationResult:
        """
        Optimize LSTM hyperparameters
        """
        try:
            logger.info("Starting LSTM hyperparameter optimization...")
            
            def objective(trial):
                # Define hyperparameter search space
                params = {
                    'sequence_length': trial.suggest_int('sequence_length', 30, 120),
                    'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256, 512]),
                    'num_layers': trial.suggest_int('num_layers', 1, 4),
                    'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                    'epochs': trial.suggest_int('epochs', 50, 200)
                }
                
                # Create and train model with suggested parameters
                try:
                    if model_class:
                        from .deep_learning import ModelConfig
                        config = ModelConfig(**params)
                        model = model_class(config)
                    else:
                        # Import here to avoid circular import
                        from .deep_learning import LSTMPricePredictor, ModelConfig
                        config = ModelConfig(**params)
                        model = LSTMPricePredictor(config)
                    
                    # Train with limited epochs for optimization speed
                    config.epochs = min(params['epochs'], 50)  # Limit for optimization
                    config.patience = 5  # Early stopping
                    
                    training_result = model.train(train_data, target_column)
                    
                    # Return validation loss (to minimize)
                    val_loss = training_result.get('best_val_loss', float('inf'))
                    
                    # Add penalty for overfitting
                    train_losses = training_result.get('train_losses', [])
                    val_losses = training_result.get('val_losses', [])
                    
                    if len(train_losses) > 10 and len(val_losses) > 10:
                        # Check for overfitting
                        recent_train_loss = np.mean(train_losses[-5:])
                        recent_val_loss = np.mean(val_losses[-5:])
                        
                        overfitting_penalty = max(0, (recent_val_loss - recent_train_loss) / recent_train_loss)
                        val_loss += overfitting_penalty * 0.1
                    
                    return val_loss
                    
                except Exception as e:
                    logger.warning(f"Trial failed: {e}")
                    return float('inf')
            
            # Create study
            sampler = self._get_sampler(self.config.sampler_type)
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
            
            study = optuna.create_study(
                direction='minimize',  # Minimize validation loss
                sampler=sampler,
                pruner=pruner,
                study_name=f"lstm_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Run optimization
            start_time = datetime.now()
            study.optimize(
                objective, 
                n_trials=self.config.n_trials,
                timeout=self.config.timeout,
                n_jobs=1,  # LSTM training is already parallelized
                show_progress_bar=True
            )
            end_time = datetime.now()
            
            # Analyze results
            best_params = study.best_params
            best_value = study.best_value
            optimization_time = (end_time - start_time).total_seconds()
            
            # Get convergence history
            convergence_history = [trial.value for trial in study.trials if trial.value is not None]
            
            # Calculate parameter importance
            try:
                importance = optuna.importance.get_param_importances(study)
            except:
                importance = {}
            
            result = OptimizationResult(
                best_params=best_params,
                best_value=best_value,
                n_trials=len(study.trials),
                optimization_time=optimization_time,
                convergence_history=convergence_history,
                parameter_importance=importance,
                study_metadata={
                    'direction': 'minimize',
                    'n_complete_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                    'n_pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
                },
                timestamp=datetime.now()
            )
            
            self.best_results['lstm'] = result
            self.study = study
            
            logger.info(f"LSTM optimization completed. Best validation loss: {best_value:.6f}")
            logger.info(f"Best parameters: {best_params}")
            
            return result
            
        except Exception as e:
            logger.error(f"LSTM optimization failed: {e}")
            raise
    
    def optimize_ensemble_parameters(self, 
                                   train_data: pd.DataFrame,
                                   target_column: str = 'close') -> OptimizationResult:
        """
        Optimize ensemble model parameters
        """
        try:
            logger.info("Starting ensemble hyperparameter optimization...")
            
            def objective(trial):
                # Define ensemble-specific parameters
                params = {
                    'lstm_weight': trial.suggest_float('lstm_weight', 0.1, 0.8),
                    'rf_n_estimators': trial.suggest_int('rf_n_estimators', 50, 300),
                    'rf_max_depth': trial.suggest_int('rf_max_depth', 3, 20),
                    'xgb_n_estimators': trial.suggest_int('xgb_n_estimators', 50, 300),
                    'xgb_learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
                    'xgb_max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
                    'lgb_n_estimators': trial.suggest_int('lgb_n_estimators', 50, 300),
                    'lgb_learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.3),
                    'lgb_num_leaves': trial.suggest_int('lgb_num_leaves', 10, 100)
                }
                
                try:
                    # Create ensemble with suggested parameters
                    from .deep_learning import EnsembleMLPredictor
                    ensemble = EnsembleMLPredictor()
                    
                    # Update ML model parameters
                    if hasattr(ensemble, 'ml_models'):
                        ensemble.ml_models['random_forest'].set_params(
                            n_estimators=params['rf_n_estimators'],
                            max_depth=params['rf_max_depth']
                        )
                        ensemble.ml_models['xgboost'].set_params(
                            n_estimators=params['xgb_n_estimators'],
                            learning_rate=params['xgb_learning_rate'],
                            max_depth=params['xgb_max_depth']
                        )
                        ensemble.ml_models['lightgbm'].set_params(
                            n_estimators=params['lgb_n_estimators'],
                            learning_rate=params['lgb_learning_rate'],
                            num_leaves=params['lgb_num_leaves']
                        )
                    
                    # Train ensemble with time series split
                    split_idx = int(len(train_data) * 0.8)
                    train_subset = train_data.iloc[:split_idx]
                    val_subset = train_data.iloc[split_idx:]
                    
                    ensemble.train(train_subset, target_column)
                    
                    # Evaluate on validation set
                    val_predictions = []
                    val_actuals = []
                    
                    for i in range(len(val_subset) - 60):  # Need enough history for LSTM
                        try:
                            # Get subset for prediction
                            pred_data = train_data.iloc[:split_idx + i + 60]
                            prediction = ensemble.predict(pred_data)
                            
                            if isinstance(prediction, dict):
                                prediction = prediction['ensemble_prediction']
                            
                            val_predictions.append(prediction)
                            val_actuals.append(val_subset[target_column].iloc[i + 60])
                            
                        except Exception as e:
                            continue
                    
                    if len(val_predictions) > 5:
                        mse = mean_squared_error(val_actuals, val_predictions)
                        return mse
                    else:
                        return float('inf')
                        
                except Exception as e:
                    logger.debug(f"Ensemble trial failed: {e}")
                    return float('inf')
            
            # Create and run study
            study = optuna.create_study(
                direction='minimize',
                sampler=self._get_sampler(self.config.sampler_type),
                study_name=f"ensemble_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            start_time = datetime.now()
            study.optimize(
                objective,
                n_trials=min(self.config.n_trials, 50),  # Limit for ensemble
                timeout=self.config.timeout,
                n_jobs=1
            )
            end_time = datetime.now()
            
            # Process results
            result = self._process_optimization_result(study, start_time, end_time)
            self.best_results['ensemble'] = result
            
            logger.info(f"Ensemble optimization completed. Best MSE: {result.best_value:.6f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Ensemble optimization failed: {e}")
            raise
    
    def optimize_rl_parameters(self, 
                             environment_class=None,
                             agent_class=None) -> OptimizationResult:
        """
        Optimize RL agent hyperparameters
        """
        try:
            logger.info("Starting RL hyperparameter optimization...")
            
            def objective(trial):
                # RL hyperparameter space
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                    'gamma': trial.suggest_float('gamma', 0.9, 0.999),
                    'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096]),
                    'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                    'n_epochs': trial.suggest_int('n_epochs', 3, 15),
                    'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
                    'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0),
                    'ent_coef': trial.suggest_float('ent_coef', 0.001, 0.1, log=True)
                }
                
                try:
                    # Create and train RL agent
                    from .reinforcement_learning import RLConfig, TradingRLAgent
                    
                    config = RLConfig(**params)
                    
                    if agent_class:
                        agent = agent_class(config=config)
                    else:
                        # Use default setup
                        agent = TradingRLAgent(
                            data_manager=None,  # Mock for optimization
                            ta_engine=None,
                            risk_manager=None,
                            config=config
                        )
                    
                    # Quick training for optimization
                    agent.train(total_timesteps=5000)  # Limited for speed
                    
                    # Evaluate performance
                    mean_reward, std_reward = agent.evaluate_performance(n_episodes=3)
                    
                    # Return negative mean reward (since we want to maximize)
                    return -mean_reward
                    
                except Exception as e:
                    logger.debug(f"RL trial failed: {e}")
                    return float('inf')
            
            # Run optimization
            study = optuna.create_study(
                direction='minimize',  # Minimize negative reward
                sampler=self._get_sampler(self.config.sampler_type)
            )
            
            start_time = datetime.now()
            study.optimize(objective, n_trials=20, timeout=1800, n_jobs=1)  # Limited scope
            end_time = datetime.now()
            
            result = self._process_optimization_result(study, start_time, end_time)
            self.best_results['rl'] = result
            
            logger.info(f"RL optimization completed. Best reward: {-result.best_value:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"RL optimization failed: {e}")
            raise
    
    def _get_sampler(self, sampler_type: str):
        """Get Optuna sampler based on type"""
        if sampler_type == 'tpe':
            return TPESampler(n_startup_trials=10, n_ei_candidates=24)
        elif sampler_type == 'cmaes':
            return CmaEsSampler(n_startup_trials=10)
        else:  # random
            return optuna.samplers.RandomSampler()
    
    def _process_optimization_result(self, study, start_time, end_time) -> OptimizationResult:
        """Process Optuna study results"""
        optimization_time = (end_time - start_time).total_seconds()
        convergence_history = [trial.value for trial in study.trials if trial.value is not None]
        
        try:
            importance = optuna.importance.get_param_importances(study)
        except:
            importance = {}
        
        return OptimizationResult(
            best_params=study.best_params,
            best_value=study.best_value,
            n_trials=len(study.trials),
            optimization_time=optimization_time,
            convergence_history=convergence_history,
            parameter_importance=importance,
            study_metadata={
                'direction': study.direction.name,
                'n_complete_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            },
            timestamp=datetime.now()
        )
    
    def get_optimization_summary(self) -> Dict:
        """Get summary of all optimization results"""
        return {
            'completed_optimizations': list(self.best_results.keys()),
            'best_results': {
                name: {
                    'best_value': result.best_value,
                    'best_params': result.best_params,
                    'n_trials': result.n_trials,
                    'optimization_time': result.optimization_time
                }
                for name, result in self.best_results.items()
            },
            'total_optimization_time': sum(
                result.optimization_time for result in self.best_results.values()
            )
        }


class StrategyOptimizer:
    """
    Strategy parameter optimization using genetic algorithms and Bayesian optimization
    """
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.optimization_results = {}
        self.strategy_performance_history = defaultdict(list)
        
        logger.info("Strategy Optimizer initialized")
    
    def optimize_strategy_parameters(self, 
                                   strategy_name: str,
                                   parameter_space: Dict[str, Tuple],
                                   backtest_function: Callable,
                                   optimization_method: str = 'bayesian') -> OptimizationResult:
        """
        Optimize strategy parameters using specified method
        
        Args:
            strategy_name: Name of the strategy
            parameter_space: Dict of parameter names and their (min, max) ranges
            backtest_function: Function that takes parameters and returns performance metric
            optimization_method: 'bayesian', 'genetic', 'grid'
        """
        try:
            logger.info(f"Optimizing {strategy_name} strategy parameters using {optimization_method}")
            
            if optimization_method == 'bayesian':
                result = self._bayesian_optimization(strategy_name, parameter_space, backtest_function)
            elif optimization_method == 'genetic':
                result = self._genetic_optimization(strategy_name, parameter_space, backtest_function)
            elif optimization_method == 'grid':
                result = self._grid_optimization(strategy_name, parameter_space, backtest_function)
            else:
                raise ValueError(f"Unknown optimization method: {optimization_method}")
            
            self.optimization_results[strategy_name] = result
            
            logger.info(f"Strategy optimization completed for {strategy_name}")
            logger.info(f"Best parameters: {result.best_params}")
            logger.info(f"Best performance: {result.best_value}")
            
            return result
            
        except Exception as e:
            logger.error(f"Strategy optimization failed: {e}")
            raise
    
    def _bayesian_optimization(self, 
                             strategy_name: str,
                             parameter_space: Dict[str, Tuple],
                             backtest_function: Callable) -> OptimizationResult:
        """Bayesian optimization using Optuna"""
        
        def objective(trial):
            # Suggest parameters based on their ranges
            params = {}
            for param_name, (min_val, max_val) in parameter_space.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                else:
                    params[param_name] = trial.suggest_float(param_name, min_val, max_val)
            
            try:
                # Run backtest with suggested parameters
                performance = backtest_function(params)
                
                # Store performance for analysis
                self.strategy_performance_history[strategy_name].append({
                    'params': params,
                    'performance': performance,
                    'timestamp': datetime.now()
                })
                
                return performance
                
            except Exception as e:
                logger.warning(f"Backtest failed for {strategy_name}: {e}")
                return float('-inf')  # Return very bad performance
        
        # Create and run study
        study = optuna.create_study(
            direction='maximize',  # Assume we want to maximize performance
            sampler=TPESampler(),
            study_name=f"{strategy_name}_optimization"
        )
        
        start_time = datetime.now()
        study.optimize(objective, n_trials=self.config.n_trials, timeout=self.config.timeout)
        end_time = datetime.now()
        
        return self._process_optimization_result(study, start_time, end_time)
    
    def _genetic_optimization(self, 
                            strategy_name: str,
                            parameter_space: Dict[str, Tuple],
                            backtest_function: Callable) -> OptimizationResult:
        """Genetic algorithm optimization using scipy"""
        
        # Convert parameter space to bounds for scipy
        bounds = list(parameter_space.values())
        param_names = list(parameter_space.keys())
        
        def objective(x):
            # Convert array to parameter dict
            params = {name: val for name, val in zip(param_names, x)}
            
            try:
                performance = backtest_function(params)
                
                # Store performance
                self.strategy_performance_history[strategy_name].append({
                    'params': params,
                    'performance': performance,
                    'timestamp': datetime.now()
                })
                
                return -performance  # Minimize negative performance
                
            except Exception as e:
                logger.warning(f"Genetic optimization backtest failed: {e}")
                return float('inf')
        
        start_time = datetime.now()
        
        # Run differential evolution
        result = differential_evolution(
            objective,
            bounds,
            maxiter=self.config.n_trials // 10,  # Adjust iterations
            popsize=10,
            seed=self.config.random_state,
            workers=1
        )
        
        end_time = datetime.now()
        
        # Convert result to our format
        best_params = {name: val for name, val in zip(param_names, result.x)}
        
        return OptimizationResult(
            best_params=best_params,
            best_value=-result.fun,  # Convert back to positive
            n_trials=result.nfev,
            optimization_time=(end_time - start_time).total_seconds(),
            convergence_history=[],  # Not available from scipy
            parameter_importance={},  # Not available from scipy
            study_metadata={'success': result.success, 'message': result.message},
            timestamp=datetime.now()
        )
    
    def _grid_optimization(self, 
                         strategy_name: str,
                         parameter_space: Dict[str, Tuple],
                         backtest_function: Callable) -> OptimizationResult:
        """Grid search optimization"""
        
        # Create parameter grids
        param_grids = {}
        for param_name, (min_val, max_val) in parameter_space.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                # Integer parameters
                param_grids[param_name] = np.linspace(min_val, max_val, 10, dtype=int)
            else:
                # Float parameters
                param_grids[param_name] = np.linspace(min_val, max_val, 10)
        
        best_params = None
        best_performance = float('-inf')
        n_trials = 0
        
        start_time = datetime.now()
        
        # Generate all combinations
        param_names = list(param_grids.keys())
        param_values = list(param_grids.values())
        
        # Use itertools.product to get all combinations
        import itertools
        
        for param_combination in itertools.product(*param_values):
            params = {name: val for name, val in zip(param_names, param_combination)}
            
            try:
                performance = backtest_function(params)
                n_trials += 1
                
                # Store performance
                self.strategy_performance_history[strategy_name].append({
                    'params': params,
                    'performance': performance,
                    'timestamp': datetime.now()
                })
                
                if performance > best_performance:
                    best_performance = performance
                    best_params = params.copy()
                    
            except Exception as e:
                logger.warning(f"Grid search backtest failed: {e}")
                continue
            
            # Break if we exceed trial limit
            if n_trials >= self.config.n_trials:
                break
        
        end_time = datetime.now()
        
        return OptimizationResult(
            best_params=best_params or {},
            best_value=best_performance,
            n_trials=n_trials,
            optimization_time=(end_time - start_time).total_seconds(),
            convergence_history=[],
            parameter_importance={},
            study_metadata={'method': 'grid_search'},
            timestamp=datetime.now()
        )
    
    def analyze_parameter_sensitivity(self, strategy_name: str) -> Dict:
        """Analyze parameter sensitivity from optimization history"""
        try:
            if strategy_name not in self.strategy_performance_history:
                return {'error': 'No optimization history available'}
            
            history = self.strategy_performance_history[strategy_name]
            
            if len(history) < 10:
                return {'error': 'Insufficient data for sensitivity analysis'}
            
            # Extract parameters and performances
            param_names = list(history[0]['params'].keys())
            sensitivity_analysis = {}
            
            for param_name in param_names:
                param_values = [entry['params'][param_name] for entry in history]
                performances = [entry['performance'] for entry in history]
                
                # Calculate correlation between parameter and performance
                correlation = np.corrcoef(param_values, performances)[0, 1]
                
                # Calculate parameter range impact
                sorted_indices = np.argsort(param_values)
                sorted_performances = np.array(performances)[sorted_indices]
                
                # Compare top and bottom quartiles
                q25_idx = len(sorted_performances) // 4
                q75_idx = 3 * len(sorted_performances) // 4
                
                bottom_quartile_perf = np.mean(sorted_performances[:q25_idx])
                top_quartile_perf = np.mean(sorted_performances[q75_idx:])
                
                sensitivity_analysis[param_name] = {
                    'correlation': correlation if not np.isnan(correlation) else 0.0,
                    'quartile_difference': top_quartile_perf - bottom_quartile_perf,
                    'parameter_range': (min(param_values), max(param_values)),
                    'optimal_range': self._find_optimal_parameter_range(param_values, performances)
                }
            
            return sensitivity_analysis
            
        except Exception as e:
            logger.error(f"Parameter sensitivity analysis failed: {e}")
            return {'error': str(e)}
    
    def _find_optimal_parameter_range(self, param_values: List, performances: List) -> Tuple:
        """Find optimal parameter range based on top-performing trials"""
        try:
            # Get top 20% performing trials
            n_top = max(1, len(performances) // 5)
            top_indices = np.argsort(performances)[-n_top:]
            
            top_param_values = [param_values[i] for i in top_indices]
            
            return (min(top_param_values), max(top_param_values))
            
        except:
            return (min(param_values), max(param_values))
    
    def get_strategy_optimization_summary(self) -> Dict:
        """Get summary of all strategy optimizations"""
        return {
            'optimized_strategies': list(self.optimization_results.keys()),
            'optimization_results': {
                name: {
                    'best_value': result.best_value,
                    'best_params': result.best_params,
                    'n_trials': result.n_trials
                }
                for name, result in self.optimization_results.items()
            },
            'performance_history_size': {
                name: len(history) for name, history in self.strategy_performance_history.items()
            }
        }


class PortfolioOptimizer:
    """
    Advanced portfolio optimization using Modern Portfolio Theory and alternatives
    """
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.optimization_results = {}
        
        if not PORTFOLIO_OPT_AVAILABLE:
            logger.warning("Portfolio optimization libraries not available. Install cvxpy and pypfopt.")
        
        logger.info("Portfolio Optimizer initialized")
    
    def optimize_mean_variance_portfolio(self, 
                                       returns_data: pd.DataFrame,
                                       target_return: Optional[float] = None,
                                       risk_free_rate: float = 0.02) -> Dict:
        """
        Optimize portfolio using mean-variance optimization
        """
        try:
            if not PORTFOLIO_OPT_AVAILABLE:
                return {'error': 'Portfolio optimization libraries not available'}
            
            logger.info("Running mean-variance portfolio optimization")
            
            # Calculate expected returns and covariance matrix
            mu = expected_returns.mean_historical_return(returns_data)
            S = risk_models.sample_cov(returns_data)
            
            # Create efficient frontier
            ef = EfficientFrontier(mu, S)
            
            if target_return:
                # Optimize for specific return target
                weights = ef.efficient_return(target_return)
            else:
                # Optimize for maximum Sharpe ratio
                weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
            
            # Get portfolio performance
            performance = ef.portfolio_performance(risk_free_rate=risk_free_rate)
            
            # Clean weights (remove tiny allocations)
            cleaned_weights = ef.clean_weights()
            
            result = {
                'weights': dict(cleaned_weights),
                'expected_return': performance[0],
                'volatility': performance[1],
                'sharpe_ratio': performance[2],
                'optimization_method': 'mean_variance',
                'target_return': target_return,
                'risk_free_rate': risk_free_rate
            }
            
            self.optimization_results['mean_variance'] = result
            
            logger.info(f"Portfolio optimization completed. Sharpe ratio: {performance[2]:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Mean-variance optimization failed: {e}")
            return {'error': str(e)}
    
    def optimize_risk_parity_portfolio(self, returns_data: pd.DataFrame) -> Dict:
        """
        Optimize portfolio using risk parity approach
        """
        try:
            logger.info("Running risk parity portfolio optimization")
            
            # Calculate covariance matrix
            cov_matrix = returns_data.cov()
            n_assets = len(cov_matrix)
            
            # Risk parity objective function
            def risk_parity_objective(weights):
                weights = np.array(weights)
                portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
                
                # Calculate marginal contributions to risk
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_var
                
                # Risk contributions
                risk_contrib = weights * marginal_contrib
                
                # Target is equal risk contribution (1/n for each asset)
                target_contrib = np.ones(n_assets) / n_assets
                
                # Minimize sum of squared deviations from target
                return np.sum((risk_contrib - target_contrib) ** 2)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            ]
            
            # Bounds (no short selling)
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Initial guess (equal weights)
            x0 = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(
                risk_parity_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimal_weights = dict(zip(returns_data.columns, result.x))
                
                # Calculate portfolio metrics
                portfolio_return = np.dot(result.x, returns_data.mean()) * 252  # Annualized
                portfolio_vol = np.sqrt(np.dot(result.x.T, np.dot(cov_matrix * 252, result.x)))
                
                optimization_result = {
                    'weights': optimal_weights,
                    'expected_return': portfolio_return,
                    'volatility': portfolio_vol,
                    'sharpe_ratio': portfolio_return / portfolio_vol,
                    'optimization_method': 'risk_parity',
                    'success': True
                }
            else:
                optimization_result = {
                    'error': 'Risk parity optimization failed to converge',
                    'success': False
                }
            
            self.optimization_results['risk_parity'] = optimization_result
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Risk parity optimization failed: {e}")
            return {'error': str(e)}
    
    def optimize_black_litterman_portfolio(self, 
                                         returns_data: pd.DataFrame,
                                         market_caps: Optional[Dict] = None,
                                         views: Optional[Dict] = None,
                                         view_confidences: Optional[Dict] = None) -> Dict:
        """
        Optimize portfolio using Black-Litterman model
        """
        try:
            if not PORTFOLIO_OPT_AVAILABLE:
                return {'error': 'Portfolio optimization libraries not available'}
            
            logger.info("Running Black-Litterman portfolio optimization")
            
            # If no market caps provided, use equal weighting as prior
            if market_caps is None:
                market_caps = {asset: 1.0 for asset in returns_data.columns}
            
            # Calculate sample covariance
            S = risk_models.sample_cov(returns_data)
            
            # Market capitalization weights (prior)
            total_market_cap = sum(market_caps.values())
            market_weights = {asset: cap / total_market_cap for asset, cap in market_caps.items()}
            
            # Implied returns (equilibrium returns)
            risk_aversion = 2.5  # Typical value
            pi = risk_aversion * np.dot(S, list(market_weights.values()))
            
            # Black-Litterman adjustment
            if views and view_confidences:
                # This is a simplified implementation
                # In practice, you'd use the full Black-Litterman formula
                mu_bl = pi  # Simplified: just use implied returns
            else:
                mu_bl = pi
            
            # Optimize portfolio
            ef = EfficientFrontier(mu_bl, S)
            weights = ef.max_sharpe()
            performance = ef.portfolio_performance()
            
            result = {
                'weights': dict(ef.clean_weights()),
                'expected_return': performance[0],
                'volatility': performance[1],
                'sharpe_ratio': performance[2],
                'optimization_method': 'black_litterman',
                'market_weights': market_weights,
                'implied_returns': dict(zip(returns_data.columns, pi))
            }
            
            self.optimization_results['black_litterman'] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Black-Litterman optimization failed: {e}")
            return {'error': str(e)}
    
    def optimize_cvar_portfolio(self, returns_data: pd.DataFrame, alpha: float = 0.05) -> Dict:
        """
        Optimize portfolio using Conditional Value at Risk (CVaR)
        """
        try:
            if not PORTFOLIO_OPT_AVAILABLE:
                return {'error': 'Portfolio optimization libraries not available'}
            
            logger.info(f"Running CVaR portfolio optimization (alpha={alpha})")
            
            # Convert returns to numpy array
            returns_array = returns_data.values
            n_assets = returns_array.shape[1]
            n_scenarios = returns_array.shape[0]
            
            # Define optimization variables
            w = cp.Variable(n_assets)  # Portfolio weights
            z = cp.Variable(n_scenarios)  # Auxiliary variables for CVaR
            var = cp.Variable()  # Value at Risk
            
            # Portfolio returns for each scenario
            portfolio_returns = returns_array @ w
            
            # CVaR constraints
            constraints = [
                cp.sum(w) == 1,  # Weights sum to 1
                w >= 0,  # No short selling
                z >= 0,  # Auxiliary variables non-negative
                z >= -(portfolio_returns + var)  # CVaR constraint
            ]
            
            # Objective: Minimize CVaR
            cvar = var + (1 / (alpha * n_scenarios)) * cp.sum(z)
            objective = cp.Minimize(cvar)
            
            # Solve
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status == cp.OPTIMAL:
                optimal_weights = dict(zip(returns_data.columns, w.value))
                
                # Calculate portfolio metrics
                portfolio_returns_opt = np.dot(returns_array, w.value)
                expected_return = np.mean(portfolio_returns_opt) * 252
                volatility = np.std(portfolio_returns_opt) * np.sqrt(252)
                
                # Calculate actual CVaR
                var_value = np.percentile(portfolio_returns_opt, alpha * 100)
                cvar_value = np.mean(portfolio_returns_opt[portfolio_returns_opt <= var_value])
                
                result = {
                    'weights': optimal_weights,
                    'expected_return': expected_return,
                    'volatility': volatility,
                    'var': var_value,
                    'cvar': cvar_value,
                    'optimization_method': 'cvar',
                    'alpha': alpha,
                    'success': True
                }
            else:
                result = {
                    'error': f'CVaR optimization failed: {problem.status}',
                    'success': False
                }
            
            self.optimization_results['cvar'] = result
            
            return result
            
        except Exception as e:
            logger.error(f"CVaR optimization failed: {e}")
            return {'error': str(e)}
    
    def compare_optimization_methods(self, returns_data: pd.DataFrame) -> Dict:
        """
        Compare multiple portfolio optimization methods
        """
        try:
            logger.info("Comparing portfolio optimization methods")
            
            comparison_results = {}
            
            # Run different optimization methods
            methods = [
                ('mean_variance', lambda: self.optimize_mean_variance_portfolio(returns_data)),
                ('risk_parity', lambda: self.optimize_risk_parity_portfolio(returns_data)),
                ('cvar', lambda: self.optimize_cvar_portfolio(returns_data))
            ]
            
            for method_name, method_func in methods:
                try:
                    result = method_func()
                    if 'error' not in result:
                        comparison_results[method_name] = result
                except Exception as e:
                    logger.warning(f"Method {method_name} failed: {e}")
            
            # Rank methods by Sharpe ratio
            if comparison_results:
                sharpe_rankings = sorted(
                    [(name, result.get('sharpe_ratio', 0)) for name, result in comparison_results.items()],
                    key=lambda x: x[1],
                    reverse=True
                )
                
                return {
                    'comparison_results': comparison_results,
                    'sharpe_rankings': sharpe_rankings,
                    'best_method': sharpe_rankings[0][0] if sharpe_rankings else None
                }
            else:
                return {'error': 'No optimization methods succeeded'}
                
        except Exception as e:
            logger.error(f"Portfolio comparison failed: {e}")
            return {'error': str(e)}
    
    def get_portfolio_optimization_summary(self) -> Dict:
        """Get summary of portfolio optimizations"""
        return {
            'available_methods': ['mean_variance', 'risk_parity', 'black_litterman', 'cvar'],
            'completed_optimizations': list(self.optimization_results.keys()),
            'optimization_results': self.optimization_results,
            'portfolio_opt_available': PORTFOLIO_OPT_AVAILABLE
        }