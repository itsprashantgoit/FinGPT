"""
High-Performance Configuration for FinGPT Trading System
Optimized for 16-core ARM Neoverse-N1, 62GB RAM, 116GB Storage
"""

import os
from typing import Dict, Any

class PerformanceConfig:
    """Optimized performance configuration for cloud infrastructure"""
    
    # Hardware-optimized settings
    SYSTEM_SPECS = {
        "cpu_cores": 16,
        "memory_gb": 62,
        "storage_gb": 116,
        "architecture": "ARM64",
        "cpu_model": "Neoverse-N1"
    }
    
    # Trading Engine Configuration
    TRADING_ENGINE = {
        "max_concurrent_symbols": 50,  # Increased from 10 (more CPU cores)
        "parallel_analysis_workers": 12,  # Use 75% of CPU cores
        "data_processing_threads": 8,
        "strategy_evaluation_workers": 6,
        "risk_calculation_threads": 4,
    }
    
    # Data Management Optimization
    DATA_CONFIG = {
        "memory_cache_size_mb": 8192,  # 8GB cache (vs 512MB RTX limit)
        "batch_processing_size": 10000,  # Larger batches
        "concurrent_data_feeds": 20,  # Multiple simultaneous feeds
        "historical_data_retention_days": 365,  # More storage available
        "compression_enabled": True,
        "parallel_database_connections": 16,
    }
    
    # Technical Analysis Enhancement
    TECHNICAL_ANALYSIS = {
        "indicators_per_symbol": 25,  # Increased from 15
        "lookback_periods": {
            "short_term": 50,
            "medium_term": 200, 
            "long_term": 500
        },
        "parallel_indicator_calculation": True,
        "advanced_pattern_recognition": True,
        "ml_prediction_models": True,
    }
    
    # Portfolio Management Scaling
    PORTFOLIO_CONFIG = {
        "max_portfolios_per_user": 20,  # Increased capacity
        "max_positions_per_portfolio": 100,
        "real_time_pnl_calculation": True,
        "advanced_risk_metrics": True,
        "portfolio_optimization_frequency": "1min",  # Real-time
    }
    
    # Strategy Configuration
    STRATEGY_CONFIG = {
        "simultaneous_strategies": 10,  # Multiple strategies per symbol
        "machine_learning_features": {
            "sentiment_analysis": True,
            "news_integration": True,
            "social_media_signals": True,
            "options_flow_analysis": True,
        },
        "advanced_algorithms": {
            "reinforcement_learning": True,
            "ensemble_methods": True,
            "adaptive_parameters": True,
        }
    }
    
    # Risk Management Enhancement
    RISK_MANAGEMENT = {
        "real_time_var_calculation": True,
        "stress_testing_scenarios": 50,
        "correlation_analysis_depth": 500,  # More symbols
        "risk_monitoring_frequency": "30s",
        "advanced_position_sizing": True,
        "portfolio_heat_map": True,
    }
    
    # Performance Monitoring
    MONITORING_CONFIG = {
        "performance_metrics_retention_days": 90,
        "detailed_execution_analysis": True,
        "latency_optimization": True,
        "throughput_monitoring": True,
        "memory_usage_optimization": True,
    }
    
    @classmethod
    def get_optimized_config(cls) -> Dict[str, Any]:
        """Get complete optimized configuration"""
        return {
            "system": cls.SYSTEM_SPECS,
            "trading_engine": cls.TRADING_ENGINE,
            "data": cls.DATA_CONFIG,
            "technical_analysis": cls.TECHNICAL_ANALYSIS,
            "portfolio": cls.PORTFOLIO_CONFIG,
            "strategy": cls.STRATEGY_CONFIG,
            "risk_management": cls.RISK_MANAGEMENT,
            "monitoring": cls.MONITORING_CONFIG,
        }
    
    @classmethod
    def get_worker_counts(cls) -> Dict[str, int]:
        """Get optimized worker thread counts"""
        return {
            "data_workers": cls.TRADING_ENGINE["parallel_analysis_workers"],
            "strategy_workers": cls.TRADING_ENGINE["strategy_evaluation_workers"],
            "risk_workers": cls.TRADING_ENGINE["risk_calculation_threads"],
            "db_connections": cls.DATA_CONFIG["parallel_database_connections"],
        }

# Environment-specific optimizations
def get_environment_config():
    """Get environment-specific configuration overrides"""
    env = os.getenv("TRADING_ENV", "production")
    
    base_config = PerformanceConfig.get_optimized_config()
    
    if env == "production":
        # Production optimizations
        base_config["trading_engine"]["max_concurrent_symbols"] = 100
        base_config["data"]["memory_cache_size_mb"] = 12288  # 12GB cache
        base_config["monitoring"]["latency_optimization"] = True
        
    elif env == "development":
        # Development settings (slightly reduced)
        base_config["trading_engine"]["max_concurrent_symbols"] = 25
        base_config["data"]["memory_cache_size_mb"] = 4096  # 4GB cache
    
    return base_config