from fastapi import FastAPI, APIRouter
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List
import uuid
from datetime import datetime

# Import FinGPT Trading System
from trading_system.trading_engine import TradingEngine
from trading_system.storage import MongoDBManager
from trading_system.api_routes import router as trading_router, set_trading_engine


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(
    title="FinGPT Trading System",
    description="AI-Powered Trading System with Real-time Data Feeds",
    version="1.0.0"
)

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Define Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "FinGPT Trading System API - Ready for Market Analysis"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

# Include the routers in the main app
app.include_router(api_router)
app.include_router(trading_router)  # Add trading system routes

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global trading engine instance
trading_engine_instance = None

@app.on_event("startup")
async def startup_event():
    """Initialize the trading engine on startup"""
    global trading_engine_instance
    
    try:
        logger.info("Initializing FinGPT Trading System...")
        
        # Create MongoDB manager
        mongo_manager = MongoDBManager(client, os.environ['DB_NAME'])
        
        # Create trading engine
        trading_engine_instance = TradingEngine(mongo_manager)
        
        # Set the trading engine for API routes
        set_trading_engine(trading_engine_instance)
        
        logger.info("FinGPT Trading System initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize trading system: {e}")

@app.on_event("shutdown")
async def shutdown_db_client():
    """Cleanup on shutdown"""
    global trading_engine_instance
    
    try:
        if trading_engine_instance and trading_engine_instance.is_running:
            await trading_engine_instance.stop_engine()
        
        client.close()
        logger.info("FinGPT Trading System shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Add system information endpoint
@api_router.get("/system/info")
async def get_system_info():
    """Get system information and capabilities"""
    return {
        "system": "FinGPT Trading System",
        "version": "1.0.0",
        "description": "AI-Powered Trading System optimized for RTX 4050 (6GB VRAM)",
        "features": [
            "Real-time market data from MEXC, Binance, Yahoo Finance",
            "Advanced technical analysis with 15+ indicators",
            "Multi-strategy trading engine (Momentum, Mean Reversion, Breakout)",
            "Comprehensive risk management system",
            "Paper trading with performance tracking",
            "Memory-efficient data storage with compression",
            "RESTful API for system integration"
        ],
        "data_sources": {
            "crypto": ["MEXC WebSocket", "Binance Public API"],
            "stocks": ["Yahoo Finance"],
            "forex": ["Yahoo Finance"]
        },
        "supported_intervals": ["1m", "5m", "15m", "1h", "4h", "1d"],
        "risk_management": {
            "position_sizing": "Kelly Criterion + Fixed Fractional",
            "stop_loss": "ATR-based dynamic stops",
            "portfolio_limits": "Max 10% drawdown, 5% position size",
            "daily_limits": "2% daily loss limit"
        },
        "hardware_optimized": {
            "target_gpu": "RTX 4050 (6GB VRAM)",
            "storage_limit": "526GB with compression",
            "memory_efficient": "Quantized models, compressed storage"
        }
    }
