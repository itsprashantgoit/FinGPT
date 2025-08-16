"""
Memory-efficient data storage optimized for limited resources
Compressed storage with automatic cleanup and space management
"""
import gzip
import pickle
import sqlite3
import asyncio
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from motor.motor_asyncio import AsyncIOMotorClient
from .data_models import KlineData, TradingSignal, TechnicalIndicators, COLLECTIONS

logger = logging.getLogger(__name__)


class MemoryEfficientDataStorage:
    """
    Optimized data storage for limited disk space (526GB total)
    Uses compression and intelligent data retention
    """
    
    def __init__(self, base_path: str = "./trading_data", max_storage_gb: int = 50):
        self.base_path = Path(base_path)
        self.max_storage_gb = max_storage_gb
        self.compression_level = 9  # Maximum compression
        
        # Create directories
        self.base_path.mkdir(parents=True, exist_ok=True)
        (self.base_path / "klines").mkdir(exist_ok=True)
        (self.base_path / "models").mkdir(exist_ok=True)
        (self.base_path / "backups").mkdir(exist_ok=True)
        
        # Initialize SQLite database for metadata
        self.db_path = self.base_path / "metadata.db"
        self._init_database()
        
        # Data retention limits (to save space)
        self.retention_limits = {
            '1m': 30 * 24 * 60,      # 30 days of 1-minute data
            '5m': 60 * 24 * 12,      # 60 days of 5-minute data
            '15m': 90 * 24 * 4,      # 90 days of 15-minute data
            '1h': 180 * 24,          # 180 days of hourly data
            '4h': 365 * 6,           # 365 days of 4-hour data
            '1d': 365 * 5            # 5 years of daily data
        }
    
    def _init_database(self):
        """Initialize SQLite database for metadata tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS kline_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exchange TEXT,
                    symbol TEXT,
                    interval TEXT,
                    start_time INTEGER,
                    end_time INTEGER,
                    record_count INTEGER,
                    file_size_mb REAL,
                    compression_ratio REAL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(exchange, symbol, interval)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS storage_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_files INTEGER,
                    total_size_mb REAL,
                    compressed_size_mb REAL,
                    compression_savings REAL,
                    last_cleanup TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_symbol_interval 
                ON kline_metadata(symbol, interval)
            ''')
    
    def store_kline_data(self, klines: List[KlineData]) -> bool:
        """Store kline data with compression and deduplication"""
        if not klines:
            return False
        
        # Group by exchange, symbol, interval
        grouped_data = defaultdict(list)
        for kline in klines:
            key = (kline.exchange, kline.symbol, kline.interval)
            grouped_data[key].append(kline.to_dict())
        
        success_count = 0
        for (exchange, symbol, interval), data in grouped_data.items():
            try:
                # Create safe file path
                safe_symbol = symbol.replace('/', '_').replace('-', '_')
                file_path = self.base_path / "klines" / f"{exchange}_{safe_symbol}_{interval}.pkl.gz"
                
                # Load existing data
                existing_data = []
                original_size = 0
                if file_path.exists():
                    try:
                        original_size = file_path.stat().st_size
                        with gzip.open(file_path, 'rb') as f:
                            existing_data = pickle.load(f)
                    except Exception as e:
                        logger.warning(f"Could not load existing data from {file_path}: {e}")
                
                # Combine and sort data by timestamp
                combined_data = existing_data + data
                combined_data.sort(key=lambda x: x['timestamp'])
                
                # Remove duplicates based on timestamp
                seen_timestamps = set()
                unique_data = []
                for item in combined_data:
                    timestamp = item['timestamp']
                    if timestamp not in seen_timestamps:
                        unique_data.append(item)
                        seen_timestamps.add(timestamp)
                
                # Apply retention limits to save space
                limit = self.retention_limits.get(interval, 10000)
                if len(unique_data) > limit:
                    unique_data = unique_data[-limit:]
                    logger.info(f"Applied retention limit for {symbol} {interval}: kept {len(unique_data)} records")
                
                # Calculate uncompressed size for metrics
                uncompressed_size = len(pickle.dumps(unique_data))
                
                # Save with maximum compression
                with gzip.open(file_path, 'wb', compresslevel=self.compression_level) as f:
                    pickle.dump(unique_data, f)
                
                # Calculate compression metrics
                compressed_size = file_path.stat().st_size
                compression_ratio = uncompressed_size / compressed_size if compressed_size > 0 else 1.0
                file_size_mb = compressed_size / (1024 * 1024)
                
                # Update metadata
                self._update_metadata(exchange, symbol, interval, unique_data, file_size_mb, compression_ratio)
                
                success_count += 1
                logger.info(f"Stored {len(unique_data)} klines for {exchange} {symbol} {interval} "
                          f"(compression: {compression_ratio:.1f}x, size: {file_size_mb:.2f}MB)")
                
            except Exception as e:
                logger.error(f"Error storing data for {exchange} {symbol} {interval}: {e}")
        
        # Check storage usage periodically
        if success_count > 0:
            self._check_storage_usage()
        
        return success_count > 0
    
    def _update_metadata(self, exchange: str, symbol: str, interval: str, 
                        data: List[Dict], file_size_mb: float, compression_ratio: float):
        """Update metadata in SQLite database"""
        if not data:
            return
        
        start_time = min(item['timestamp'] for item in data)
        end_time = max(item['timestamp'] for item in data)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO kline_metadata 
                (exchange, symbol, interval, start_time, end_time, record_count, 
                 file_size_mb, compression_ratio, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (exchange, symbol, interval, start_time, end_time, 
                  len(data), file_size_mb, compression_ratio))
    
    def _check_storage_usage(self):
        """Monitor and manage storage usage"""
        try:
            # Calculate total storage usage
            total_size = 0
            file_count = 0
            
            for file_path in self.base_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            total_size_gb = total_size / (1024**3)
            total_size_mb = total_size / (1024**2)
            
            logger.info(f"Storage usage: {total_size_gb:.2f} GB ({file_count} files)")
            
            # Update storage statistics
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO storage_stats 
                    (total_files, total_size_mb, compressed_size_mb, last_cleanup)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ''', (file_count, total_size_mb, total_size_mb))
            
            # Trigger cleanup if approaching limit
            if total_size_gb > self.max_storage_gb * 0.85:  # 85% threshold
                logger.warning(f"Storage usage high ({total_size_gb:.2f} GB), triggering cleanup")
                self._cleanup_old_data()
                
        except Exception as e:
            logger.error(f"Error checking storage usage: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old data to free space"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Find oldest and largest files to remove
                cursor = conn.execute('''
                    SELECT exchange, symbol, interval, file_size_mb, last_updated
                    FROM kline_metadata 
                    ORDER BY last_updated ASC 
                    LIMIT 20
                ''')
                
                space_freed = 0
                files_removed = 0
                
                for row in cursor.fetchall():
                    exchange, symbol, interval, file_size_mb, last_updated = row
                    safe_symbol = symbol.replace('/', '_').replace('-', '_')
                    file_path = self.base_path / "klines" / f"{exchange}_{safe_symbol}_{interval}.pkl.gz"
                    
                    if file_path.exists():
                        file_path.unlink()
                        space_freed += file_size_mb
                        files_removed += 1
                        
                        # Remove from metadata
                        conn.execute('''
                            DELETE FROM kline_metadata 
                            WHERE exchange=? AND symbol=? AND interval=?
                        ''', (exchange, symbol, interval))
                        
                        logger.info(f"Cleaned up {file_path.name} ({file_size_mb:.2f} MB)")
                        
                        # Stop if we've freed enough space
                        if space_freed > 1024:  # 1GB freed
                            break
                
                logger.info(f"Cleanup complete: removed {files_removed} files, freed {space_freed:.2f} MB")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def load_kline_data(self, exchange: str, symbol: str, interval: str, 
                       limit: Optional[int] = None) -> List[KlineData]:
        """Load kline data from compressed storage"""
        safe_symbol = symbol.replace('/', '_').replace('-', '_')
        file_path = self.base_path / "klines" / f"{exchange}_{safe_symbol}_{interval}.pkl.gz"
        
        if not file_path.exists():
            logger.warning(f"No data file found: {file_path}")
            return []
        
        try:
            with gzip.open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Apply limit if specified
            if limit and len(data) > limit:
                data = data[-limit:]
            
            # Convert back to KlineData objects
            klines = []
            for item in data:
                # Ensure we have the exchange field
                if 'exchange' not in item:
                    item['exchange'] = exchange
                
                kline = KlineData(**item)
                klines.append(kline)
            
            logger.info(f"Loaded {len(klines)} klines for {exchange} {symbol} {interval}")
            return klines
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return []
    
    def get_storage_stats(self) -> Dict:
        """Get comprehensive storage statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get overall stats
                cursor = conn.execute('''
                    SELECT 
                        COUNT(*) as total_symbols,
                        SUM(record_count) as total_records,
                        SUM(file_size_mb) as total_size_mb,
                        AVG(compression_ratio) as avg_compression
                    FROM kline_metadata
                ''')
                
                stats = cursor.fetchone()
                
                # Get breakdown by exchange
                cursor = conn.execute('''
                    SELECT exchange, COUNT(*), SUM(file_size_mb)
                    FROM kline_metadata 
                    GROUP BY exchange
                ''')
                
                exchange_stats = cursor.fetchall()
                
                # Calculate total disk usage
                total_disk_usage = sum(f.stat().st_size for f in self.base_path.rglob("*") if f.is_file())
                total_disk_gb = total_disk_usage / (1024**3)
                
                return {
                    "total_symbols": stats[0] if stats[0] else 0,
                    "total_records": stats[1] if stats[1] else 0,
                    "total_size_mb": stats[2] if stats[2] else 0.0,
                    "avg_compression_ratio": stats[3] if stats[3] else 1.0,
                    "total_disk_usage_gb": total_disk_gb,
                    "max_storage_gb": self.max_storage_gb,
                    "usage_percentage": (total_disk_gb / self.max_storage_gb) * 100,
                    "exchange_breakdown": [
                        {"exchange": row[0], "symbols": row[1], "size_mb": row[2]}
                        for row in exchange_stats
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {}
    
    def cleanup_by_age(self, days_old: int = 90):
        """Clean up data older than specified days"""
        cutoff_timestamp = int((datetime.utcnow() - timedelta(days=days_old)).timestamp() * 1000)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT exchange, symbol, interval 
                    FROM kline_metadata 
                    WHERE end_time < ?
                ''', (cutoff_timestamp,))
                
                for row in cursor.fetchall():
                    exchange, symbol, interval = row
                    safe_symbol = symbol.replace('/', '_').replace('-', '_')
                    file_path = self.base_path / "klines" / f"{exchange}_{safe_symbol}_{interval}.pkl.gz"
                    
                    if file_path.exists():
                        file_size = file_path.stat().st_size / (1024 * 1024)
                        file_path.unlink()
                        
                        conn.execute('''
                            DELETE FROM kline_metadata 
                            WHERE exchange=? AND symbol=? AND interval=?
                        ''', (exchange, symbol, interval))
                        
                        logger.info(f"Removed old data: {file_path.name} ({file_size:.2f} MB)")
                        
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")


class MongoDBManager:
    """
    MongoDB integration for trading system data
    Handles real-time data, signals, and performance metrics
    """
    
    def __init__(self, mongo_client: AsyncIOMotorClient, db_name: str):
        self.client = mongo_client
        self.db = self.client[db_name]
    
    async def store_kline(self, kline: KlineData) -> bool:
        """Store a single kline data point"""
        try:
            collection = self.db[COLLECTIONS["klines"]]
            
            # Create index for efficient queries
            await collection.create_index([
                ("symbol", 1),
                ("interval", 1),
                ("timestamp", 1)
            ], unique=True, background=True)
            
            # Insert or update
            result = await collection.replace_one(
                {
                    "symbol": kline.symbol,
                    "interval": kline.interval,
                    "timestamp": kline.timestamp
                },
                kline.to_dict(),
                upsert=True
            )
            
            return result.acknowledged
            
        except Exception as e:
            logger.error(f"Error storing kline data: {e}")
            return False
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 100) -> List[Dict]:
        """Retrieve kline data from MongoDB"""
        try:
            collection = self.db[COLLECTIONS["klines"]]
            
            cursor = collection.find({
                "symbol": symbol,
                "interval": interval
            }).sort("timestamp", -1).limit(limit)
            
            return await cursor.to_list(length=limit)
            
        except Exception as e:
            logger.error(f"Error retrieving klines: {e}")
            return []
    
    async def store_trading_signal(self, signal: TradingSignal) -> bool:
        """Store trading signal"""
        try:
            collection = self.db[COLLECTIONS["signals"]]
            
            await collection.create_index([
                ("symbol", 1),
                ("created_at", -1)
            ], background=True)
            
            result = await collection.insert_one(signal.dict())
            return result.acknowledged
            
        except Exception as e:
            logger.error(f"Error storing trading signal: {e}")
            return False
    
    async def get_latest_signals(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get latest trading signals"""
        try:
            collection = self.db[COLLECTIONS["signals"]]
            
            filter_query = {}
            if symbol:
                filter_query["symbol"] = symbol
            
            cursor = collection.find(filter_query).sort("created_at", -1).limit(limit)
            return await cursor.to_list(length=limit)
            
        except Exception as e:
            logger.error(f"Error retrieving trading signals: {e}")
            return []