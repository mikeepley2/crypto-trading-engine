#!/usr/bin/env python3
"""
Redis Signal Publisher for Real-Time Signal Streaming
Publishes trading signals to Redis pub/sub for instant delivery
"""

import os
import redis
import json
import logging
import asyncio
import mysql.connector
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [REDIS_PUBLISHER] %(message)s'
)
logger = logging.getLogger(__name__)

class RedisSignalPublisher:
    """Real-time signal publisher using Redis pub-sub"""
    
    def __init__(self):
        # Redis configuration
        self.redis_host = os.environ.get('REDIS_HOST', 'redis')
        self.redis_port = int(os.environ.get('REDIS_PORT', 6379))
        self.redis_db = int(os.environ.get('REDIS_DB', 0))
        
        # Database configuration
        self.db_config = {
            'host': os.environ.get('DATABASE_HOST', 'host.docker.internal'),
            'user': os.environ.get('DATABASE_USER', 'news_collector'),
            'password': os.environ.get('DATABASE_PASSWORD', '99Rules!'),
            'database': os.environ.get('DATABASE_NAME', 'crypto_prices'),
            'port': int(os.environ.get('DATABASE_PORT', 3306))
        }
        
        # Redis connection
        self.redis_client = None
        self.last_signal_timestamp = None
        
        # Metrics
        self.signals_published = 0
        self.publish_errors = 0
        
    def connect_redis(self) -> bool:
        """Connect to Redis server"""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"✅ Connected to Redis: {self.redis_host}:{self.redis_port}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            return False
    
    def get_new_signals(self) -> List[Dict]:
        """Get new trading signals from database"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor(dictionary=True)
            
            # Get signals newer than last processed
            if self.last_signal_timestamp:
                query = """
                SELECT id, symbol, signal_type, confidence, price, created_at, 
                       model_version, features_used, xgboost_confidence, signal_strength
                FROM trading_signals 
                WHERE created_at > %s
                AND confidence >= 0.6
                AND signal_type IN ('BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL')
                ORDER BY created_at ASC
                """
                cursor.execute(query, (self.last_signal_timestamp,))
            else:
                # First run - get signals from last 5 minutes
                query = """
                SELECT id, symbol, signal_type, confidence, price, created_at,
                       model_version, features_used, xgboost_confidence, signal_strength
                FROM trading_signals 
                WHERE created_at >= DATE_SUB(NOW(), INTERVAL 5 MINUTE)
                AND confidence >= 0.6
                AND signal_type IN ('BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL')
                ORDER BY created_at ASC
                """
                cursor.execute(query)
            
            signals = cursor.fetchall()
            
            if signals:
                # Update last processed timestamp
                self.last_signal_timestamp = max(signal['created_at'] for signal in signals)
            
            cursor.close()
            conn.close()
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error fetching signals: {e}")
            return []
    
    def publish_signal(self, signal: Dict) -> bool:
        """Publish a single signal to Redis"""
        try:
            if not self.redis_client:
                if not self.connect_redis():
                    return False
            
            # Create signal message
            signal_message = {
                'signal_id': signal['id'],
                'symbol': signal['symbol'],
                'signal_type': signal['signal_type'],
                'confidence': float(signal['confidence']),
                'xgboost_confidence': float(signal.get('xgboost_confidence', 0)),
                'price': float(signal['price']),
                'created_at': signal['created_at'].isoformat(),
                'model_version': signal.get('model_version'),
                'signal_strength': float(signal.get('signal_strength', 0)),
                'features_used': signal.get('features_used'),
                'timestamp': datetime.utcnow().isoformat(),
                'source': 'k8s_signal_generator',
                'published_via': 'redis'
            }
            
            # Publish to multiple channels for redundancy
            channels = [
                'trading_signals',  # Main channel
                f'trading_signals_{signal["symbol"]}',  # Symbol-specific channel
                f'trading_signals_{signal["signal_type"].lower()}'  # Signal type channel
            ]
            
            for channel in channels:
                self.redis_client.publish(channel, json.dumps(signal_message))
            
            # Store in Redis with expiration (1 hour)
            signal_key = f"signal:{signal['id']}"
            self.redis_client.setex(signal_key, 3600, json.dumps(signal_message))
            
            self.signals_published += 1
            logger.info(f"📡 Published signal: {signal['symbol']} {signal['signal_type']} "
                       f"({signal['confidence']:.3f} confidence) to {len(channels)} channels")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error publishing signal {signal.get('id', 'unknown')}: {e}")
            self.publish_errors += 1
            return False
    
    def publish_signals(self, signals: List[Dict]) -> int:
        """Publish multiple signals"""
        published_count = 0
        
        for signal in signals:
            if self.publish_signal(signal):
                published_count += 1
        
        if published_count > 0:
            logger.info(f"🚀 Published {published_count}/{len(signals)} signals to Redis")
        
        return published_count
    
    def get_metrics(self) -> Dict:
        """Get publisher metrics"""
        return {
            'signals_published': self.signals_published,
            'publish_errors': self.publish_errors,
            'last_signal_timestamp': self.last_signal_timestamp.isoformat() if self.last_signal_timestamp else None,
            'redis_connected': self.redis_client is not None,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def run_continuous_publishing(self):
        """Run continuous signal publishing"""
        logger.info("🚀 Starting Redis signal publisher...")
        
        while True:
            try:
                # Get new signals
                signals = self.get_new_signals()
                
                if signals:
                    # Publish signals
                    published = self.publish_signals(signals)
                    
                    if published > 0:
                        logger.info(f"📊 Published {published} new signals")
                
                # Sleep for 1 second for real-time responsiveness
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Error in publishing loop: {e}")
                await asyncio.sleep(5)  # Longer sleep on error

def main():
    """Main function"""
    publisher = RedisSignalPublisher()
    
    # Connect to Redis
    if not publisher.connect_redis():
        logger.error("❌ Failed to connect to Redis. Exiting...")
        return
    
    # Run continuous publishing
    try:
        asyncio.run(publisher.run_continuous_publishing())
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown signal received. Stopping publisher...")
    except Exception as e:
        logger.error(f"❌ Publisher error: {e}")

if __name__ == "__main__":
    main()