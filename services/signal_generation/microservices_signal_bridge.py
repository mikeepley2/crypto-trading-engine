#!/usr/bin/env python3
"""
Microservices Signal Generator Bridge - Kubernetes Version
Directly calls microservices to generate signals via Kubernetes service discovery
"""

import asyncio
import aiohttp
import json
import logging
import mysql.connector
from datetime import datetime
import time
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MicroservicesSignalBridge:
    def __init__(self):
        self.session = None
        
        # Use orchestrator instead of individual microservices - NO FALLBACK MODE
        self.orchestrator_url = 'http://signal-gen-orchestrator.crypto-trading.svc.cluster.local:8025'
        
        # Keep microservice endpoints for health checks only
        self.services = {
            'feature_engine': 'http://signal-gen-feature-engine:8052',
            'market_context': 'http://signal-gen-market-context:8053',
            'portfolio': 'http://signal-gen-portfolio:8054',
            'risk_mgmt': 'http://signal-gen-risk-mgmt:8055',
            'analytics': 'http://signal-gen-analytics:8056'
        }
        
        # Database connection for K8s environment
        self.db_config = {
            'host': os.getenv('DB_HOST', 'mysql'),
            'user': os.getenv('DB_USER', 'news_collector'), 
            'password': os.getenv('DB_PASSWORD', '99Rules!'),
            'database': os.getenv('DB_NAME', 'crypto_prices')
        }

    async def setup_session(self):
        """Setup HTTP session"""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))

    async def cleanup_session(self):
        """Cleanup HTTP session"""
        if self.session:
            await self.session.close()

    async def call_service(self, service_name: str, endpoint: str, data: dict = None):
        """Call a microservice endpoint"""
        try:
            url = f"{self.services[service_name]}{endpoint}"
            
            if data:
                async with self.session.post(url, json=data) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"Service {service_name}{endpoint} returned {response.status}")
                        return None
            else:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"Service {service_name}{endpoint} returned {response.status}")
                        return None
                        
        except Exception as e:
            logger.debug(f"Service {service_name}{endpoint} unavailable: {e}")
            return None

    async def generate_signal_for_symbol(self, symbol: str) -> dict:
        """Generate signal for a single symbol using orchestrator - NO FALLBACK MODE"""
        start_time = time.time()
        
        try:
            logger.info(f"🔄 Generating signal for {symbol} via orchestrator")
            
            # Call orchestrator for comprehensive signal generation
            request_data = {
                'symbol': symbol,
                'analysis_type': 'comprehensive',
                'timestamp': datetime.now().isoformat()
            }
            
            async with self.session.post(
                f"{self.orchestrator_url}/generate_signal",
                json=request_data,
                timeout=30
            ) as response:
                if response.status == 200:
                    orchestrator_result = await response.json()
                    
                    # Map orchestrator response to expected format
                    signal_result = {
                        'signal': orchestrator_result.get('signal', 'HOLD').upper(),
                        'confidence': float(orchestrator_result.get('confidence', 0.0)),
                        'symbol': symbol,
                        'price': float(orchestrator_result.get('price', 0.0)),
                        'timestamp': datetime.now(),
                        'model': 'orchestrator_microservices',
                        'data_source': 'orchestrator',
                        'signal_strength': float(orchestrator_result.get('signal_strength', 1.0)),
                        'features_used': orchestrator_result.get('features_used', 0),
                        'processing_time_ms': (time.time() - start_time) * 1000,
                        'analysis_details': orchestrator_result.get('analysis_details', {})
                    }
                    
                    # Validate that we have proper ML-driven confidence
                    if signal_result['confidence'] < 0.1:
                        raise ValueError(f"Orchestrator returned invalid confidence {signal_result['confidence']} for {symbol}")
                    
                    logger.info(f"✅ Generated signal for {symbol}: {signal_result['signal']} (confidence: {signal_result['confidence']:.3f})")
                    return signal_result
                else:
                    error_msg = f"Orchestrator returned status {response.status} for {symbol}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Signal generation failed for {symbol}: {e}")
            # NO FALLBACK MODE - Fail hard if orchestrator is unavailable
            raise Exception(f"Signal generation failed for {symbol} - orchestrator unavailable: {e}")

    def save_signal_to_database(self, signal_data: dict):
        """Save generated signal to trading_signals table"""
        try:
            db = mysql.connector.connect(**self.db_config)
            cursor = db.cursor()
            
            # Insert signal with all required fields - SET is_mock = 0 for LIVE TRADING
            insert_query = """
                INSERT INTO trading_signals (
                    timestamp, symbol, price, signal_type, model, confidence, 
                    threshold, regime, model_version, features_used, xgboost_confidence,
                    data_source, signal_strength, is_mock
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                signal_data['timestamp'],
                signal_data['symbol'],
                signal_data['price'],
                signal_data['signal'],
                signal_data['model'],
                signal_data['confidence'],
                0.7,  # threshold
                'sideways',  # regime - default
                'microservices_bridge_v1',  # model_version
                signal_data['features_used'],
                signal_data['confidence'],  # xgboost_confidence (same as confidence)
                signal_data['data_source'],
                signal_data['signal_strength'],
                0  # is_mock = 0 for LIVE TRADING (not mock)
            )
            
            cursor.execute(insert_query, values)
            db.commit()
            
            logger.info(f"💾 Saved signal for {signal_data['symbol']} to database")
            
            cursor.close()
            db.close()
            
        except Exception as e:
            logger.error(f"❌ Error saving signal to database: {e}")

    async def generate_signals_batch(self):
        """Generate signals for all major crypto symbols"""
        symbols = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'XRP', 'LTC', 'BCH', 'UNI', 'AAVE', 'MATIC', 'ATOM']
        
        logger.info(f"🚀 Starting batch signal generation for {len(symbols)} symbols")
        
        # Track statistics
        successful_signals = 0
        failed_signals = 0
        
        for symbol in symbols:
            try:
                signal_data = await self.generate_signal_for_symbol(symbol)
                self.save_signal_to_database(signal_data)
                successful_signals += 1
                
                # Small delay between symbols
                await asyncio.sleep(0.3)
                
            except Exception as e:
                logger.error(f"❌ Failed to process {symbol}: {e}")
                failed_signals += 1
                continue
                
        logger.info(f"✅ Batch signal generation completed - {successful_signals} successful, {failed_signals} failed")

    async def run_continuous(self):
        """Run continuous signal generation every 5 minutes"""
        await self.setup_session()
        
        try:
            while True:
                logger.info("🔄 Starting signal generation cycle")
                await self.generate_signals_batch()
                
                logger.info("⏰ Waiting 5 minutes until next cycle")
                await asyncio.sleep(300)  # 5 minutes
                
        except KeyboardInterrupt:
            logger.info("⏹️ Stopping signal generation")
        finally:
            await self.cleanup_session()

async def main():
    """Main function"""
    bridge = MicroservicesSignalBridge()
    
    logger.info("🚀 Starting Microservices Signal Bridge")
    logger.info("This service generates signals using microservices architecture")
    
    # Test database connectivity at startup
    try:
        db = mysql.connector.connect(**bridge.db_config)
        db.close()
        logger.info("✅ Database connection test successful")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        logger.info("Continuing anyway - signals will be generated but not saved")
    
    await bridge.run_continuous()

if __name__ == "__main__":
    asyncio.run(main())
