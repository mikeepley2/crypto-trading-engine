#!/usr/bin/env python3
"""
Live Trading Engine Local Launcher
Starts the live trading engine with local database configuration
"""

import os
import sys
import subprocess
import time
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_live_trading_engine():
    """Start the live trading engine with proper local configuration"""
    
    logger.info("🚀 STARTING LIVE TRADING ENGINE (LOCAL MODE)")
    logger.info("⚠️  WARNING: This will execute REAL TRADES!")
    
    # Load base environment from .env.live
    load_dotenv('.env.live')
    
    # Set up environment with local database configuration
    env = os.environ.copy()
    env.update({
        # Trading mode
        'EXECUTION_MODE': 'live',
        'TRADE_EXECUTION_ENABLED': 'true',
        'LIVE_TRADING_ENABLED': 'true',
        
        # Database - use localhost for local execution
        'DB_HOST': 'localhost',
        'DB_USER': 'news_collector',
        'DB_PASSWORD': '99Rules!',
        'DB_NAME_TRANSACTIONS': 'crypto_transactions',
        'DB_NAME_PRICES': 'crypto_prices',
        
        # Trading safety limits
        'MAX_POSITION_SIZE_USD': '500.00',
        'MAX_DAILY_TRADES': '20',
        'MAX_DAILY_LOSS_USD': '100.00',
        'MIN_TRADE_SIZE_USD': '5.00',
        'RISK_MANAGEMENT_ENABLED': 'true',
        
        # Port configuration
        'LIVE_TRADING_PORT': '8021',
        
        # Python path
        'PYTHONPATH': os.getcwd()
    })
    
    # Show configuration
    logger.info("📊 LIVE TRADING CONFIGURATION:")
    logger.info(f"  Database Host: {env['DB_HOST']}")
    logger.info(f"  Max Position Size: ${env['MAX_POSITION_SIZE_USD']}")
    logger.info(f"  Max Daily Loss: ${env['MAX_DAILY_LOSS_USD']}")
    logger.info(f"  Risk Management: {env['RISK_MANAGEMENT_ENABLED']}")
    logger.info(f"  Trading Port: {env['LIVE_TRADING_PORT']}")
    
    # Start the live trading engine
    try:
        engine_path = 'backend/services/trading/live/live_trading_engine.py'
        
        logger.info(f"🔥 Starting live trading engine: {engine_path}")
        
        # Start the process
        process = subprocess.Popen([
            sys.executable, engine_path
        ], env=env, cwd=os.getcwd())
        
        logger.info("✅ Live trading engine started!")
        logger.info("🌐 Engine should be running on http://localhost:8021")
        logger.info("📊 Monitor with: curl http://localhost:8021/health")
        logger.info("💡 Stop with Ctrl+C")
        
        # Wait for the process
        try:
            process.wait()
        except KeyboardInterrupt:
            logger.info("🛑 Stopping live trading engine...")
            process.terminate()
            process.wait()
            logger.info("✅ Live trading engine stopped")
            
    except Exception as e:
        logger.error(f"❌ Failed to start live trading engine: {e}")
        return False
    
    return True

if __name__ == "__main__":
    start_live_trading_engine()
