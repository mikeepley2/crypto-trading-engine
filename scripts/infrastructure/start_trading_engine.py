#!/usr/bin/env python3
"""
Live Trading Engine Starter
Handles starting the live trading engine with proper error handling
"""
import asyncio
import uvicorn
import subprocess
import sys
import os
from pathlib import Path
import time
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_trade_execution_engine():
    """Start the trade execution engine"""
    
    # Check if we're already in the correct directory
    trading_dir = Path("backend/services/trading/trade-execution-engine")
    if not trading_dir.exists():
        logger.error(f"❌ Trade execution engine directory not found: {trading_dir}")
        return False
    
    engine_script = trading_dir / "trade_execution_engine.py"
    if not engine_script.exists():
        logger.error(f"❌ Trade execution engine script not found: {engine_script}")
        return False
    
    try:
        logger.info("🚀 Starting trade execution engine...")
        logger.info("ℹ️  Note: Engine can run in MOCK or LIVE mode - check .env settings")
        
        # Change to the trade execution engine directory
        original_dir = os.getcwd()
        os.chdir(trading_dir)
        
        # Set environment variables for trade execution
        env = os.environ.copy()
        env['TRADE_EXECUTION_PORT'] = '8024'  # Use the trade execution engine port
        env['PYTHONPATH'] = str(Path(original_dir).absolute())
        
        # Start the engine
        process = subprocess.Popen([
            sys.executable, "trade_execution_engine.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        
        # Give it a moment to start
        time.sleep(5)
        
        # Check if it's still running
        if process.poll() is None:
            logger.info("✅ Trade execution engine started successfully")
            logger.info(f"📊 PID: {process.pid}")
            logger.info("🌐 Health check: http://localhost:8024/health")
            return True
        else:
            stdout, stderr = process.communicate()
            logger.error(f"❌ Trade execution engine failed to start")
            logger.error(f"STDOUT: {stdout.decode()}")
            logger.error(f"STDERR: {stderr.decode()}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error starting trade execution engine: {e}")
        return False
    finally:
        # Return to original directory
        os.chdir(original_dir)

def check_trade_execution_engine():
    """Check if trade execution engine is running"""
    try:
        import requests
        response = requests.get("http://localhost:8024/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            mode = health_data.get('mode', 'unknown')
            logger.info(f"✅ Trade execution engine is running and healthy (Mode: {mode})")
            return True
        else:
            logger.warning(f"⚠️ Trade execution engine unhealthy: {response.status_code}")
            return False
    except Exception as e:
        logger.warning(f"⚠️ Trade execution engine not accessible: {e}")
        return False

def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1].lower() == "check":
        check_trade_execution_engine()
    else:
        if check_trade_execution_engine():
            logger.info("🎯 Trade execution engine already running")
        else:
            start_trade_execution_engine()

if __name__ == "__main__":
    main()
