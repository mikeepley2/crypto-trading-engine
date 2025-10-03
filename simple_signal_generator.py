#!/usr/bin/env python3
"""
Simple Signal Generator - Minimal Working Version
"""

import os
import sys
import logging
import time
import requests
from datetime import datetime
from fastapi import FastAPI
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Simple Signal Generator")

# Global variables
health_status = {
    "status": "healthy",
    "service": "simple-signal-generator",
    "started_at": datetime.now().isoformat(),
    "signals_generated": 0
}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "simple-signal-generator",
        "timestamp": datetime.now().isoformat(),
        "signals_generated": health_status["signals_generated"]
    }

@app.get("/status")
async def get_status():
    """Status endpoint"""
    return health_status

@app.post("/generate_test_signal")
async def generate_test_signal():
    """Generate a test signal and send it to the real trading system"""
    try:
        # Generate signal
        signal = {
            "symbol": "BTC",
            "signal_type": "BUY",
            "confidence": 0.75,
            "timestamp": datetime.now().isoformat()
        }
        
        health_status["signals_generated"] += 1
        logger.info(f"✅ Generated signal: {signal}")
        
        # Send directly to trade execution
        try:
            trade_response = requests.post(
                "http://localhost:8024/execute_trade",
                json={
                    "symbol": signal["symbol"],
                    "action": signal["signal_type"],
                    "size_usd": 10.0,  # Fixed amount for testing
                    "order_type": "MARKET"
                },
                timeout=10
            )
            if trade_response.status_code == 200:
                trade_result = trade_response.json()
                logger.info(f"✅ Trade executed: {trade_result}")
                return {
                    "status": "success",
                    "signal": signal,
                    "trade_result": trade_result,
                    "message": "Signal generated and trade executed successfully"
                }
            else:
                logger.warning(f"⚠️ Trade execution returned status {trade_response.status_code}")
        except Exception as trade_error:
            logger.warning(f"⚠️ Could not connect to trade execution: {trade_error}")
        
        return {
            "status": "success",
            "signal": signal,
            "message": "Signal generated successfully (trading system not available)"
        }
    except Exception as e:
        logger.error(f"❌ Error generating signal: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Simple Signal Generator is running",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "generate_test_signal": "POST /generate_test_signal"
        }
    }

if __name__ == "__main__":
    logger.info("🚀 Starting Simple Signal Generator...")
    port = int(os.getenv('PORT', 8025))
    logger.info(f"📡 Service will be available at http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
