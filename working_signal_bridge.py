#!/usr/bin/env python3
"""
Working Signal Bridge - Fixed Version
Converts ML signals into actionable trade recommendations
"""

import os
import sys
import logging
import mysql.connector
import time
from datetime import datetime, timedelta
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Working Signal Bridge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
health_status = {
    "status": "starting",
    "database_connected": False,
    "last_bridge_cycle": None,
    "signals_processed_today": 0,
    "recommendations_created_today": 0,
    "last_error": None
}

def get_db_connection():
    """Get database connection"""
    try:
        config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'user': os.getenv('DB_USER', 'news_collector'),
            'password': os.getenv('DB_PASSWORD', '99Rules!'),
            'database': os.getenv('DB_NAME', 'crypto_transactions'),
            'charset': 'utf8mb4'
        }
        conn = mysql.connector.connect(**config)
        health_status["database_connected"] = True
        return conn
    except Exception as e:
        logger.error(f"❌ Database connection error: {e}")
        health_status["database_connected"] = False
        health_status["last_error"] = f"Database error: {e}"
        return None

def get_unprocessed_signals():
    """Get unprocessed signals from the database"""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Get recent unprocessed signals
        query = """
        SELECT * FROM trading_signals 
        WHERE processed = 0 
        AND timestamp >= NOW() - INTERVAL 1 HOUR
        AND signal_type IN ('BUY', 'SELL')
        ORDER BY timestamp DESC
        LIMIT 50
        """
        
        cursor.execute(query)
        signals = cursor.fetchall()
        
        logger.info(f"🔍 Found {len(signals)} unprocessed signals")
        return signals
        
    except Exception as e:
        logger.error(f"❌ Error getting unprocessed signals: {e}")
        return []
    finally:
        if conn:
            conn.close()

def create_trade_recommendation(signal):
    """Create a trade recommendation from a signal"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Calculate position size based on confidence
        confidence = signal.get('confidence', 0.5)
        base_amount = 100.0  # Base amount in USD
        
        # Scale position size based on confidence
        if confidence >= 0.8:
            position_size = base_amount * 1.5  # High confidence
        elif confidence >= 0.6:
            position_size = base_amount * 1.0  # Medium confidence
        else:
            position_size = base_amount * 0.5  # Low confidence
        
        # Determine action
        if signal['signal_type'] == 'BUY':
            action = 'BUY'
            amount_usd = position_size
        else:  # SELL
            action = 'SELL'
            amount_usd = position_size
        
        # Create trade recommendation
        query = """
        INSERT INTO trade_recommendations (
            symbol, action, amount_usd, confidence, signal_id, 
            created_at, status, priority, source
        ) VALUES (
            %s, %s, %s, %s, %s,
            NOW(), 'pending', 'normal', 'signal_bridge'
        )
        """
        
        cursor.execute(query, (
            signal['symbol'],
            action,
            amount_usd,
            confidence,
            signal['id']
        ))
        
        conn.commit()
        recommendation_id = cursor.lastrowid
        
        logger.info(f"✅ Created trade recommendation ID {recommendation_id}: {signal['symbol']} {action} ${amount_usd:.2f} (confidence: {confidence:.3f})")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating trade recommendation: {e}")
        return False
    finally:
        if conn:
            conn.close()

def mark_signal_processed(signal_id):
    """Mark a signal as processed"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        query = "UPDATE trading_signals SET processed = 1 WHERE id = %s"
        cursor.execute(query, (signal_id,))
        conn.commit()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error marking signal as processed: {e}")
        return False
    finally:
        if conn:
            conn.close()

def bridge_signals_cycle():
    """Process signals and create trade recommendations"""
    try:
        logger.info("🚀 Starting signal bridge cycle...")
        
        signals = get_unprocessed_signals()
        if not signals:
            logger.info("ℹ️ No unprocessed signals found")
            return
        
        signals_processed = 0
        recommendations_created = 0
        
        for signal in signals:
            try:
                # Create trade recommendation
                if create_trade_recommendation(signal):
                    recommendations_created += 1
                    
                    # Mark signal as processed
                    if mark_signal_processed(signal['id']):
                        signals_processed += 1
                        logger.info(f"✅ Processed signal {signal['id']}: {signal['symbol']} {signal['signal_type']}")
                    else:
                        logger.error(f"❌ Failed to mark signal {signal['id']} as processed")
                else:
                    logger.error(f"❌ Failed to create recommendation for signal {signal['id']}")
                
            except Exception as e:
                logger.error(f"❌ Error processing signal {signal.get('id', 'unknown')}: {e}")
                continue
        
        health_status["last_bridge_cycle"] = datetime.now().isoformat()
        health_status["signals_processed_today"] += signals_processed
        health_status["recommendations_created_today"] += recommendations_created
        health_status["status"] = "healthy"
        
        logger.info(f"✅ Signal bridge cycle complete: {signals_processed} signals processed, {recommendations_created} recommendations created")
        
    except Exception as e:
        logger.error(f"❌ Error in signal bridge cycle: {e}")
        health_status["last_error"] = f"Bridge cycle error: {e}"
        health_status["status"] = "error"

def signal_bridge_worker():
    """Background worker for signal bridging"""
    logger.info("🔄 Starting signal bridge worker...")
    
    while True:
        try:
            # Only proceed if database is connected
            if health_status["database_connected"]:
                logger.info("🚀 Starting signal bridge cycle...")
                bridge_signals_cycle()
            else:
                logger.warning("⚠️ Waiting for database to be ready...")
            
            # Sleep for 30 seconds
            time.sleep(30)
            
        except Exception as e:
            logger.error(f"❌ Error in signal bridge worker: {e}")
            health_status["last_error"] = f"Worker error: {e}"
            time.sleep(60)  # Shorter sleep on error

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    is_healthy = (
        health_status["database_connected"] and
        health_status["status"] not in ["error"]
    )
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "service": "working-signal-bridge",
        "database_connected": health_status["database_connected"],
        "last_bridge_cycle": health_status["last_bridge_cycle"],
        "signals_processed_today": health_status["signals_processed_today"],
        "recommendations_created_today": health_status["recommendations_created_today"],
        "last_error": health_status["last_error"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/status")
async def get_status():
    """Detailed status endpoint"""
    return health_status

@app.post("/bridge_signals")
async def trigger_signal_bridging():
    """Manual trigger for signal bridging"""
    try:
        bridge_signals_cycle()
        return {"status": "success", "message": "Signal bridging triggered"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.on_event("startup")
async def startup_event():
    """Initialize the service"""
    logger.info("🚀 Starting Working Signal Bridge...")
    
    # Test database connection
    try:
        if get_db_connection():
            logger.info("✅ Database connection successful")
        else:
            logger.error("❌ Database connection failed")
    except Exception as e:
        logger.error(f"❌ Database connection error: {e}")
    
    # Start background worker
    worker_thread = threading.Thread(target=signal_bridge_worker, daemon=True)
    worker_thread.start()
    
    logger.info("✅ Working Signal Bridge initialized successfully")

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8022))
    uvicorn.run(app, host="0.0.0.0", port=port)
