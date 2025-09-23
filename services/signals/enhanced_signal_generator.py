#!/usr/bin/env python3
"""
Simplified Enhanced Signal Generator
Generates trading signals using ML model and stores them in the database
"""

import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
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

def send_critical_alert(alert_type: str, message: str):
    """Send critical alert for ML model failures"""
    try:
        alert_data = {
            'type': alert_type,
            'severity': 'CRITICAL',
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'service': 'enhanced-signal-generator'
        }
        
        # Log locally
        logger.critical(f"🚨 CRITICAL ALERT [{alert_type}]: {message}")
        
        # Save alert to file for external monitoring
        alert_file = f"/tmp/critical_alert_{alert_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            import json
            with open(alert_file, 'w') as f:
                json.dump(alert_data, f, indent=2)
            logger.critical(f"📝 Critical alert saved to: {alert_file}")
        except Exception as file_error:
            logger.error(f"Failed to save alert file: {file_error}")
        
        # Try to send to notification service if available
        try:
            import requests
            notification_url = "http://notification-service.crypto-monitoring.svc.cluster.local:8038/alert"
            requests.post(notification_url, json=alert_data, timeout=5)
        except:
            pass  # Notification service may not be available
            
    except Exception as e:
        logger.error(f"Failed to send critical alert: {e}")

# Add path for coinbase_asset_filter
try:
    from coinbase_asset_filter import is_asset_supported
    logger.info("✅ Coinbase asset filter imported successfully")
    ASSET_FILTER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Could not import coinbase_asset_filter: {e}")
    ASSET_FILTER_AVAILABLE = False
    # Fallback function if import fails
    def is_asset_supported(symbol):
        # Basic fallback - reject known unsupported assets
        unsupported = {'RNDR', 'RENDER'}
        return symbol not in unsupported

# FastAPI app
app = FastAPI(title="Enhanced Signal Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
health_status = {
    "status": "starting",
    "model_loaded": False,
    "database_connected": False,
    "last_signal_generation": None,
    "signals_generated_today": 0,
    "last_error": None
}

def load_model():
    """Load the XGBoost model - FAIL if model cannot be loaded (NO FALLBACK MODE)"""
    global model
    model_path = "/app/optimal_66_percent_xgboost.joblib"
    
    # Check if model file exists
    if not os.path.exists(model_path):
        error_msg = f"❌ CRITICAL: ML model file not found at {model_path}. System cannot operate without ML model."
        logger.critical(error_msg)
        health_status["model_loaded"] = False
        health_status["critical_error"] = error_msg
        raise FileNotFoundError(error_msg)
    
    try:
        logger.info(f"🔄 Loading ML model from {model_path}...")
        model = joblib.load(model_path)
        
        # Validate model is actually loaded and functional
        if not hasattr(model, 'predict') or not hasattr(model, 'predict_proba'):
            error_msg = f"❌ CRITICAL: Loaded model is not a valid ML classifier. Missing predict/predict_proba methods."
            logger.critical(error_msg)
            health_status["model_loaded"] = False
            health_status["critical_error"] = error_msg
            raise ValueError(error_msg)
        
        # Test model with dummy data to ensure it's functional
        try:
            import numpy as np
            dummy_features = np.random.random((1, 79))  # 79 features as per metadata
            test_prediction = model.predict_proba(dummy_features)
            if test_prediction is None or len(test_prediction) == 0:
                raise ValueError("Model test prediction failed")
        except Exception as test_error:
            error_msg = f"❌ CRITICAL: ML model failed functionality test: {test_error}"
            logger.critical(error_msg)
            health_status["model_loaded"] = False
            health_status["critical_error"] = error_msg
            raise ValueError(error_msg)
        
        health_status["model_loaded"] = True
        health_status["model_type"] = str(type(model))
        health_status["critical_error"] = None
        logger.info("✅ ML model loaded and validated successfully")
        return True
        
    except Exception as e:
        error_msg = f"❌ CRITICAL: Failed to load ML model: {e}"
        logger.critical(error_msg)
        health_status["model_loaded"] = False
        health_status["critical_error"] = error_msg
        # Send critical alert
        send_critical_alert("ML_MODEL_LOAD_FAILURE", error_msg)
        raise Exception(error_msg)

def get_db_connection():
    """Get database connection"""
    try:
        config = {
            'host': os.getenv('DB_HOST', 'host.docker.internal'),
            'user': os.getenv('DB_USER', 'news_collector'),
            'password': os.getenv('DB_PASSWORD', '99Rules!'),
            'database': os.getenv('DB_NAME', 'crypto_prices'),
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

def get_latest_features(symbol):
    """Get the latest features for a symbol from ml_features_materialized"""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Get the latest features for the symbol
        query = """
        SELECT * FROM ml_features_materialized 
        WHERE symbol = %s 
        ORDER BY timestamp_iso DESC 
        LIMIT 1
        """
        cursor.execute(query, (symbol,))
        result = cursor.fetchone()
        
        if result:
            # Convert to the format expected by the model
            feature_columns = [col for col in result.keys() if col not in ['symbol', 'timestamp', 'price']]
            features = [result[col] for col in feature_columns if result[col] is not None]
            
            if len(features) >= 100:  # Minimum feature count
                return np.array(features).reshape(1, -1)
        
        return None
        
    except Exception as e:
        logger.error(f"❌ Error getting features for {symbol}: {e}")
        return None
    finally:
        if conn:
            conn.close()

def generate_signal(symbol, features):
    """Generate a trading signal for a symbol"""
    global model
    
    if model is None:
        return None
    
    try:
        # REMOVED FALLBACK MODE - System must use ML model only
        if not hasattr(model, 'predict') or not hasattr(model, 'predict_proba'):
            error_msg = f"❌ CRITICAL: ML model is not functional. Cannot generate signals without valid ML model."
            logger.critical(error_msg)
            send_critical_alert("ML_MODEL_NOT_FUNCTIONAL", error_msg)
            raise ValueError(error_msg)
        
        # ML model prediction (ONLY VALID PATH)
        # ML model prediction (ONLY VALID PATH)
        try:
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            confidence = max(probabilities)
            
            # Convert prediction to signal
            if prediction == 1 and confidence > 0.7:  # Buy signal (increased threshold)
                signal_type = "BUY"
            elif prediction == 0 and confidence > 0.7:  # Sell signal (increased threshold)
                signal_type = "SELL"
            else:
                signal_type = "HOLD"
            
            return {
                'symbol': symbol,
                'signal_type': signal_type,
                'confidence': confidence,
                'prediction': prediction,
                'model_version': 'optimal_66_percent_xgboost'
            }
        except Exception as ml_error:
            error_msg = f"❌ ML model prediction failed for {symbol}: {ml_error}"
            logger.error(error_msg)
            send_critical_alert("ML_PREDICTION_FAILURE", f"{error_msg} - Features shape: {features.shape if hasattr(features, 'shape') else 'Unknown'}")
            raise Exception(error_msg)
        
    except Exception as e:
        logger.error(f"❌ Error generating signal for {symbol}: {e}")
        return None

def save_signal_to_db(signal):
    """Save a signal to the database with correct schema"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Get current price for the symbol (required field)
        try:
            price_query = "SELECT current_price FROM price_data WHERE symbol = %s ORDER BY timestamp DESC LIMIT 1"
            cursor.execute(price_query, (signal['symbol'],))
            price_result = cursor.fetchone()
            current_price = float(price_result[0]) if price_result and price_result[0] else 0.0
        except:
            current_price = 0.0
        
        query = """
        INSERT INTO trading_signals (
            timestamp, symbol, price, signal_type, model, confidence, 
            threshold, regime, model_version, features_used, xgboost_confidence,
            data_source, created_at, is_mock, processed, prediction
        ) VALUES (
            NOW(), %s, %s, %s, %s, %s, 
            0.8, 'bull', %s, 79, %s,
            'database', NOW(), 0, 0, %s
        )
        """
        
        cursor.execute(query, (
            signal['symbol'],                              # symbol
            current_price,                                 # price
            signal['signal_type'],                         # signal_type
            signal.get('model_version', 'xgboost_4h'),    # model
            signal['confidence'],                          # confidence
            signal.get('model_version', 'xgboost_4h'),    # model_version
            signal['confidence'],                          # xgboost_confidence
            float(signal.get('prediction', 1.0))          # prediction
        ))
        
        conn.commit()
        signal_id = cursor.lastrowid
        logger.info(f"✅ Signal saved to DB: ID {signal_id}, {signal['symbol']} {signal['signal_type']} (confidence: {signal['confidence']:.3f})")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving signal to database: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_active_symbols():
    """Get list of active symbols to generate signals for (filtered by Coinbase support)"""
    conn = get_db_connection()
    if not conn:
        # Fallback to hardcoded symbols if database unavailable
        fallback_symbols = ['BTC', 'ETH', 'ADA', 'DOGE', 'XRP', 'DOT', 'LINK', 'UNI', 'AAVE', 'SOL']
        # Filter fallback symbols through asset filter
        filtered_symbols = [s for s in fallback_symbols if is_asset_supported(s)]
        logger.info(f"Database unavailable, using filtered fallback symbols: {filtered_symbols}")
        return filtered_symbols
    
    try:
        cursor = conn.cursor()
        
        # Get symbols with recent price data - prioritize common crypto symbols
        query = """
        SELECT DISTINCT symbol 
        FROM ml_features_materialized 
        WHERE timestamp_iso >= NOW() - INTERVAL 30 DAY
        AND symbol IN ('BTC', 'ETH', 'ADA', 'DOGE', 'XRP', 'DOT', 'LINK', 'UNI', 'AAVE', 'SOL', 
                       'MATIC', 'AVAX', 'LTC', 'BCH', 'ATOM', 'ICP', 'FIL', 'TRX', 'ETC', 'XLM')
        LIMIT 20
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        symbols = [row[0] for row in results]
        
        # If no common symbols found, use any available symbols
        if not symbols:
            query = """
            SELECT DISTINCT symbol 
            FROM ml_features_materialized 
            WHERE timestamp_iso >= NOW() - INTERVAL 30 DAY
            LIMIT 20
            """
            cursor.execute(query)
            results = cursor.fetchall()
            symbols = [row[0] for row in results]
        
        # Filter symbols through coinbase asset filter
        original_count = len(symbols)
        filtered_symbols = []
        
        for symbol in symbols:
            if is_asset_supported(symbol):
                filtered_symbols.append(symbol)
            else:
                logger.info(f"[ASSET_FILTER] Skipping unsupported asset: {symbol}")
        
        # If still no symbols after filtering, use fallback list
        if not filtered_symbols:
            fallback_symbols = ['BTC', 'ETH', 'ADA', 'DOGE', 'XRP', 'DOT', 'LINK', 'UNI', 'AAVE', 'SOL']
            filtered_symbols = [s for s in fallback_symbols if is_asset_supported(s)]
            logger.warning(f"No supported symbols found in database, using filtered fallback: {filtered_symbols}")
        
        logger.info(f"[ASSET_FILTER] Filtered {original_count} symbols down to {len(filtered_symbols)} supported assets")
        return filtered_symbols
        
    except Exception as e:
        logger.error(f"❌ Error getting active symbols: {e}")
        # Return filtered fallback symbols
        fallback_symbols = ['BTC', 'ETH', 'ADA', 'DOGE', 'XRP', 'DOT', 'LINK', 'UNI', 'AAVE', 'SOL']
        filtered_symbols = [s for s in fallback_symbols if is_asset_supported(s)]
        logger.info(f"Error fallback - using filtered symbols: {filtered_symbols}")
        return filtered_symbols
    finally:
        if conn:
            conn.close()

def generate_signals_cycle():
    """Generate signals for all active symbols"""
    try:
        logger.info("🚀 Starting signal generation cycle...")
        
        symbols = get_active_symbols()
        if not symbols:
            logger.warning("⚠️ No active symbols found")
            return
        
        signals_generated = 0
        
        for symbol in symbols[:20]:  # Limit to top 20 symbols
            try:
                # Double-check asset support before generating signal
                if not is_asset_supported(symbol):
                    logger.info(f"[ASSET_FILTER] Skipping signal generation for unsupported asset: {symbol}")
                    continue
                
                signal = None
                
                # Always try fallback mode if model is fallback OR if features are insufficient
                if model == "fallback":
                    signal = generate_signal(symbol, None)
                    logger.debug(f"Using fallback mode for {symbol}")
                else:
                    features = get_latest_features(symbol)
                    if features is not None:
                        signal = generate_signal(symbol, features)
                        logger.debug(f"Using ML features for {symbol}")
                    else:
                        # Use fallback mode when features are insufficient
                        signal = generate_signal(symbol, None)
                        logger.debug(f"Using fallback mode for {symbol} (insufficient features)")
                
                if signal and signal['signal_type'] != 'HOLD':
                    if save_signal_to_db(signal):
                        signals_generated += 1
                        logger.info(f"✅ Generated {signal['signal_type']} signal for {symbol} (confidence: {signal['confidence']:.3f})")
                elif signal and signal['signal_type'] == 'HOLD':
                    logger.debug(f"⚠️ Generated HOLD signal for {symbol} - not saved")
                
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")
                continue
        
        health_status["last_signal_generation"] = datetime.now().isoformat()
        health_status["signals_generated_today"] += signals_generated
        health_status["status"] = "healthy"
        
        logger.info(f"✅ Signal generation cycle complete: {signals_generated} signals generated")
        
    except Exception as e:
        logger.error(f"❌ Error in signal generation cycle: {e}")
        health_status["last_error"] = f"Signal generation error: {e}"
        health_status["status"] = "error"

def signal_generation_worker():
    """Background worker for signal generation - STOPS if ML model fails"""
    logger.info("🔄 Starting signal generation worker...")
    
    while True:
        try:
            # Check for critical errors that should stop the service
            if health_status.get("critical_error"):
                error_msg = health_status["critical_error"]
                logger.critical(f"🛑 CRITICAL ERROR: Service cannot continue - {error_msg}")
                send_critical_alert("SERVICE_STOPPED", f"Signal generation stopped due to critical error: {error_msg}")
                # Sleep longer and keep checking - don't exit to allow for external intervention
                time.sleep(300)  # 5 minutes
                continue
            
            # Only proceed if model is loaded AND functional
            if health_status["model_loaded"] and health_status["database_connected"]:
                # Additional validation: ensure model is still functional
                if not hasattr(model, 'predict') or model is None:
                    error_msg = "ML model became non-functional during runtime"
                    logger.critical(f"🛑 {error_msg}")
                    health_status["critical_error"] = error_msg
                    send_critical_alert("ML_MODEL_RUNTIME_FAILURE", error_msg)
                    continue
                
                logger.info("🚀 Starting signal generation cycle with ML model...")
                generate_signals_cycle()
            else:
                logger.warning("⚠️ Waiting for model and database to be ready...")
                # Check if we're in a permanent failure state
                if not health_status["model_loaded"]:
                    logger.critical("🛑 ML model is not loaded - service cannot operate")
            
            # Sleep for 10 minutes (optimized for stability)
            time.sleep(600)
            
        except Exception as e:
            error_msg = f"Critical error in signal generation worker: {e}"
            logger.critical(error_msg)
            health_status["critical_error"] = error_msg
            send_critical_alert("WORKER_CRITICAL_ERROR", error_msg)
            time.sleep(60)  # Shorter sleep on error

@app.get("/health")
async def health_check():
    """Health check endpoint - UNHEALTHY if ML model fails"""
    
    # Check for critical errors that make service non-functional
    if health_status.get("critical_error"):
        return {
            "status": "critical",
            "healthy": False,
            "error": health_status["critical_error"],
            "message": "Service is in critical error state and cannot operate",
            "model_loaded": health_status.get("model_loaded", False),
            "database_connected": health_status.get("database_connected", False),
            "timestamp": datetime.now().isoformat()
        }
    
    # Standard health check
    is_healthy = (
        health_status["model_loaded"] and 
        health_status["database_connected"] and
        health_status["status"] not in ["error", "critical"]
    )
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "service": "enhanced-signal-generator",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/status")
async def get_status():
    """Detailed status endpoint"""
    return health_status

@app.post("/generate_signals")
async def trigger_signal_generation():
    """Manual trigger for signal generation"""
    try:
        generate_signals_cycle()
        return {"status": "success", "message": "Signal generation triggered"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.on_event("startup")
async def startup_event():
    """Initialize the service - FAIL if ML model cannot be loaded"""
    logger.info("🚀 Starting Enhanced Signal Generator...")
    
    # Load model - CRITICAL REQUIREMENT
    try:
        if load_model():
            logger.info("✅ ML model loaded and validated successfully")
        else:
            # This should never happen with the new load_model function
            error_msg = "ML model loading returned False"
            logger.critical(f"🚨 CRITICAL: {error_msg}")
            health_status["critical_error"] = error_msg
            send_critical_alert("STARTUP_ML_MODEL_FAILURE", error_msg)
    except Exception as e:
        # Critical failure - service cannot operate
        error_msg = f"CRITICAL STARTUP FAILURE: Cannot load ML model - {e}"
        logger.critical(f"🚨 {error_msg}")
        health_status["critical_error"] = error_msg
        health_status["model_loaded"] = False
        send_critical_alert("STARTUP_CRITICAL_FAILURE", error_msg)
        # Don't exit - let the service run in error state for monitoring
    
    # Test database connection
    try:
        if get_db_connection():
            logger.info("✅ Database connection successful")
        else:
            logger.error("❌ Database connection failed")
    except Exception as e:
        logger.error(f"❌ Database connection error: {e}")
    
    # Start background worker (will handle critical errors appropriately)
    worker_thread = threading.Thread(target=signal_generation_worker, daemon=True)
    worker_thread.start()
    
    # Final initialization status
    if health_status.get("critical_error"):
        logger.critical(f"🚨 Service started in CRITICAL ERROR state: {health_status['critical_error']}")
        logger.critical("🚨 Service will NOT generate signals until ML model is fixed")
    else:
        logger.info("✅ Enhanced Signal Generator initialized successfully")

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8025))
    uvicorn.run(app, host="0.0.0.0", port=port)
