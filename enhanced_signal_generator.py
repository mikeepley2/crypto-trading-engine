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
import json
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
    model_path = os.path.join(
        os.path.dirname(__file__),
        "services",
        "signals",
        "full_dataset_gpu_xgboost_model_20250827_130225.joblib"
    )
    
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
            'host': os.getenv('DB_HOST', '172.22.32.1'),
            'user': os.getenv('DB_USER', 'news_collector'),
            'password': os.getenv('DB_PASSWORD', '99Rules!'),
            'database': os.getenv('DB_NAME', 'crypto_prices'),  # ML features from crypto_prices
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

def get_trading_db_connection():
    """Get trading database connection for saving signals"""
    try:
        config = {
            'host': os.getenv('DB_HOST', '172.22.32.1'),
            'user': os.getenv('DB_USER', 'news_collector'),
            'password': os.getenv('DB_PASSWORD', '99Rules!'),
            'database': 'crypto_transactions',  # FIXED: Trading signals go to crypto_transactions database
            'charset': 'utf8mb4'
        }
        conn = mysql.connector.connect(**config)
        return conn
    except Exception as e:
        logger.error(f"❌ Trading database connection error: {e}")
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
            # Convert to the format expected by the model - EXCLUDE datetime/date columns
            excluded_columns = {
                'id', 'symbol', 'timestamp', 'price', 'price_date', 'price_hour', 
                'timestamp_iso', 'created_at', 'updated_at'
            }
            feature_columns = [col for col in result.keys() if col not in excluded_columns]
            
            # Extract features and handle NULL values with defaults
            features = []
            for col in feature_columns:
                value = result[col]
                if value is not None:
                    # Convert datetime.date to numeric (days since epoch)
                    if hasattr(value, 'toordinal'):  # datetime.date object
                        features.append(float(value.toordinal()))
                    elif isinstance(value, (int, float)):
                        features.append(float(value))
                    elif hasattr(value, 'total_seconds'):  # timedelta object
                        features.append(float(value.total_seconds()))
                    else:
                        # Try to convert to float, use 0.0 if not possible
                        try:
                            features.append(float(value))
                        except (ValueError, TypeError):
                            logger.debug(f"Converting non-numeric column {col} to 0.0: {type(value)}")
                            features.append(0.0)
                else:
                    # Use 0.0 for NULL values to maintain feature structure
                    features.append(0.0)
            
            logger.info(f"🔍 {symbol} feature extraction: {len(features)}/{len(feature_columns)} features processed from {result.get('timestamp_iso', 'unknown timestamp')}")
            
            # Ensure we have exactly 79 features (shouldn't need padding now)
            if len(features) == 79:
                feature_array = np.array(features).reshape(1, -1)
                logger.info(f"✅ {symbol} features ready for ML: {len(features)} features")
                return feature_array
            else:
                logger.error(f"❌ {symbol} feature mismatch: got {len(features)}, expected 79")
                return None
        
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
            if prediction == 1 and confidence > 0.5:  # Buy signal (optimized threshold)
                signal_type = "BUY"
            elif prediction == 0 and confidence > 0.5:  # Sell signal (optimized threshold)
                signal_type = "SELL"
            else:
                signal_type = "HOLD"
            
            return {
                'symbol': symbol,
                'signal_type': signal_type,
                'confidence': float(confidence),  # Convert numpy types to Python float
                'prediction': float(prediction) if hasattr(prediction, 'item') else prediction,
                'model_version': 'full_dataset_gpu_xgboost_20250827'
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
    """Save a signal to the trading database (crypto_prices) with correct schema"""
    conn = get_trading_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Get current price for the symbol (required field) - use timestamp_iso for recent data
        try:
            price_query = """
            SELECT current_price FROM price_data 
            WHERE symbol = %s 
            AND timestamp_iso >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
            ORDER BY timestamp_iso DESC 
            LIMIT 1
            """
            cursor.execute(price_query, (signal['symbol'],))
            price_result = cursor.fetchone()
            current_price = float(price_result[0]) if price_result and price_result[0] else 0.0
            
            if current_price == 0.0:
                # Fallback to any recent price if no data in last hour
                fallback_query = "SELECT current_price FROM price_data WHERE symbol = %s ORDER BY timestamp_iso DESC LIMIT 1"
                cursor.execute(fallback_query, (signal['symbol'],))
                fallback_result = cursor.fetchone()
                current_price = float(fallback_result[0]) if fallback_result and fallback_result[0] else 0.0
                
        except Exception as e:
            logger.warning(f"[PRICE] Error getting price for {signal['symbol']}: {e}")
            current_price = 0.0
        
        # Use the ACTUAL trading_signals schema from crypto_transactions database
        query = """
        INSERT INTO trading_signals (
            symbol, signal_type, action, confidence, price, processed, created_at, reasoning
        ) VALUES (
            %s, %s, %s, %s, %s, 0, NOW(), %s
        )
        """
        
        # Map signal_type to action for compatibility
        action = 'BUY' if signal['signal_type'] in ['BUY', 'ML_SIGNAL'] else 'SELL'
        reasoning = f"ML Signal: {signal['signal_type']} {signal['symbol']} (conf: {signal['confidence']:.3f})"
        
        cursor.execute(query, (
            signal['symbol'],                              # symbol
            signal['signal_type'],                         # signal_type
            action,                                       # action (required field)
            float(signal['confidence']),                   # confidence
            current_price,                                # price  
            reasoning                                     # reasoning
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
    """Get list of ALL Coinbase-supported symbols with recent ML features data"""
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
        
        # Get ALL Coinbase-supported symbols with recent ML features data
        # This replaces the hardcoded list to enable full asset coverage
        query = """
        SELECT DISTINCT mf.symbol
        FROM ml_features_materialized mf
        INNER JOIN crypto_assets ca ON mf.symbol = ca.symbol
        WHERE mf.timestamp_iso >= NOW() - INTERVAL 7 DAY
        AND ca.coinbase_supported = 1
        ORDER BY 
            CASE 
                WHEN mf.symbol IN ('BTC', 'ETH', 'SOL', 'ADA', 'XRP', 'LINK', 'DOT', 'UNI', 'AVAX', 'MATIC') THEN 1
                ELSE 2
            END,
            mf.symbol
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        symbols = [row[0] for row in results]
        
        # If no Coinbase assets found, fallback to any symbols with recent data
        if not symbols:
            logger.warning("No Coinbase-supported symbols found, using any available symbols")
            query = """
            SELECT DISTINCT symbol 
            FROM ml_features_materialized 
            WHERE timestamp_iso >= NOW() - INTERVAL 7 DAY
            ORDER BY symbol
            LIMIT 50
            """
            cursor.execute(query)
            results = cursor.fetchall()
            symbols = [row[0] for row in results]
        
        # Apply additional asset filter for trading compatibility
        original_count = len(symbols)
        filtered_symbols = []
        
        for symbol in symbols:
            if is_asset_supported(symbol):
                filtered_symbols.append(symbol)
            else:
                logger.debug(f"[ASSET_FILTER] Skipping unsupported asset: {symbol}")
        
        # If still no symbols after filtering, use fallback list
        if not filtered_symbols:
            fallback_symbols = ['BTC', 'ETH', 'ADA', 'DOGE', 'XRP', 'DOT', 'LINK', 'UNI', 'AAVE', 'SOL']
            filtered_symbols = [s for s in fallback_symbols if is_asset_supported(s)]
            logger.warning(f"No supported symbols found in database, using filtered fallback: {filtered_symbols}")
        
        logger.info(f"✅ EXPANDED ASSET COVERAGE: Found {len(filtered_symbols)} Coinbase-supported assets (filtered from {original_count} total)")
        logger.info(f"📈 ASSET SYMBOLS: {', '.join(filtered_symbols[:10])}{'...' if len(filtered_symbols) > 10 else ''}")
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
    """Generate signals for all Coinbase-supported symbols with smart batching"""
    try:
        logger.info("🚀 Starting EXPANDED signal generation cycle...")
        
        symbols = get_active_symbols()
        if not symbols:
            logger.warning("⚠️ No active symbols found")
            return
        
        total_symbols = len(symbols)
        signals_generated = 0
        batch_size = int(os.getenv('SIGNAL_BATCH_SIZE', 25))  # Process in batches to avoid overwhelming
        
        logger.info(f"📊 FULL ASSET COVERAGE: Processing {total_symbols} Coinbase-supported assets in batches of {batch_size}")
        
        # Process symbols in batches for better performance and stability
        for i in range(0, total_symbols, batch_size):
            batch_symbols = symbols[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_symbols + batch_size - 1) // batch_size
            
            logger.info(f"🔄 Processing batch {batch_num}/{total_batches}: {len(batch_symbols)} symbols")
            batch_signals = 0
            
            for symbol in batch_symbols:
                try:
                    # Double-check asset support before generating signal
                    if not is_asset_supported(symbol):
                        logger.debug(f"[ASSET_FILTER] Skipping signal generation for unsupported asset: {symbol}")
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
                            batch_signals += 1
                            logger.info(f"✅ Generated {signal['signal_type']} signal for {symbol} (confidence: {signal['confidence']:.3f})")
                    elif signal and signal['signal_type'] == 'HOLD':
                        logger.debug(f"⚠️ Generated HOLD signal for {symbol} - not saved")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing {symbol}: {e}")
                    continue
            
            logger.info(f"✅ Batch {batch_num} complete: {batch_signals} signals generated")
            
            # Small delay between batches to avoid overwhelming the system
            if i + batch_size < total_symbols:
                time.sleep(2)
        
        health_status["last_signal_generation"] = datetime.now().isoformat()
        health_status["signals_generated_today"] += signals_generated
        health_status["status"] = "healthy"
        
        logger.info(f"🎯 EXPANDED SIGNAL GENERATION COMPLETE: {signals_generated}/{total_symbols} signals generated across ALL Coinbase assets")
        
    except Exception as e:
        logger.error(f"❌ Error in signal generation cycle: {e}")
        health_status["last_error"] = f"Signal generation error: {e}"
        health_status["status"] = "error"

def signal_generation_worker():
    """Background worker for signal generation with intelligent scheduling"""
    logger.info("🔄 Starting intelligent signal generation worker...")
    
    # Configurable intervals (in seconds)
    DEFAULT_INTERVAL = int(os.getenv('SIGNAL_GENERATION_INTERVAL', 1800))  # 30 minutes default
    MARKET_HOURS_INTERVAL = int(os.getenv('MARKET_HOURS_INTERVAL', 900))   # 15 minutes during active hours
    QUIET_HOURS_INTERVAL = int(os.getenv('QUIET_HOURS_INTERVAL', 3600))    # 1 hour during quiet hours
    
    logger.info(f"📅 Signal generation schedule: Default {DEFAULT_INTERVAL//60}min, Active {MARKET_HOURS_INTERVAL//60}min, Quiet {QUIET_HOURS_INTERVAL//60}min")
    
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
                
                # Intelligent scheduling based on market activity
                current_hour = datetime.now().hour
                is_market_hours = 6 <= current_hour <= 22  # 6 AM to 10 PM (crypto markets are 24/7 but less active at night)
                
                # Determine appropriate interval
                if is_market_hours:
                    sleep_interval = MARKET_HOURS_INTERVAL
                    schedule_type = "ACTIVE_HOURS"
                else:
                    sleep_interval = QUIET_HOURS_INTERVAL  
                    schedule_type = "QUIET_HOURS"
                
                logger.info(f"🚀 Starting signal generation cycle ({schedule_type}, next in {sleep_interval//60}min)...")
                generate_signals_cycle()
            else:
                logger.warning("⚠️ Waiting for model and database to be ready...")
                # Check if we're in a permanent failure state
                if not health_status["model_loaded"]:
                    logger.critical("🛑 ML model is not loaded - service cannot operate")
                sleep_interval = DEFAULT_INTERVAL
            
            # Intelligent sleep based on market conditions
            logger.info(f"⏰ Next signal generation in {sleep_interval//60} minutes...")
            time.sleep(sleep_interval)
            
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
    """Detailed status endpoint with scheduling information"""
    current_hour = datetime.now().hour
    is_market_hours = 6 <= current_hour <= 22
    
    return {
        **health_status,
        "scheduling": {
            "default_interval_minutes": int(os.getenv('SIGNAL_GENERATION_INTERVAL', 1800)) // 60,
            "market_hours_interval_minutes": int(os.getenv('MARKET_HOURS_INTERVAL', 900)) // 60,
            "quiet_hours_interval_minutes": int(os.getenv('QUIET_HOURS_INTERVAL', 3600)) // 60,
            "current_hour": current_hour,
            "is_market_hours": is_market_hours,
            "next_generation_estimated": "automatic_based_on_schedule",
            "current_interval": "15min (market hours)" if is_market_hours else "60min (quiet hours)"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate_signals")
async def trigger_signal_generation():
    """Manual trigger for signal generation"""
    try:
        logger.info("📡 Manual signal generation triggered via API")
        generate_signals_cycle()
        return {"status": "success", "message": "Signal generation triggered"}
    except Exception as e:
        logger.error(f"❌ Manual signal generation failed: {e}")
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
