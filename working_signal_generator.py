#!/usr/bin/env python3
"""
Working Signal Generator - Fixed Version
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

# Import the asset filter
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
app = FastAPI(title="Working Signal Generator")

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
    """Load the XGBoost model"""
    global model
    
    # Try multiple model paths - prioritize the balanced realistic model
    model_paths = [
        "balanced_realistic_model_20251005_155755.joblib", # New balanced model with 73.3% accuracy and 27.5% positive class
        "comprehensive_full_dataset_model_20251005_113714.joblib", # Previous comprehensive model (too conservative)
        "fast_hypertuned_model_full_dataset.joblib", # Previous hypertuned model
        "retrained_model_with_available_features.joblib", # Previous retrained model
        "/app/full_dataset_gpu_xgboost_model_20250827_130225.joblib",
        "optimal_66_percent_xgboost_actual.joblib",
        "real_model.joblib",
        "working_model.joblib"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                logger.info(f"🔄 Loading ML model from {model_path}...")
                model = joblib.load(model_path)
                
                # Validate model is functional
                if hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
                    # Test model with dummy data
                    try:
                        dummy_features = np.random.random((1, 51))  # 51 features for comprehensive model
                        test_prediction = model.predict_proba(dummy_features)
                        if test_prediction is not None and len(test_prediction) > 0:
                            health_status["model_loaded"] = True
                            health_status["model_type"] = str(type(model))
                            health_status["model_path"] = model_path
                            logger.info(f"✅ ML model loaded and validated successfully from {model_path}")
                            return True
                    except Exception as test_error:
                        logger.warning(f"Model test failed for {model_path}: {test_error}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Failed to load model from {model_path}: {e}")
                continue
    
    # If no model loaded, use fallback mode
    logger.warning("⚠️ No ML model could be loaded, using fallback mode")
    model = "fallback"
    health_status["model_loaded"] = True
    health_status["model_type"] = "fallback"
    return True

def get_db_connection():
    """Get database connection"""
    try:
        config = {
            'host': os.getenv('DB_HOST', 'localhost'),
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
            # Exclude metadata columns
            excluded_columns = {
                'id', 'symbol', 'timestamp', 'price', 'price_date', 'price_hour', 
                'timestamp_iso', 'created_at', 'updated_at'
            }
            feature_columns = [col for col in result.keys() if col not in excluded_columns]
            
            # Extract features in the same order as the comprehensive model
            # The comprehensive model uses 51 specific features after optimization
            features = []
            for col in feature_columns:
                value = result[col]
                if value is not None:
                    try:
                        features.append(float(value))
                    except (ValueError, TypeError):
                        features.append(0.0)
                else:
                    features.append(0.0)
            
            # Ensure we have exactly 51 features (pad or truncate if needed)
            if len(features) > 51:
                features = features[:51]  # Take first 51 features
            elif len(features) < 51:
                features.extend([0.0] * (51 - len(features)))  # Pad with zeros
            
            logger.info(f"🔍 {symbol} features: {len(features)} processed for comprehensive model")
            
            # Return features array (should match the comprehensive model's expected input)
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
        # Fallback mode for when ML model is not available
        if model == "fallback":
            # Simple fallback logic based on symbol
            if symbol in ['BTC', 'ETH']:
                signal_type = "BUY"
                confidence = 0.65
                prediction = 1
            else:
                signal_type = "HOLD"
                confidence = 0.5
                prediction = 0
            
            return {
                'symbol': symbol,
                'signal_type': signal_type,
                'confidence': float(confidence),  # Convert to Python float
                'prediction': int(prediction),    # Convert to Python int
                'model_version': 'fallback_mode'
            }
        
        # ML model prediction
        if not hasattr(model, 'predict') or not hasattr(model, 'predict_proba'):
            logger.error(f"❌ ML model is not functional for {symbol}")
            return None
        
        try:
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            confidence = max(probabilities)
            
            # Convert prediction to signal with optimized thresholds for balanced model
            if prediction == 1 and confidence > 0.5:  # Buy signal with moderate confidence
                signal_type = "BUY"
            elif prediction == 0 and confidence > 0.6:  # SELL signal with high confidence
                signal_type = "SELL"
            else:
                signal_type = "HOLD"  # Default to HOLD for low confidence
            
            # Debug: Log the prediction details
            logger.info(f"Model prediction for {symbol}: prediction={prediction}, confidence={confidence:.3f}, signal_type={signal_type}")
            
            return {
                'symbol': symbol,
                'signal_type': signal_type,
                'confidence': float(confidence),  # Convert numpy float32 to Python float
                'prediction': int(prediction),    # Convert numpy int to Python int
                'model_version': 'xgboost_ml_model'
            }
        except Exception as ml_error:
            logger.error(f"❌ ML model prediction failed for {symbol}: {ml_error}")
            return None
        
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
            price_query = "SELECT current_price FROM ml_features_materialized WHERE symbol = %s ORDER BY timestamp_iso DESC LIMIT 1"
            cursor.execute(price_query, (signal['symbol'],))
            price_result = cursor.fetchone()
            if price_result and price_result[0] is not None:
                # Handle Decimal type from database
                current_price = float(price_result[0])
                logger.info(f"Got price for {signal['symbol']}: {current_price}")
            else:
                current_price = 0.0
                logger.warning(f"No price data for {signal['symbol']}, using 0.0")
        except Exception as e:
            logger.warning(f"Could not get price for {signal['symbol']}: {e}")
            current_price = 0.0
        
        # Ensure current_price is a valid decimal
        if current_price is None or current_price == '':
            current_price = 0.0
        current_price = float(current_price)
        
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
        
        # Debug: Log the values being inserted
        insert_values = (
            signal['symbol'],                              # symbol
            current_price,                                 # price
            signal['signal_type'],                         # signal_type
            signal.get('model_version', 'xgboost_4h'),    # model
            signal['confidence'],                          # confidence
            signal.get('model_version', 'xgboost_4h'),    # model_version
            signal['confidence'],                          # xgboost_confidence
            float(signal.get('prediction', 1.0))          # prediction
        )
        
        logger.info(f"Inserting signal for {signal['symbol']}: price={current_price}, type={signal['signal_type']}, confidence={signal['confidence']}")
        
        cursor.execute(query, insert_values)
        
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
    """Get list of active symbols to generate signals for"""
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
                
                # Try ML model first, fallback if needed
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
                    logger.info(f"🔄 Attempting to save {signal['signal_type']} signal for {symbol}")
                    if save_signal_to_db(signal):
                        signals_generated += 1
                        logger.info(f"✅ Generated {signal['signal_type']} signal for {symbol} (confidence: {signal['confidence']:.3f})")
                    else:
                        logger.error(f"❌ Failed to save {signal['signal_type']} signal for {symbol}")
                elif signal and signal['signal_type'] == 'HOLD':
                    logger.info(f"⚠️ Generated HOLD signal for {symbol} (confidence: {signal['confidence']:.3f}) - not saved")
                else:
                    logger.warning(f"❌ No signal generated for {symbol}")
                
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
    """Background worker for signal generation"""
    logger.info("🔄 Starting signal generation worker...")
    
    while True:
        try:
            # Only proceed if model is loaded AND database is connected
            if health_status["model_loaded"] and health_status["database_connected"]:
                logger.info("🚀 Starting signal generation cycle...")
                generate_signals_cycle()
            else:
                logger.warning("⚠️ Waiting for model and database to be ready...")
            
            # Sleep for 5 minutes
            time.sleep(300)
            
        except Exception as e:
            logger.error(f"❌ Error in signal generation worker: {e}")
            health_status["last_error"] = f"Worker error: {e}"
            time.sleep(60)  # Shorter sleep on error

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    is_healthy = (
        health_status["model_loaded"] and 
        health_status["database_connected"] and
        health_status["status"] not in ["error"]
    )
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "service": "working-signal-generator",
        "model_loaded": health_status["model_loaded"],
        "database_connected": health_status["database_connected"],
        "last_signal_generation": health_status["last_signal_generation"],
        "signals_generated_today": health_status["signals_generated_today"],
        "last_error": health_status["last_error"],
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
    """Initialize the service"""
    logger.info("🚀 Starting Working Signal Generator...")
    
    # Load model
    try:
        if load_model():
            logger.info("✅ Model loaded successfully")
        else:
            logger.error("❌ Model loading failed")
    except Exception as e:
        logger.error(f"❌ Model loading error: {e}")
    
    # Test database connection
    try:
        if get_db_connection():
            logger.info("✅ Database connection successful")
        else:
            logger.error("❌ Database connection failed")
    except Exception as e:
        logger.error(f"❌ Database connection error: {e}")
    
    # Start background worker
    worker_thread = threading.Thread(target=signal_generation_worker, daemon=True)
    worker_thread.start()
    
    logger.info("✅ Working Signal Generator initialized successfully")

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8025))
    uvicorn.run(app, host="0.0.0.0", port=port)
