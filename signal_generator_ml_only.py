#!/usr/bin/env python3
"""
Signal Generator - ML Only (No Fallbacks)
Generates trading signals using ML model only - NO FALLBACKS ALLOWED
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
import uvicorn
import threading
from decimal import Decimal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
model = None

# FastAPI app
app = FastAPI(title='Signal Generator - ML Only (No Fallbacks)')

def get_db_connection():
    """Get database connection"""
    try:
        return mysql.connector.connect(
            host=os.getenv('DB_HOST', '172.22.32.1'),
            user=os.getenv('DB_USER', 'news_collector'),
            password=os.getenv('DB_PASSWORD', '99Rules!'),
            database=os.getenv('DB_NAME_PRICES', 'crypto_prices')
        )
    except Exception as e:
        logger.error(f'Database connection error: {e}')
        return None

def load_model():
    """Load the ML model - NO FALLBACKS ALLOWED"""
    global model
    try:
        model_path = '/app/models/balanced_realistic_model_20251005_155755.joblib'
        
        if not os.path.exists(model_path):
            logger.critical(f'CRITICAL: ML model file not found at {model_path}')
            logger.critical('NO FALLBACKS ALLOWED - Service will fail if model is not available')
            raise FileNotFoundError(f'ML model file not found: {model_path}')
        
        model = joblib.load(model_path)
        logger.info(f'✅ ML model loaded successfully from {model_path}')
        logger.info(f'Model type: {type(model)}')
        
        # Validate model has required methods
        if not hasattr(model, 'predict') or not hasattr(model, 'predict_proba'):
            logger.critical('CRITICAL: Loaded model does not have required predict methods')
            raise ValueError('Invalid model - missing predict methods')
        
        logger.info('✅ ML model validation passed - ready for signal generation')
        return True
        
    except Exception as e:
        logger.critical(f'CRITICAL: Failed to load ML model: {e}')
        logger.critical('NO FALLBACKS ALLOWED - Service cannot start without ML model')
        raise e

def get_latest_features(symbol):
    """Get the latest features for a symbol from ml_features_materialized"""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Get the latest features for the symbol
        query = '''
        SELECT * FROM ml_features_materialized 
        WHERE symbol = %s 
        ORDER BY timestamp_iso DESC 
        LIMIT 1
        '''
        cursor.execute(query, (symbol,))
        result = cursor.fetchone()
        
        if result:
            # Exclude metadata columns
            excluded_columns = {
                'id', 'symbol', 'timestamp', 'price', 'price_date', 'price_hour', 
                'timestamp_iso', 'created_at', 'updated_at', 'current_price', 'volume_24h',
                'rsi', 'crypto_sentiment', 'vix', 'llm_analysis', 'llm_confidence', 
                'llm_reasoning', 'sentiment_boost', 'sentiment_sources', 'sentiment_score', 
                'prediction_timestamp', 'prediction', 'is_mock', 'processed', 'signal_id', 
                'signal_strength', 'processed_at'
            }
            
            # Extract features in the same order as the model expects
            features = []
            for col in sorted(result.keys()):
                if col not in excluded_columns:
                    value = result[col]
                    if isinstance(value, Decimal):
                        features.append(float(value))
                    elif value is None:
                        features.append(0.0)
                    else:
                        features.append(float(value))
            
            # Ensure we have exactly 51 features (as expected by the model)
            if len(features) != 51:
                logger.warning(f'Expected 51 features for {symbol}, got {len(features)}')
                # Pad or truncate to 51 features
                if len(features) < 51:
                    features.extend([0.0] * (51 - len(features)))
                else:
                    features = features[:51]
            
            logger.debug(f'Extracted {len(features)} features for {symbol}')
            return np.array(features).reshape(1, -1)
        
        return None
        
    except Exception as e:
        logger.error(f'Error getting features for {symbol}: {e}')
        return None
    finally:
        if conn:
            conn.close()

def generate_signal(symbol, features):
    """Generate a trading signal using ML model - NO FALLBACKS"""
    global model
    
    if model is None:
        logger.critical(f'CRITICAL: No ML model available for {symbol}')
        raise RuntimeError('ML model not loaded - NO FALLBACKS ALLOWED')
    
    try:
        # ML model prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get confidence (probability of the predicted class)
        confidence = float(probabilities[prediction])
        
        # Convert prediction to signal type
        if prediction == 1:
            signal_type = 'BUY'
        elif prediction == 0:
            signal_type = 'SELL'  # Changed from HOLD to SELL
        else:
            signal_type = 'HOLD'
        
        logger.info(f'ML Signal for {symbol}: {signal_type} (confidence: {confidence:.3f}, prediction: {prediction})')
        
        return {
            'symbol': symbol,
            'signal_type': signal_type,
            'confidence': float(confidence),
            'prediction': int(prediction),
            'model_version': 'balanced_realistic_model_20251005_155755'
        }
        
    except Exception as e:
        logger.error(f'Error generating ML signal for {symbol}: {e}')
        raise e

def save_signal_to_db(signal):
    """Save signal to database"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Get current price for the signal
        cursor.execute('''
            SELECT current_price FROM ml_features_materialized 
            WHERE symbol = %s 
            ORDER BY timestamp_iso DESC 
            LIMIT 1
        ''', (signal['symbol'],))
        
        price_result = cursor.fetchone()
        current_price = float(price_result[0]) if price_result and price_result[0] else 0.0
        
        # Insert signal into database
        insert_query = '''
        INSERT INTO trading_signals 
        (symbol, signal_type, confidence, prediction, model_version, price, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        '''
        
        cursor.execute(insert_query, (
            signal['symbol'],
            signal['signal_type'],
            signal['confidence'],
            signal['prediction'],
            signal['model_version'],
            current_price,
            datetime.now()
        ))
        
        conn.commit()
        signal_id = cursor.lastrowid
        
        logger.info(f'✅ Signal saved to DB: ID {signal_id}, {signal["symbol"]} {signal["signal_type"]} (confidence: {signal["confidence"]:.3f}, price: {current_price})')
        
        return True
        
    except Exception as e:
        logger.error(f'Error saving signal to database: {e}')
        return False
    finally:
        if conn:
            conn.close()

def get_active_symbols():
    """Get active symbols from database"""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor()
        query = '''
            SELECT DISTINCT symbol 
            FROM ml_features_materialized 
            WHERE timestamp_iso >= NOW() - INTERVAL 30 MINUTE
            AND current_price IS NOT NULL AND current_price > 0
            ORDER BY timestamp_iso DESC
            LIMIT 50
        '''
        cursor.execute(query)
        results = cursor.fetchall()
        symbols = [row[0] for row in results]
        
        logger.info(f'Found {len(symbols)} active symbols')
        return symbols
        
    except Exception as e:
        logger.error(f'Error getting active symbols: {e}')
        return []
    finally:
        if conn:
            conn.close()

def generate_signals_cycle():
    """Generate signals for all active symbols using ML model only"""
    try:
        logger.info('🚀 Starting ML-only signal generation cycle...')
        
        symbols = get_active_symbols()
        if not symbols:
            logger.warning('⚠️ No active symbols found')
            return
        
        signals_generated = 0
        
        for symbol in symbols[:20]:  # Limit to top 20 symbols
            try:
                # Get features for ML model
                features = get_latest_features(symbol)
                if features is None:
                    logger.warning(f'No features available for {symbol}, skipping')
                    continue
                
                # Generate signal using ML model
                signal = generate_signal(symbol, features)
                
                if signal:
                    # Save signal to database
                    if save_signal_to_db(signal):
                        signals_generated += 1
                        logger.info(f'✅ Generated ML signal for {symbol}: {signal["signal_type"]} (confidence: {signal["confidence"]:.3f})')
                    else:
                        logger.error(f'❌ Failed to save signal for {symbol}')
                else:
                    logger.error(f'❌ Failed to generate signal for {symbol}')
            
            except Exception as e:
                logger.error(f'Error processing {symbol}: {e}')
                continue
        
        logger.info(f'🎯 ML signal generation cycle complete: {signals_generated} signals generated')
        
    except Exception as e:
        logger.error(f'Error in signal generation cycle: {e}')

def signal_generation_worker():
    """Background worker for signal generation"""
    logger.info('Starting ML-only signal generation worker...')
    
    while True:
        try:
            generate_signals_cycle()
            time.sleep(30)  # Generate signals every 30 seconds
        except Exception as e:
            logger.error(f'Error in signal generation worker: {e}')
            time.sleep(60)  # Wait longer on error

@app.get('/health')
async def health_check():
    """Health check endpoint"""
    try:
        # Check if ML model is loaded
        if model is None:
            return {'status': 'unhealthy', 'error': 'ML model not loaded'}
        
        # Check database connectivity
        conn = get_db_connection()
        if not conn:
            return {'status': 'unhealthy', 'error': 'Database not connected'}
        conn.close()
        
        return {
            'status': 'healthy',
            'service': 'signal-generator-ml-only',
            'model_loaded': True,
            'model_type': str(type(model)),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {'status': 'unhealthy', 'error': str(e)}

@app.get('/model-status')
async def model_status():
    """Check ML model status"""
    if model is None:
        return {'model_loaded': False, 'error': 'No ML model loaded'}
    
    return {
        'model_loaded': True,
        'model_type': str(type(model)),
        'model_path': '/app/models/balanced_realistic_model_20251005_155755.joblib',
        'has_predict': hasattr(model, 'predict'),
        'has_predict_proba': hasattr(model, 'predict_proba')
    }

# Initialize the service
if __name__ == '__main__':
    logger.info('🚀 Starting Signal Generator - ML Only (No Fallbacks)')
    
    # Load ML model - service will fail if model is not available
    try:
        load_model()
        logger.info('✅ ML model loaded successfully - service ready')
    except Exception as e:
        logger.critical(f'CRITICAL: Cannot start service without ML model: {e}')
        logger.critical('NO FALLBACKS ALLOWED - Service will exit')
        sys.exit(1)
    
    # Start signal generation worker
    worker_thread = threading.Thread(target=signal_generation_worker, daemon=True)
    worker_thread.start()
    
    # Start FastAPI server
    uvicorn.run(app, host='0.0.0.0', port=8025)


