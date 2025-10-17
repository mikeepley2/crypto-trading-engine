#!/usr/bin/env python3

import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
import mysql.connector
import time
import random
from datetime import datetime, timedelta
from fastapi import FastAPI, Response
import uvicorn
import threading
from decimal import Decimal
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Prometheus metrics
signals_generated = Counter('signals_generated_total', 'Total signals generated', ['symbol', 'signal_type'])
signal_confidence = Histogram('signal_confidence', 'Signal confidence scores', buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
model_inference_time = Histogram('model_inference_time_seconds', 'ML model inference time')
database_query_time = Histogram('database_query_time_seconds', 'Database query latency')
model_load_status = Gauge('model_load_status', 'ML model load status (1=loaded, 0=not loaded)')
signals_generated_today = Gauge('signals_generated_today', 'Signals generated today')

# Global variables
model = None
scaler = None
app = FastAPI(title='Signal Generator - Real ML Model')

def get_db_connection():
    """Get database connection"""
    try:
        return mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME_PRICES')
        )
    except Exception as e:
        logger.error(f'Database connection error: {e}')
        return None

def load_model():
    """Load the ML model and scaler - NO FALLBACKS ALLOWED"""
    global model, scaler
    
    try:
        # Get model paths from environment or use defaults
        model_path = os.getenv('MODEL_PATH', '/app/models/model.joblib')
        scaler_path = os.getenv('SCALER_PATH', '/app/models/scaler.joblib')
        
        logger.info(f'Loading model from: {model_path}')
        logger.info(f'Loading scaler from: {scaler_path}')
        
        # Verify files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model file not found: {model_path}')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f'Scaler file not found: {scaler_path}')
        
        # Load the model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        logger.info('✅ Real ML model loaded successfully')
        logger.info(f'Model type: {type(model)}')
        logger.info(f'Scaler type: {type(scaler)}')
        
        # Test the model to ensure it works
        test_features = np.random.randn(1, 114)  # 114 features to match database
        test_scaled = scaler.transform(test_features)
        test_prediction = model.predict(test_scaled)
        test_probabilities = model.predict_proba(test_scaled)
        
        logger.info(f'Model test successful - prediction: {test_prediction[0]}, probabilities: {test_probabilities[0]}')
        
        return True
        
    except Exception as e:
        logger.critical(f'CRITICAL: Failed to load ML model: {e}')
        logger.critical('NO FALLBACK MODE - Service cannot start without real model')
        raise e

def get_latest_features(symbol):
    """Get latest features for a symbol from the database"""
    try:
        conn = get_db_connection()
        if not conn:
            return None
        
        cursor = conn.cursor()
        
        # Get latest features from ml_features_materialized table
        cursor.execute('''
            SELECT * FROM ml_features_materialized 
            WHERE symbol = %s 
            ORDER BY timestamp_iso DESC 
            LIMIT 1
        ''', (symbol,))
        
        features = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if features:
            # Convert to numpy array, handling data types properly
            feature_values = []
            for i, value in enumerate(features):
                if i < 3:  # Skip id, symbol, timestamp columns
                    continue
                try:
                    # Convert to float, handling datetime and other types
                    if isinstance(value, (int, float)):
                        feature_values.append(float(value))
                    elif isinstance(value, str):
                        feature_values.append(float(value))
                    else:
                        # Skip non-numeric values
                        feature_values.append(0.0)
                except (ValueError, TypeError):
                    feature_values.append(0.0)
            
            return np.array(feature_values).reshape(1, -1)
        
        return None
        
    except Exception as e:
        logger.error(f'Error getting features for {symbol}: {e}')
        return None

def generate_ml_signal(symbol):
    """Generate signal using ML model"""
    try:
        if model is None or scaler is None:
            logger.error('Model or scaler not loaded')
            return None
        
        # Get latest features
        features = get_latest_features(symbol)
        if features is None:
            logger.warning(f'No features available for {symbol}')
            return None
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Map prediction to signal type
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        signal_type = signal_map.get(prediction, 'HOLD')
        
        # Get confidence from probabilities
        confidence = float(max(probabilities))
        
        return {
            'symbol': symbol,
            'signal_type': signal_type,
            'confidence': confidence,
            'reasoning': f'ML model prediction: {signal_type} with {confidence*100:.2f}% confidence'
        }
        
    except Exception as e:
        logger.error(f'Error generating ML signal for {symbol}: {e}')
        return None

def save_signal_to_db(signal):
    """Save signal to database"""
    try:
        conn = get_db_connection()
        if not conn:
            logger.error('No database connection')
            return False
        
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trading_signals (symbol, signal_type, confidence, threshold, regime, xgboost_confidence, llm_reasoning, timestamp, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
        ''', (signal['symbol'], signal['signal_type'], signal['confidence'], 0.5, 'sideways', signal['confidence'], signal['reasoning']))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f'Signal saved: {signal["symbol"]} {signal["signal_type"]} {signal["confidence"]}')
        return True
    except Exception as e:
        logger.error(f'Error saving signal to database: {e}')
        return False

def signal_generation_loop():
    """Background thread to generate signals using ML model"""
    symbols = ['BTC', 'ETH', 'LINK', 'ADA', 'DOT']
    
    while True:
        try:
            for symbol in symbols:
                signal = generate_ml_signal(symbol)
                if signal and save_signal_to_db(signal):
                    signals_generated.labels(
                        symbol=signal['symbol'],
                        signal_type=signal['signal_type']
                    ).inc()
                    signal_confidence.observe(signal['confidence'])
                    signals_generated_today.inc()
                
                time.sleep(10)  # Wait between symbols
            
            # Wait 30 minutes before next cycle (reduced frequency for better quality)
            time.sleep(1800)
            
        except Exception as e:
            logger.error(f'Error in signal generation loop: {e}')
            time.sleep(60)

@app.get('/health')
async def health_check():
    """Health check endpoint"""
    try:
        if model is None or scaler is None:
            model_load_status.set(0)
            return {'status': 'unhealthy', 'error': 'ML model not loaded'}
        model_load_status.set(1)
        return {'status': 'healthy', 'service': 'signal-generator-real', 'model_loaded': True}
    except Exception as e:
        model_load_status.set(0)
        return {'status': 'unhealthy', 'error': str(e)}

@app.get('/generate-signal')
async def generate_signal_endpoint():
    """Manually generate a signal for testing"""
    symbol = 'BTC'  # Default symbol
    signal = generate_ml_signal(symbol)
    if signal and save_signal_to_db(signal):
        signals_generated.labels(
            symbol=signal['symbol'],
            signal_type=signal['signal_type']
        ).inc()
        signal_confidence.observe(signal['confidence'])
        signals_generated_today.inc()
        return {'success': True, 'signal': signal}
    else:
        return {'success': False, 'error': 'Failed to generate or save signal'}

@app.get('/metrics')
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type='text/plain')

if __name__ == '__main__':
    logger.info('🚀 Starting Signal Generator - Real ML Model')
    
    try:
        # Load the model - NO FALLBACKS
        load_model()
        logger.info('✅ ML model loaded successfully - service ready')
        
        # Start background signal generation
        signal_thread = threading.Thread(target=signal_generation_loop, daemon=True)
        signal_thread.start()
        
        logger.info('✅ Signal generator ready - generating ML signals every 30 minutes')
        
    except Exception as e:
        logger.critical(f'CRITICAL: Cannot start service without ML model: {e}')
        sys.exit(1)
    
    uvicorn.run(app, host='0.0.0.0', port=8025)

