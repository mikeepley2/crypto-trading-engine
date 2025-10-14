#!/usr/bin/env python3
"""
Script to force reload the new balanced model in the signal generator
"""

import os
import sys
import logging
import mysql.connector
import joblib
import numpy as np
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def generate_test_signals():
    """Generate test signals using the new balanced model"""
    logger.info("🧪 Generating test signals with new balanced model...")
    
    # Load the new balanced model
    model_path = '/app/models/balanced_retrained_model_20251008_210451.joblib'
    scaler_path = '/app/models/balanced_retrained_scaler_20251008_210451.joblib'
    features_path = '/app/models/balanced_retrained_features_20251008_210451.json'
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
    
    # Load model, scaler, and feature names
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(features_path, 'r') as f:
        feature_names = json.load(f)
    
    logger.info(f"✅ Loaded balanced model with {len(feature_names)} features")
    
    # Get real data and generate signals
    conn = get_db_connection()
    if not conn:
        logger.error("Cannot connect to database")
        return False
    
    try:
        cursor = conn.cursor()
        
        # Get a few symbols for testing
        symbols = ['BTC', 'ETH', 'XRP', 'ADA', 'SOL', 'DOT', 'MATIC', 'LINK', 'UNI', 'LTC']
        
        signals_generated = 0
        buy_signals = 0
        sell_signals = 0
        hold_signals = 0
        
        for symbol in symbols:
            cursor.execute('''
                SELECT * FROM ml_features_materialized 
                WHERE symbol = %s 
                ORDER BY timestamp_iso DESC 
                LIMIT 1
            ''', (symbol,))
            
            result = cursor.fetchone()
            if result:
                # Extract features using the same logic as the signal generator
                features = []
                for feature_name in feature_names:
                    # Find the column index
                    col_index = None
                    for i, col in enumerate(cursor.description):
                        if col[0] == feature_name:
                            col_index = i
                            break
                    
                    if col_index is not None and result[col_index] is not None:
                        try:
                            features.append(float(result[col_index]))
                        except (ValueError, TypeError):
                            features.append(0.0)
                    else:
                        features.append(0.0)
                
                # Make prediction
                features_array = np.array(features).reshape(1, -1)
                features_scaled = scaler.transform(features_array)
                
                prediction = model.predict(features_scaled)[0]
                probabilities = model.predict_proba(features_scaled)[0]
                confidence = max(probabilities)
                
                # Determine signal
                if prediction == 1 and confidence > 0.5:
                    signal_type = 'BUY'
                    buy_signals += 1
                elif prediction == 0 and confidence > 0.6:
                    signal_type = 'SELL'
                    sell_signals += 1
                else:
                    signal_type = 'HOLD'
                    hold_signals += 1
                
                logger.info(f"{symbol}: prediction={prediction}, confidence={confidence:.3f}, signal={signal_type}")
                
                # Save signal to database
                if signal_type != 'HOLD':
                    try:
                        # Get current price
                        cursor.execute('''
                            SELECT current_price FROM ml_features_materialized
                            WHERE symbol = %s
                            ORDER BY timestamp_iso DESC
                            LIMIT 1
                        ''', (symbol,))
                        
                        price_result = cursor.fetchone()
                        current_price = float(price_result[0]) if price_result and price_result[0] else 0.0
                        
                        # Insert signal into database
                        cursor.execute('''
                            INSERT INTO trading_signals
                            (symbol, signal_type, confidence, prediction, model_version, price, timestamp)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ''', (
                            symbol,
                            signal_type,
                            confidence,
                            prediction,
                            'balanced_retrained_model_20251008_210451',
                            current_price,
                            datetime.now()
                        ))
                        
                        conn.commit()
                        signals_generated += 1
                        logger.info(f"✅ Saved {signal_type} signal for {symbol}")
                        
                    except Exception as e:
                        logger.error(f"Error saving signal for {symbol}: {e}")
        
        conn.close()
        
        logger.info(f"🎯 Signal generation complete:")
        logger.info(f"   Total signals: {signals_generated}")
        logger.info(f"   BUY signals: {buy_signals}")
        logger.info(f"   SELL signals: {sell_signals}")
        logger.info(f"   HOLD signals: {hold_signals}")
        
        if sell_signals > 0:
            logger.info("🎉 SUCCESS: Generated SELL signals with new balanced model!")
            return True
        else:
            logger.warning("⚠️ WARNING: No SELL signals generated")
            return False
            
    except Exception as e:
        logger.error(f"Error generating signals: {e}")
        return False

if __name__ == "__main__":
    success = generate_test_signals()
    sys.exit(0 if success else 1)


