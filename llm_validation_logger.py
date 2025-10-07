#!/usr/bin/env python3
"""
LLM Validation Logger - Comprehensive logging and monitoring
"""

import mysql.connector
import os
import json
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_db_connection():
    try:
        db_config = {
            'host': os.getenv('DB_HOST', '172.22.32.1'),
            'user': os.getenv('DB_USER', 'news_collector'),
            'password': os.getenv('DB_PASSWORD'),
            'database': os.getenv('DB_NAME_PRICES', 'crypto_prices')
        }
        return mysql.connector.connect(**db_config)
    except Exception as e:
        logger.error(f'Database connection failed: {e}')
        raise Exception(f'Database connection failed: {e}')

def log_llm_validation_attempt(recommendation_id, symbol, signal_type, confidence, amount_usd, 
                              market_context, prompt, response, error=None, processing_time=None):
    """Log LLM validation attempt with full details"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Create validation log entry
        log_data = {
            'recommendation_id': recommendation_id,
            'symbol': symbol,
            'signal_type': signal_type,
            'confidence': confidence,
            'amount_usd': amount_usd,
            'market_context': market_context,
            'prompt': prompt,
            'response': response,
            'error': error,
            'processing_time': processing_time,
            'timestamp': datetime.now()
        }
        
        # Insert into validation_logs table (create if not exists)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS llm_validation_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                recommendation_id INT,
                symbol VARCHAR(10),
                signal_type VARCHAR(10),
                confidence DECIMAL(5,3),
                amount_usd DECIMAL(15,8),
                market_context JSON,
                prompt TEXT,
                response TEXT,
                error TEXT,
                processing_time DECIMAL(8,3),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_recommendation_id (recommendation_id),
                INDEX idx_symbol (symbol),
                INDEX idx_timestamp (timestamp)
            )
        ''')
        
        cursor.execute('''
            INSERT INTO llm_validation_logs 
            (recommendation_id, symbol, signal_type, confidence, amount_usd, 
             market_context, prompt, response, error, processing_time, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
            recommendation_id, symbol, signal_type, confidence, amount_usd,
            json.dumps(market_context), prompt, response, error, processing_time, datetime.now()
        ))
        
        conn.commit()
        logger.info(f'Logged LLM validation attempt for recommendation {recommendation_id}')
        
    except Exception as e:
        logger.error(f'Error logging LLM validation attempt: {e}')
        conn.rollback()
    finally:
        if conn:
            conn.close()

def get_llm_validation_stats():
    """Get comprehensive LLM validation statistics"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Check if logs table exists
        cursor.execute("SHOW TABLES LIKE 'llm_validation_logs'")
        if not cursor.fetchone():
            return {
                'total_attempts': 0,
                'successful_validations': 0,
                'failed_validations': 0,
                'avg_processing_time': 0,
                'recent_errors': []
            }
        
        # Get total attempts
        cursor.execute('SELECT COUNT(*) FROM llm_validation_logs')
        total_attempts = cursor.fetchone()[0]
        
        # Get successful validations (stored in trade_recommendations)
        cursor.execute('SELECT COUNT(*) FROM trade_recommendations WHERE llm_validation IS NOT NULL')
        successful_validations = cursor.fetchone()[0]
        
        # Get failed validations
        cursor.execute('SELECT COUNT(*) FROM llm_validation_logs WHERE error IS NOT NULL')
        failed_validations = cursor.fetchone()[0]
        
        # Get average processing time
        cursor.execute('SELECT AVG(processing_time) FROM llm_validation_logs WHERE processing_time IS NOT NULL')
        avg_processing_time = cursor.fetchone()[0] or 0
        
        # Get recent errors
        cursor.execute('''
            SELECT recommendation_id, symbol, error, timestamp 
            FROM llm_validation_logs 
            WHERE error IS NOT NULL 
            ORDER BY timestamp DESC 
            LIMIT 5
        ''')
        recent_errors = cursor.fetchall()
        
        return {
            'total_attempts': total_attempts,
            'successful_validations': successful_validations,
            'failed_validations': failed_validations,
            'avg_processing_time': round(avg_processing_time, 3),
            'recent_errors': [{'id': r[0], 'symbol': r[1], 'error': r[2], 'timestamp': r[3]} for r in recent_errors]
        }
        
    except Exception as e:
        logger.error(f'Error getting LLM validation stats: {e}')
        return {}
    finally:
        if conn:
            conn.close()

def get_pending_recommendations():
    """Get pending recommendations awaiting LLM validation"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute('''
            SELECT id, symbol, signal_type, confidence, amount_usd, created_at
            FROM trade_recommendations 
            WHERE execution_status = 'PENDING'
            AND (llm_validation IS NULL OR llm_validation = '')
            AND created_at >= NOW() - INTERVAL 30 MINUTE
            ORDER BY created_at DESC
            LIMIT 10
        ''')
        return cursor.fetchall()
    except Exception as e:
        logger.error(f'Error getting pending recommendations: {e}')
        return []
    finally:
        if conn:
            conn.close()

def main():
    """Main function to display LLM validation status"""
    print("=" * 60)
    print("LLM VALIDATION LOGGING AND MONITORING")
    print("=" * 60)
    
    # Get statistics
    stats = get_llm_validation_stats()
    print(f"\n📊 LLM VALIDATION STATISTICS:")
    print(f"   Total Attempts: {stats.get('total_attempts', 0)}")
    print(f"   Successful Validations: {stats.get('successful_validations', 0)}")
    print(f"   Failed Validations: {stats.get('failed_validations', 0)}")
    print(f"   Average Processing Time: {stats.get('avg_processing_time', 0)}s")
    
    # Get pending recommendations
    pending = get_pending_recommendations()
    print(f"\n⏳ PENDING RECOMMENDATIONS ({len(pending)}):")
    for rec in pending[:5]:
        print(f"   ID: {rec['id']}, {rec['symbol']} {rec['signal_type']} (confidence: {rec['confidence']:.3f})")
    
    # Show recent errors
    recent_errors = stats.get('recent_errors', [])
    if recent_errors:
        print(f"\n❌ RECENT ERRORS:")
        for error in recent_errors:
            print(f"   ID: {error['id']}, {error['symbol']} - {error['error'][:100]}...")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
