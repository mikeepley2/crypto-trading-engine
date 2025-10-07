#!/usr/bin/env python3
"""
Patch the existing trade executor to add the missing process_recommendation endpoint
"""

import os
import mysql.connector
import requests
import json

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv('DB_HOST', '172.22.32.1'),
        user=os.getenv('DB_USER', 'news_collector'),
        password=os.getenv('DB_PASSWORD'),
        database='crypto_prices'
    )

def process_recommendation(recommendation_id: int):
    """Process a recommendation - this will be called by the orchestrator"""
    try:
        # Get recommendation from database
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT * FROM trade_recommendations WHERE id = %s', (recommendation_id,))
        rec = cursor.fetchone()
        
        if not rec:
            return {'status': 'error', 'message': 'Recommendation not found'}
        
        # Update status to EXECUTED (mock execution for now)
        cursor.execute('UPDATE trade_recommendations SET execution_status = %s, executed_at = NOW() WHERE id = %s', ('EXECUTED', recommendation_id))
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return {
            'status': 'success',
            'message': f'Recommendation {recommendation_id} processed successfully',
            'recommendation': {
                'id': rec['id'],
                'symbol': rec['symbol'],
                'signal_type': rec['signal_type'],
                'amount_usd': float(rec['amount_usd']),
                'confidence': float(rec['confidence'])
            }
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

# Test the function
if __name__ == '__main__':
    # Test with a recommendation ID
    result = process_recommendation(1)
    print(json.dumps(result, indent=2))
