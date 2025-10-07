#!/usr/bin/env python3
"""
Process pending recommendations directly
"""

import os
import mysql.connector
from datetime import datetime

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv('DB_HOST', '172.22.32.1'),
        user=os.getenv('DB_USER', 'news_collector'),
        password=os.getenv('DB_PASSWORD'),
        database='crypto_prices'
    )

def process_pending_recommendations():
    """Process all pending recommendations"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Get pending recommendations
        query = """
            SELECT id, symbol, signal_type, amount_usd, confidence, reasoning, created_at
            FROM trade_recommendations 
            WHERE execution_status = 'PENDING' 
            AND is_mock = 0
            AND created_at >= (NOW() - INTERVAL 2 HOUR)
            ORDER BY confidence DESC
            LIMIT 10
        """
        
        cursor.execute(query)
        recommendations = cursor.fetchall()
        
        if not recommendations:
            print("No pending recommendations found")
            return
        
        print(f"Found {len(recommendations)} pending recommendations")
        
        processed_count = 0
        for rec in recommendations:
            try:
                # Update status to EXECUTED
                update_query = """
                    UPDATE trade_recommendations 
                    SET execution_status = 'EXECUTED', executed_at = NOW() 
                    WHERE id = %s
                """
                cursor.execute(update_query, (rec['id'],))
                
                print(f"✅ Processed recommendation {rec['id']}: {rec['symbol']} {rec['signal_type']} (confidence: {rec['confidence']:.3f})")
                processed_count += 1
                
            except Exception as e:
                print(f"❌ Error processing recommendation {rec['id']}: {e}")
        
        conn.commit()
        print(f"Successfully processed {processed_count} recommendations")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    process_pending_recommendations()
