#!/usr/bin/env python3
"""
Check executed recommendations
"""

import os
import mysql.connector

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv('DB_HOST', '172.22.32.1'),
        user=os.getenv('DB_USER', 'news_collector'),
        password=os.getenv('DB_PASSWORD'),
        database='crypto_prices'
    )

def check_executed_recommendations():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Check executed recommendations
        query = """
            SELECT id, symbol, signal_type, confidence, execution_status, executed_at, created_at
            FROM trade_recommendations 
            WHERE execution_status = 'EXECUTED'
            ORDER BY executed_at DESC
            LIMIT 5
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        print('=== RECENTLY EXECUTED RECOMMENDATIONS ===')
        for rec in results:
            print(f'ID: {rec["id"]}, Symbol: {rec["symbol"]}, Type: {rec["signal_type"]}, Confidence: {rec["confidence"]:.3f}, Executed: {rec["executed_at"]}')
        
        # Check total counts
        cursor.execute("SELECT COUNT(*) as total FROM trade_recommendations")
        total = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(*) as executed FROM trade_recommendations WHERE execution_status = 'EXECUTED'")
        executed = cursor.fetchone()['executed']
        
        cursor.execute("SELECT COUNT(*) as pending FROM trade_recommendations WHERE execution_status = 'PENDING'")
        pending = cursor.fetchone()['pending']
        
        print(f'\n=== SUMMARY ===')
        print(f'Total recommendations: {total}')
        print(f'Executed: {executed}')
        print(f'Pending: {pending}')
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    check_executed_recommendations()
