#!/usr/bin/env python3

import mysql.connector
from datetime import datetime

def check_recommendations():
    try:
        conn = mysql.connector.connect(
            host='172.22.32.1',
            user='news_collector',
            password='99Rules!',
            database='crypto_prices'
        )
        
        cursor = conn.cursor()
        
        print('=== TRADE RECOMMENDATIONS CHECK ===')
        print(f'Time: {datetime.now()}')
        print()
        
        # Check recent recommendations
        cursor.execute('''
            SELECT id, symbol, signal_type, confidence, execution_status, llm_validation, created_at 
            FROM trade_recommendations 
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR) 
            ORDER BY created_at DESC 
            LIMIT 20
        ''')
        
        recommendations = cursor.fetchall()
        print(f'Recent recommendations (last hour): {len(recommendations)}')
        print()
        
        for rec in recommendations:
            print(f'ID: {rec[0]}')
            print(f'  Symbol: {rec[1]}, Type: {rec[2]}, Confidence: {rec[3]}')
            print(f'  Status: {rec[4]}, LLM Validated: {rec[5]}')
            print(f'  Created: {rec[6]}')
            print()
        
        # Check execution status breakdown
        cursor.execute('''
            SELECT execution_status, COUNT(*) 
            FROM trade_recommendations 
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
            GROUP BY execution_status
        ''')
        
        status_breakdown = cursor.fetchall()
        print('Status breakdown (last hour):')
        for status, count in status_breakdown:
            print(f'  {status}: {count}')
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    check_recommendations()

