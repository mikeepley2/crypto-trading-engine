#!/usr/bin/env python3

import mysql.connector
import os
from datetime import datetime, timedelta

def check_signals():
    try:
        conn = mysql.connector.connect(
            host='172.22.32.1',
            user='news_collector',
            password='99Rules!',
            database='crypto_prices'
        )
        
        cursor = conn.cursor()
        
        print('=== SIGNAL STATUS CHECK ===')
        print(f'Test Time: {datetime.now()}')
        print()
        
        # Check recent signals
        cursor.execute('''
            SELECT id, symbol, signal_type, confidence, processed, created_at 
            FROM trading_signals 
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR) 
            ORDER BY created_at DESC 
            LIMIT 10
        ''')
        
        signals = cursor.fetchall()
        print(f'Recent signals (last hour): {len(signals)}')
        
        for signal in signals:
            print(f'  ID: {signal[0]}, Symbol: {signal[1]}, Type: {signal[2]}, Confidence: {signal[3]}, Processed: {signal[4]}, Time: {signal[5]}')
        
        print()
        
        # Check unprocessed signals
        cursor.execute('''
            SELECT COUNT(*) 
            FROM trading_signals 
            WHERE processed = 0 
            AND created_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
        ''')
        
        unprocessed_count = cursor.fetchone()[0]
        print(f'Unprocessed signals (last hour): {unprocessed_count}')
        
        # Check trade recommendations
        cursor.execute('''
            SELECT COUNT(*) 
            FROM trade_recommendations 
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
        ''')
        
        recommendations_count = cursor.fetchone()[0]
        print(f'Trade recommendations (last hour): {recommendations_count}')
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    check_signals()
