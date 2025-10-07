#!/usr/bin/env python3
"""
Real-time Pipeline Flow Monitor
Shows signals flowing through the complete pipeline in real-time
"""

import mysql.connector
import time
from datetime import datetime

def monitor_pipeline_flow():
    print('=' * 80)
    print('REAL-TIME PIPELINE FLOW MONITOR')
    print('=' * 80)
    print('Monitoring signal flow through the complete pipeline...')
    print('Press Ctrl+C to stop')
    print('=' * 80)
    
    # Connect to database
    conn = mysql.connector.connect(
        host='172.22.32.1',
        user='news_collector',
        password='99Rules!',
        database='crypto_prices'
    )
    cursor = conn.cursor(dictionary=True)
    
    # Track processed signals to avoid duplicates
    processed_signals = set()
    processed_recommendations = set()
    processed_executions = set()
    
    try:
        while True:
            current_time = datetime.now().strftime('%H:%M:%S')
            
            # Check for new signals
            cursor.execute('''
                SELECT id, symbol, signal_type, confidence, timestamp
                FROM trading_signals 
                WHERE timestamp >= NOW() - INTERVAL 2 MINUTE
                ORDER BY timestamp DESC
                LIMIT 5
            ''')
            recent_signals = cursor.fetchall()
            
            for signal in recent_signals:
                if signal['id'] not in processed_signals:
                    print(f'[{current_time}] 📊 NEW SIGNAL: ID {signal["id"]} - {signal["symbol"]} {signal["signal_type"]} (confidence: {signal["confidence"]:.3f})')
                    processed_signals.add(signal['id'])
            
            # Check for new recommendations
            cursor.execute('''
                SELECT id, signal_id, symbol, signal_type, confidence, created_at
                FROM trade_recommendations 
                WHERE created_at >= NOW() - INTERVAL 2 MINUTE
                ORDER BY created_at DESC
                LIMIT 5
            ''')
            recent_recommendations = cursor.fetchall()
            
            for rec in recent_recommendations:
                if rec['id'] not in processed_recommendations:
                    print(f'[{current_time}] 📋 NEW RECOMMENDATION: ID {rec["id"]} (from signal {rec["signal_id"]}) - {rec["symbol"]} {rec["signal_type"]} (confidence: {rec["confidence"]:.3f})')
                    processed_recommendations.add(rec['id'])
            
            # Check for new executions
            cursor.execute('''
                SELECT id, symbol, signal_type, executed_at
                FROM trade_recommendations 
                WHERE executed_at >= NOW() - INTERVAL 2 MINUTE
                AND execution_status = 'EXECUTED'
                ORDER BY executed_at DESC
                LIMIT 5
            ''')
            recent_executions = cursor.fetchall()
            
            for exec_rec in recent_executions:
                if exec_rec['id'] not in processed_executions:
                    print(f'[{current_time}] 💰 TRADE EXECUTED: ID {exec_rec["id"]} - {exec_rec["symbol"]} {exec_rec["signal_type"]} at {exec_rec["executed_at"]}')
                    processed_executions.add(exec_rec['id'])
            
            # Show pipeline statistics every 30 seconds
            if int(time.time()) % 30 == 0:
                cursor.execute('SELECT COUNT(*) as count FROM trade_recommendations WHERE execution_status = "PENDING"')
                pending_count = cursor.fetchone()['count']
                print(f'[{current_time}] 📈 Pipeline Status: {pending_count} pending recommendations')
            
            time.sleep(5)  # Check every 5 seconds
            
    except KeyboardInterrupt:
        print('\n\nMonitoring stopped by user')
    except Exception as e:
        print(f'Error during monitoring: {e}')
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    monitor_pipeline_flow()
