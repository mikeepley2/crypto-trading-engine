#!/usr/bin/env python3
"""
Trace Signal Flow Script
Traces a new signal all the way through the pipeline
"""

import mysql.connector
from datetime import datetime, timedelta

def trace_signal_flow():
    print('=' * 60)
    print('TRACING NEW SIGNAL THROUGH PIPELINE')
    print('=' * 60)
    
    # Connect to database
    conn = mysql.connector.connect(
        host='172.22.32.1',
        user='news_collector',
        password='99Rules!',
        database='crypto_prices'
    )
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Get the most recent signal
        cursor.execute('''
            SELECT id, symbol, signal_type, confidence, timestamp, model_version
            FROM trading_signals 
            WHERE timestamp >= NOW() - INTERVAL 5 MINUTE
            ORDER BY timestamp DESC
            LIMIT 1
        ''')
        recent_signal = cursor.fetchone()
        
        if recent_signal:
            print(f'\n📊 STEP 1: NEW SIGNAL GENERATED')
            print(f'   Signal ID: {recent_signal["id"]}')
            print(f'   Symbol: {recent_signal["symbol"]}')
            print(f'   Type: {recent_signal["signal_type"]}')
            print(f'   Confidence: {recent_signal["confidence"]:.3f}')
            print(f'   Timestamp: {recent_signal["timestamp"]}')
            print(f'   Model: {recent_signal["model_version"]}')
            
            # Check if recommendation was created for this signal
            cursor.execute('''
                SELECT id, symbol, signal_type, confidence, execution_status, created_at, reasoning
                FROM trade_recommendations 
                WHERE signal_id = %s
                ORDER BY created_at DESC
                LIMIT 1
            ''', (recent_signal['id'],))
            recommendation = cursor.fetchone()
            
            if recommendation:
                print(f'\n📋 STEP 2: RECOMMENDATION CREATED')
                print(f'   Recommendation ID: {recommendation["id"]}')
                print(f'   Symbol: {recommendation["symbol"]}')
                print(f'   Type: {recommendation["signal_type"]}')
                print(f'   Confidence: {recommendation["confidence"]:.3f}')
                print(f'   Status: {recommendation["execution_status"]}')
                print(f'   Created: {recommendation["created_at"]}')
                print(f'   Reasoning: {recommendation["reasoning"]}')
                
                # Check if trade was executed
                if recommendation['execution_status'] == 'EXECUTED':
                    cursor.execute('''
                        SELECT executed_at, amount_usd, entry_price
                        FROM trade_recommendations 
                        WHERE id = %s
                    ''', (recommendation['id'],))
                    execution_details = cursor.fetchone()
                    
                    print(f'\n💰 STEP 3: TRADE EXECUTED')
                    print(f'   Recommendation ID: {recommendation["id"]}')
                    print(f'   Executed At: {execution_details["executed_at"]}')
                    print(f'   Amount USD: ${execution_details["amount_usd"]}')
                    print(f'   Entry Price: ${execution_details["entry_price"]}')
                    
                    print(f'\n✅ COMPLETE PIPELINE TRACE SUCCESSFUL')
                    print(f'   Signal {recent_signal["id"]} → Recommendation {recommendation["id"]} → Trade Executed')
                else:
                    print(f'\n⏳ STEP 3: TRADE PENDING')
                    print(f'   Recommendation ID: {recommendation["id"]}')
                    print(f'   Status: {recommendation["execution_status"]}')
                    print(f'   Waiting for trade orchestrator to process...')
                    
                    # Check if it's in the queue
                    cursor.execute('''
                        SELECT COUNT(*) as position
                        FROM trade_recommendations 
                        WHERE execution_status = 'PENDING'
                        AND created_at <= %s
                    ''', (recommendation['created_at'],))
                    queue_position = cursor.fetchone()['position']
                    print(f'   Queue Position: {queue_position}')
            else:
                print(f'\n❌ STEP 2: NO RECOMMENDATION FOUND')
                print(f'   Signal ID {recent_signal["id"]} has not been converted to recommendation yet')
                print(f'   This may indicate the signal bridge is not processing signals')
        else:
            print('\n❌ NO RECENT SIGNALS FOUND')
            print('   No signals generated in the last 5 minutes')
            
            # Check for any recent signals
            cursor.execute('''
                SELECT id, symbol, signal_type, confidence, timestamp
                FROM trading_signals 
                WHERE timestamp >= NOW() - INTERVAL 1 HOUR
                ORDER BY timestamp DESC
                LIMIT 5
            ''')
            recent_signals = cursor.fetchall()
            
            if recent_signals:
                print('\n📊 Most recent signals (last hour):')
                for signal in recent_signals:
                    print(f'   ID: {signal["id"]}, {signal["symbol"]} {signal["signal_type"]} (confidence: {signal["confidence"]:.3f}) - {signal["timestamp"]}')
            else:
                print('\n❌ No signals found in the last hour')
                print('   Signal generator may not be working')
        
        # Show pipeline statistics
        print(f'\n📈 PIPELINE STATISTICS')
        print('-' * 30)
        
        # Recent activity counts
        cursor.execute('''
            SELECT COUNT(*) as count
            FROM trading_signals 
            WHERE timestamp >= NOW() - INTERVAL 10 MINUTE
        ''')
        signals_10min = cursor.fetchone()['count']
        
        cursor.execute('''
            SELECT COUNT(*) as count
            FROM trade_recommendations 
            WHERE created_at >= NOW() - INTERVAL 10 MINUTE
        ''')
        recommendations_10min = cursor.fetchone()['count']
        
        cursor.execute('''
            SELECT COUNT(*) as count
            FROM trade_recommendations 
            WHERE executed_at >= NOW() - INTERVAL 10 MINUTE
            AND execution_status = 'EXECUTED'
        ''')
        executions_10min = cursor.fetchone()['count']
        
        print(f'   Signals (last 10 min): {signals_10min}')
        print(f'   Recommendations (last 10 min): {recommendations_10min}')
        print(f'   Executions (last 10 min): {executions_10min}')
        
        # Pending queue
        cursor.execute('''
            SELECT COUNT(*) as count
            FROM trade_recommendations 
            WHERE execution_status = 'PENDING'
        ''')
        pending_count = cursor.fetchone()['count']
        print(f'   Pending recommendations: {pending_count}')
        
    except Exception as e:
        print(f'Error during signal trace: {e}')
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    trace_signal_flow()
