#!/usr/bin/env python3
"""
Complete Pipeline Test Script
Tests the entire signal-to-trade flow
"""

import os
import time
import mysql.connector
from datetime import datetime, timedelta

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv('DB_HOST', '172.22.32.1'),
        user=os.getenv('DB_USER', 'news_collector'),
        password=os.getenv('DB_PASSWORD'),
        database='crypto_prices'
    )

def test_pipeline():
    print("=" * 60)
    print("CRYPTO TRADING SYSTEM - COMPLETE PIPELINE TEST")
    print("=" * 60)
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # 1. Check recent signals
        print("\n1. RECENT SIGNALS")
        print("-" * 30)
        cursor.execute("""
            SELECT id, symbol, signal_type, confidence, timestamp, model_version
            FROM trading_signals 
            WHERE timestamp >= NOW() - INTERVAL 1 HOUR
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        signals = cursor.fetchall()
        
        if signals:
            print(f"Found {len(signals)} recent signals:")
            for signal in signals:
                print(f"  ID: {signal['id']}, {signal['symbol']} {signal['signal_type']} (confidence: {signal['confidence']:.3f}) - {signal['timestamp']}")
        else:
            print("No recent signals found")
        
        # 2. Check recent recommendations
        print("\n2. RECENT RECOMMENDATIONS")
        print("-" * 30)
        cursor.execute("""
            SELECT id, symbol, signal_type, confidence, execution_status, created_at, executed_at
            FROM trade_recommendations 
            WHERE created_at >= NOW() - INTERVAL 1 HOUR
            ORDER BY created_at DESC
            LIMIT 10
        """)
        recommendations = cursor.fetchall()
        
        if recommendations:
            print(f"Found {len(recommendations)} recent recommendations:")
            for rec in recommendations:
                status_emoji = "✅" if rec['execution_status'] == 'EXECUTED' else "⏳" if rec['execution_status'] == 'PENDING' else "❌"
                print(f"  {status_emoji} ID: {rec['id']}, {rec['symbol']} {rec['signal_type']} (confidence: {rec['confidence']:.3f}) - {rec['execution_status']}")
        else:
            print("No recent recommendations found")
        
        # 3. Check executed trades
        print("\n3. EXECUTED TRADES")
        print("-" * 30)
        cursor.execute("""
            SELECT COUNT(*) as total_executed
            FROM trade_recommendations 
            WHERE execution_status = 'EXECUTED'
            AND executed_at >= NOW() - INTERVAL 1 HOUR
        """)
        executed_count = cursor.fetchone()['total_executed']
        print(f"Trades executed in last hour: {executed_count}")
        
        # 4. Check pending recommendations
        print("\n4. PENDING RECOMMENDATIONS")
        print("-" * 30)
        cursor.execute("""
            SELECT COUNT(*) as total_pending
            FROM trade_recommendations 
            WHERE execution_status = 'PENDING'
        """)
        pending_count = cursor.fetchone()['total_pending']
        print(f"Pending recommendations: {pending_count}")
        
        # 5. Pipeline flow analysis
        print("\n5. PIPELINE FLOW ANALYSIS")
        print("-" * 30)
        
        # Check signal generation rate
        cursor.execute("""
            SELECT COUNT(*) as signals_last_hour
            FROM trading_signals 
            WHERE timestamp >= NOW() - INTERVAL 1 HOUR
        """)
        signals_last_hour = cursor.fetchone()['signals_last_hour']
        print(f"Signals generated (last hour): {signals_last_hour}")
        
        # Check recommendation creation rate
        cursor.execute("""
            SELECT COUNT(*) as recommendations_last_hour
            FROM trade_recommendations 
            WHERE created_at >= NOW() - INTERVAL 1 HOUR
        """)
        recommendations_last_hour = cursor.fetchone()['recommendations_last_hour']
        print(f"Recommendations created (last hour): {recommendations_last_hour}")
        
        # Check execution rate
        cursor.execute("""
            SELECT COUNT(*) as executions_last_hour
            FROM trade_recommendations 
            WHERE executed_at >= NOW() - INTERVAL 1 HOUR
            AND execution_status = 'EXECUTED'
        """)
        executions_last_hour = cursor.fetchone()['executions_last_hour']
        print(f"Trades executed (last hour): {executions_last_hour}")
        
        # 6. System health summary
        print("\n6. SYSTEM HEALTH SUMMARY")
        print("-" * 30)
        
        if signals_last_hour > 0:
            print("✅ Signal Generation: ACTIVE")
        else:
            print("⚠️ Signal Generation: NO RECENT ACTIVITY")
        
        if recommendations_last_hour > 0:
            print("✅ Signal Bridge: ACTIVE")
        else:
            print("⚠️ Signal Bridge: NO RECENT ACTIVITY")
        
        if executions_last_hour > 0:
            print("✅ Trade Execution: ACTIVE")
        else:
            print("⚠️ Trade Execution: NO RECENT ACTIVITY")
        
        if pending_count > 0:
            print(f"⏳ Pending Trades: {pending_count}")
        
        # 7. Performance metrics
        print("\n7. PERFORMANCE METRICS")
        print("-" * 30)
        
        if recommendations_last_hour > 0 and signals_last_hour > 0:
            conversion_rate = (recommendations_last_hour / signals_last_hour) * 100
            print(f"Signal → Recommendation conversion rate: {conversion_rate:.1f}%")
        
        if executions_last_hour > 0 and recommendations_last_hour > 0:
            execution_rate = (executions_last_hour / recommendations_last_hour) * 100
            print(f"Recommendation → Execution rate: {execution_rate:.1f}%")
        
        # 8. Recent activity timeline
        print("\n8. RECENT ACTIVITY TIMELINE")
        print("-" * 30)
        
        cursor.execute("""
            SELECT 'SIGNAL' as type, id, symbol, signal_type, confidence, timestamp as event_time
            FROM trading_signals 
            WHERE timestamp >= NOW() - INTERVAL 30 MINUTE
            UNION ALL
            SELECT 'RECOMMENDATION' as type, id, symbol, signal_type, confidence, created_at as event_time
            FROM trade_recommendations 
            WHERE created_at >= NOW() - INTERVAL 30 MINUTE
            UNION ALL
            SELECT 'EXECUTION' as type, id, symbol, signal_type, confidence, executed_at as event_time
            FROM trade_recommendations 
            WHERE executed_at >= NOW() - INTERVAL 30 MINUTE
            AND execution_status = 'EXECUTED'
            ORDER BY event_time DESC
            LIMIT 15
        """)
        
        timeline = cursor.fetchall()
        if timeline:
            print("Last 15 events:")
            for event in timeline:
                emoji = "📊" if event['type'] == 'SIGNAL' else "📋" if event['type'] == 'RECOMMENDATION' else "💰"
                print(f"  {emoji} {event['type']}: {event['symbol']} {event['signal_type']} (confidence: {event['confidence']:.3f}) - {event['event_time']}")
        else:
            print("No recent activity in last 30 minutes")
        
        print("\n" + "=" * 60)
        print("PIPELINE TEST COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during pipeline test: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    test_pipeline()
