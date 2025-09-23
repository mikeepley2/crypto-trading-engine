#!/usr/bin/env python3
"""
Quick Database Check for Fresh Recommendations
"""

import mysql.connector
from datetime import datetime, timedelta

def check_database():
    """Check what's actually in the database"""
    
    # Database configuration
    db_config = {
        'host': 'host.docker.internal',
        'user': 'news_collector',
        'password': '99Rules!',
        'database': 'crypto_transactions'
    }
    
    try:
        # Connect to database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        print("🔍 CHECKING TRADE RECOMMENDATIONS TABLE...")
        
        # Check latest recommendations
        cursor.execute("""
            SELECT id, generated_at, symbol, action, confidence, entry_price, reasoning, is_mock
            FROM trade_recommendations 
            ORDER BY generated_at DESC 
            LIMIT 10
        """)
        
        recommendations = cursor.fetchall()
        
        print(f"\n📊 Found {len(recommendations)} recent recommendations:")
        print("-" * 80)
        
        current_time = datetime.now()
        one_hour_ago = current_time - timedelta(hours=1)
        
        fresh_count = 0
        for rec in recommendations:
            generated_at = rec['generated_at']
            
            # Check if it's fresh (within 1 hour)
            is_fresh = generated_at >= one_hour_ago if generated_at else False
            fresh_indicator = "🟢 FRESH" if is_fresh else "🔴 OLD"
            
            if is_fresh:
                fresh_count += 1
                
            print(f"{fresh_indicator} | ID: {rec['id']} | {generated_at} | {rec['symbol']} {rec['action']} @ ${rec['entry_price']} | Conf: {rec['confidence']}")
        
        print("-" * 80)
        print(f"🎯 Fresh recommendations (within 1h): {fresh_count}")
        print(f"⏰ Current time: {current_time}")
        print(f"🕐 One hour ago: {one_hour_ago}")
        
        # Also check crypto_prices database for trading signals
        cursor.close()
        conn.close()
        
        # Check crypto_prices database
        db_config['database'] = 'crypto_prices'
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        print("\n🔍 CHECKING TRADING SIGNALS TABLE...")
        
        cursor.execute("""
            SELECT id, timestamp, symbol, signal, confidence, signal_strength
            FROM trading_signals 
            ORDER BY timestamp DESC 
            LIMIT 10
        """)
        
        signals = cursor.fetchall()
        print(f"\n📊 Found {len(signals)} recent signals:")
        print("-" * 80)
        
        for signal in signals:
            timestamp = signal['timestamp']
            is_fresh = timestamp >= one_hour_ago if timestamp else False
            fresh_indicator = "🟢 FRESH" if is_fresh else "🔴 OLD"
            
            print(f"{fresh_indicator} | ID: {signal['id']} | {timestamp} | {signal['symbol']} {signal['signal']} | Conf: {signal['confidence']}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")

if __name__ == "__main__":
    check_database()
