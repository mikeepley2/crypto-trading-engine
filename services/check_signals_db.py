#!/usr/bin/env python3
"""
Quick database check for trading signals
"""
import mysql.connector
from datetime import datetime, timedelta

def check_trading_signals():
    try:
        # Connect to database
        db_config = {
            'host': 'host.docker.internal',
            'user': 'news_collector',
            'password': '99Rules!',
            'database': 'crypto_prices'
        }
        
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # Check recent signals
        cursor.execute("""
            SELECT COUNT(*) as signal_count 
            FROM trading_signals 
            WHERE created_at > %s
        """, (datetime.now() - timedelta(minutes=5),))
        
        recent_count = cursor.fetchone()[0]
        print(f"Recent signals (last 5 minutes): {recent_count}")
        
        # Check latest signals
        cursor.execute("""
            SELECT symbol, signal_type, confidence, price, created_at 
            FROM trading_signals 
            ORDER BY created_at DESC 
            LIMIT 10
        """)
        
        signals = cursor.fetchall()
        print("\nLatest 10 signals:")
        for signal in signals:
            print(f"  {signal[4]} - {signal[0]}: {signal[1]} (confidence: {signal[2]:.3f}, price: ${signal[3]:.2f})")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Database check failed: {e}")
        return False

if __name__ == "__main__":
    check_trading_signals()
