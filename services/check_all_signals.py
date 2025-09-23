#!/usr/bin/env python3
"""
Check all signal tables for recent activity
"""
import mysql.connector
from datetime import datetime, timedelta

def check_all_signal_tables():
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
        
        signal_tables = [
            'trading_signals',
            'trading_signals_v2',
            'trading_signals',
            'trading_signals',
            'real_time_sentiment_signals'
        ]
        
        for table in signal_tables:
            try:
                # Check recent signals (last 10 minutes)
                cursor.execute(f"""
                    SELECT COUNT(*) as count 
                    FROM {table} 
                    WHERE created_at > %s OR timestamp > %s OR updated_at > %s
                """, (datetime.now() - timedelta(minutes=10), datetime.now() - timedelta(minutes=10), datetime.now() - timedelta(minutes=10)))
                
                count = cursor.fetchone()[0]
                print(f"{table}: {count} recent signals")
                
                if count > 0:
                    # Show latest entries
                    cursor.execute(f"""
                        SELECT * FROM {table}
                        ORDER BY COALESCE(created_at, timestamp, updated_at) DESC
                        LIMIT 3
                    """)
                    
                    latest = cursor.fetchall()
                    print(f"  Latest entries:")
                    for entry in latest:
                        print(f"    {entry}")
                    print()
                    
            except Exception as e:
                print(f"Error checking {table}: {e}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Database check failed: {e}")
        return False

if __name__ == "__main__":
    check_all_signal_tables()
