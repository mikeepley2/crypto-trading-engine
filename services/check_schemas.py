#!/usr/bin/env python3
"""
Check signal table schemas and recent data
"""
import mysql.connector
from datetime import datetime, timedelta

def check_signal_schemas():
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
            'trading_signals'
        ]
        
        for table in signal_tables:
            try:
                print(f"\n--- {table} ---")
                
                # Check schema
                cursor.execute(f"DESCRIBE {table}")
                columns = cursor.fetchall()
                date_columns = []
                for col in columns:
                    if 'time' in col[0].lower() or 'date' in col[0].lower():
                        date_columns.append(col[0])
                
                print(f"Date columns: {date_columns}")
                
                # Get total count
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                total = cursor.fetchone()[0]
                print(f"Total records: {total}")
                
                # Get latest records
                if date_columns:
                    main_date_col = date_columns[0]  # Use first date column
                    cursor.execute(f"""
                        SELECT * FROM {table}
                        ORDER BY {main_date_col} DESC
                        LIMIT 3
                    """)
                    
                    latest = cursor.fetchall()
                    print(f"Latest 3 records:")
                    for record in latest:
                        print(f"  {record}")
                
            except Exception as e:
                print(f"Error checking {table}: {e}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Database check failed: {e}")
        return False

if __name__ == "__main__":
    check_signal_schemas()
