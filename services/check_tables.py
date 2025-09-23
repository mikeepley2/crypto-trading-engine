#!/usr/bin/env python3
"""
Check database tables and recent signal generation engine activity
"""
import mysql.connector
from datetime import datetime, timedelta

def check_database_tables():
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
        
        # Check available tables
        cursor.execute("SHOW TABLES LIKE '%signal%'")
        tables = cursor.fetchall()
        print("Signal-related tables:")
        for table in tables:
            print(f"  - {table[0]}")
        
        # Check if there's a more recent signals table
        cursor.execute("SHOW TABLES")
        all_tables = cursor.fetchall()
        print("\nAll tables:")
        for table in all_tables:
            print(f"  - {table[0]}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Database check failed: {e}")
        return False

if __name__ == "__main__":
    check_database_tables()
