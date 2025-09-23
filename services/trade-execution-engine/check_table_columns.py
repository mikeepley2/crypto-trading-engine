#!/usr/bin/env python3
"""
Check specific table columns
"""

import mysql.connector
import os
from dotenv import load_dotenv

# Load environment
load_dotenv('e:/git/aitest/.env.live')

def check_table_columns():
    """Check specific table columns"""
    
    tables_to_check = [
        ('crypto_prices', 'price_data'),
        ('crypto_prices', 'unified_sentiment_data'),
        ('crypto_prices', 'trading_signals')
    ]
    
    for db_name, table_name in tables_to_check:
        print(f"\n=== Columns in {db_name}.{table_name} ===")
        
        try:
            conn = mysql.connector.connect(
                host=os.getenv('DB_HOST', 'host.docker.internal'),
                user=os.getenv('DB_USER', 'news_collector'),
                password=os.getenv('DB_PASSWORD', '99Rules!'),
                database=db_name
            )
            cursor = conn.cursor()
            
            cursor.execute(f"DESCRIBE {table_name}")
            columns = cursor.fetchall()
            
            for column in columns:
                print(f"  {column[0]} ({column[1]})")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Error checking {db_name}.{table_name}: {e}")

if __name__ == "__main__":
    check_table_columns()
