#!/usr/bin/env python3
"""
Check database structure
"""

import mysql.connector
import os
from dotenv import load_dotenv

# Load environment
load_dotenv('e:/git/aitest/.env.live')

def check_database_structure():
    """Check database structure"""
    
    databases = ['crypto_prices', 'crypto_transactions']
    
    for db_name in databases:
        print(f"\n=== Checking {db_name} database ===")
        
        try:
            conn = mysql.connector.connect(
                host=os.getenv('DB_HOST', 'host.docker.internal'),
                user=os.getenv('DB_USER', 'news_collector'),
                password=os.getenv('DB_PASSWORD', '99Rules!'),
                database=db_name
            )
            cursor = conn.cursor()
            
            # Show tables
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            
            if tables:
                print(f"Tables in {db_name}:")
                for table in tables:
                    print(f"  - {table[0]}")
                    
                    # Show first few columns for each table
                    cursor.execute(f"DESCRIBE {table[0]}")
                    columns = cursor.fetchall()
                    print(f"    Columns: {', '.join([col[0] for col in columns[:5]])}")
                    if len(columns) > 5:
                        print(f"    ... and {len(columns) - 5} more columns")
            else:
                print(f"No tables found in {db_name}")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Error checking {db_name}: {e}")

if __name__ == "__main__":
    check_database_structure()
