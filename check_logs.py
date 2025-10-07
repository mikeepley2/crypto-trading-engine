#!/usr/bin/env python3
import mysql.connector
import os

conn = mysql.connector.connect(
    host='172.22.32.1',
    user='news_collector', 
    password=os.getenv('DB_PASSWORD'),
    database='crypto_prices'
)
cursor = conn.cursor()
cursor.execute('SHOW TABLES LIKE "llm_validation_logs"')
if cursor.fetchone():
    cursor.execute('SELECT COUNT(*) FROM llm_validation_logs')
    count = cursor.fetchone()[0]
    print(f'LLM validation logs table exists with {count} entries')
    
    cursor.execute('SELECT recommendation_id, symbol, error, timestamp FROM llm_validation_logs ORDER BY timestamp DESC LIMIT 3')
    logs = cursor.fetchall()
    print('Recent logs:')
    for log in logs:
        print(f'  ID: {log[0]}, {log[1]} - Error: {log[2][:50] if log[2] else "None"}...')
else:
    print('LLM validation logs table does not exist yet')
conn.close()
