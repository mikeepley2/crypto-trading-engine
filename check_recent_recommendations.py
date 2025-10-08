#!/usr/bin/env python3
import mysql.connector
import os

conn = mysql.connector.connect(
    host='172.22.32.1',
    user='news_collector', 
    password=os.getenv('DB_PASSWORD'),
    database='crypto_prices'
)
cursor = conn.cursor(dictionary=True)

# Check the most recent recommendations
cursor.execute('''
    SELECT id, symbol, signal_type, confidence, amount_usd, execution_status, 
           llm_validation, llm_confidence, created_at, validation_timestamp
    FROM trade_recommendations 
    ORDER BY created_at DESC 
    LIMIT 10
''')
recent = cursor.fetchall()
print('Most Recent Recommendations:')
for rec in recent:
    llm_status = rec['llm_validation'] if rec['llm_validation'] else 'NOT_VALIDATED'
    llm_conf = f"({rec['llm_confidence']:.2f})" if rec['llm_confidence'] else ''
    print(f'  ID: {rec["id"]}, {rec["symbol"]} {rec["signal_type"]} - {rec["execution_status"]} | LLM: {llm_status} {llm_conf}')
    print(f'    Created: {rec["created_at"]}, Validated: {rec["validation_timestamp"]}')

# Check if there are any pending recommendations that could be LLM validated
cursor.execute('''
    SELECT COUNT(*) as pending_count
    FROM trade_recommendations 
    WHERE execution_status = 'PENDING'
    AND llm_validation IS NULL
    AND created_at > DATE_SUB(NOW(), INTERVAL 1 HOUR)
''')
pending = cursor.fetchone()['pending_count']
print(f'\nPending recommendations (last hour) awaiting LLM validation: {pending}')

conn.close()
