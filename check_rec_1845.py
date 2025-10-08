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

# Check recommendation 1845 specifically
cursor.execute('''
    SELECT id, symbol, signal_type, confidence, amount_usd, execution_status, 
           llm_validation, llm_confidence, created_at, validation_timestamp
    FROM trade_recommendations 
    WHERE id = 1845
''')
rec = cursor.fetchone()
if rec:
    print(f'Recommendation 1845:')
    print(f'  Symbol: {rec["symbol"]}, Type: {rec["signal_type"]}')
    print(f'  Confidence: {rec["confidence"]}, Amount: ${rec["amount_usd"]}')
    print(f'  Execution Status: {rec["execution_status"]}')
    print(f'  LLM Validation: {rec["llm_validation"]}')
    print(f'  Created: {rec["created_at"]}')
else:
    print('Recommendation 1845 not found')

# Check if there are any new pending recommendations
cursor.execute('''
    SELECT id, symbol, signal_type, confidence, amount_usd, execution_status, 
           llm_validation, created_at
    FROM trade_recommendations 
    WHERE execution_status = 'PENDING'
    AND llm_validation IS NULL
    ORDER BY created_at DESC 
    LIMIT 5
''')
pending = cursor.fetchall()
print(f'\nPending recommendations awaiting LLM validation ({len(pending)}):')
for rec in pending:
    print(f'  ID: {rec["id"]}, {rec["symbol"]} {rec["signal_type"]} - {rec["execution_status"]}')
    print(f'    Confidence: {rec["confidence"]}, Amount: ${rec["amount_usd"]}, Created: {rec["created_at"]}')

conn.close()
