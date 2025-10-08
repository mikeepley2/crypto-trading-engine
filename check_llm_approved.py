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

# Check recommendation 1844 specifically
cursor.execute('''
    SELECT id, symbol, signal_type, confidence, amount_usd, execution_status, 
           llm_validation, llm_confidence, created_at, validation_timestamp
    FROM trade_recommendations 
    WHERE id = 1844
''')
rec = cursor.fetchone()
if rec:
    print(f'Recommendation 1844 (BCH):')
    print(f'  Symbol: {rec["symbol"]}, Type: {rec["signal_type"]}')
    print(f'  Execution Status: {rec["execution_status"]}')
    print(f'  LLM Validation: {rec["llm_validation"]}')
    print(f'  LLM Confidence: {rec["llm_confidence"]}')
    print(f'  Validation Timestamp: {rec["validation_timestamp"]}')

# Check all LLM-approved recommendations
cursor.execute('''
    SELECT id, symbol, signal_type, execution_status, llm_validation, llm_confidence, validation_timestamp
    FROM trade_recommendations 
    WHERE llm_validation = 'APPROVE'
    ORDER BY validation_timestamp DESC 
    LIMIT 5
''')
approved = cursor.fetchall()
print(f'\nAll LLM-approved recommendations ({len(approved)}):')
for rec in approved:
    print(f'  ID: {rec["id"]}, {rec["symbol"]} {rec["signal_type"]} - {rec["execution_status"]} (LLM conf: {rec["llm_confidence"]})')
    print(f'    Validated: {rec["validation_timestamp"]}')

# Check pending LLM-approved recommendations
cursor.execute('''
    SELECT id, symbol, signal_type, execution_status, llm_validation, llm_confidence, validation_timestamp
    FROM trade_recommendations 
    WHERE llm_validation = 'APPROVE'
    AND execution_status = 'PENDING'
    ORDER BY validation_timestamp DESC 
    LIMIT 5
''')
pending_approved = cursor.fetchall()
print(f'\nPending LLM-approved recommendations ({len(pending_approved)}):')
for rec in pending_approved:
    print(f'  ID: {rec["id"]}, {rec["symbol"]} {rec["signal_type"]} - {rec["execution_status"]} (LLM conf: {rec["llm_confidence"]})')
    print(f'    Validated: {rec["validation_timestamp"]}')

conn.close()
