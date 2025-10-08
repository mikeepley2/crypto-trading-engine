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

# Check recent LLM validated recommendations
cursor.execute('''
    SELECT id, symbol, signal_type, confidence, amount_usd, execution_status, 
           llm_validation, llm_confidence, validation_timestamp
    FROM trade_recommendations 
    WHERE llm_validation = 'APPROVE' 
    AND execution_status = 'PENDING'
    ORDER BY validation_timestamp DESC 
    LIMIT 5
''')
approved = cursor.fetchall()
print(f'LLM Approved Recommendations awaiting execution ({len(approved)}):')
for rec in approved:
    print(f'  ID: {rec["id"]}, {rec["symbol"]} {rec["signal_type"]} - {rec["execution_status"]} (LLM confidence: {rec["llm_confidence"]})')
    print(f'    Amount: ${rec["amount_usd"]}, Validated: {rec["validation_timestamp"]}')

# Check recent executed trades
cursor.execute('''
    SELECT id, symbol, signal_type, execution_status, amount_usd, created_at
    FROM trade_recommendations 
    WHERE execution_status = 'EXECUTED'
    ORDER BY created_at DESC 
    LIMIT 5
''')
executed = cursor.fetchall()
print(f'\nRecently Executed Trades ({len(executed)}):')
for trade in executed:
    print(f'  ID: {trade["id"]}, {trade["symbol"]} {trade["signal_type"]} - {trade["execution_status"]}')
    print(f'    Amount: ${trade["amount_usd"]}, Created: {trade["created_at"]}')

# Check trade orchestrator status
cursor.execute('''
    SELECT COUNT(*) as total_pending
    FROM trade_recommendations 
    WHERE execution_status = 'PENDING'
    AND llm_validation = 'APPROVE'
''')
pending_count = cursor.fetchone()['total_pending']
print(f'\nTotal LLM-approved recommendations pending execution: {pending_count}')

conn.close()
