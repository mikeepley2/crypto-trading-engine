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
cursor.execute('''
    SELECT id, symbol, signal_type, confidence, amount_usd, execution_status, 
           llm_validation, llm_confidence, llm_reasoning, risk_assessment, 
           suggested_amount, validation_timestamp, created_at
    FROM trade_recommendations 
    WHERE id = 1645
''')
result = cursor.fetchone()
if result:
    print(f'Recommendation 1645:')
    print(f'  Symbol: {result["symbol"]}, Type: {result["signal_type"]}')
    print(f'  LLM Validation: {result["llm_validation"]}')
    print(f'  LLM Confidence: {result["llm_confidence"]}')
    print(f'  LLM Reasoning: {result["llm_reasoning"]}')
    print(f'  Risk Assessment: {result["risk_assessment"]}')
    print(f'  Suggested Amount: {result["suggested_amount"]}')
    print(f'  Validation Timestamp: {result["validation_timestamp"]}')
else:
    print('Recommendation 1645 not found')

# Also check recent validations
cursor.execute('''
    SELECT id, symbol, signal_type, llm_validation, llm_confidence, validation_timestamp
    FROM trade_recommendations 
    WHERE llm_validation IS NOT NULL 
    ORDER BY validation_timestamp DESC 
    LIMIT 5
''')
recent = cursor.fetchall()
print(f'\nRecent LLM validations ({len(recent)}):')
for rec in recent:
    print(f'  ID: {rec["id"]}, {rec["symbol"]} {rec["signal_type"]} - {rec["llm_validation"]} (confidence: {rec["llm_confidence"]})')
    print(f'    Validated: {rec["validation_timestamp"]}')

conn.close()
