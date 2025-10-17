#!/usr/bin/env python3

import mysql.connector
import os

DB_HOST = os.getenv('DB_HOST', '172.22.32.1')
DB_USER = os.getenv('DB_USER', 'news_collector')
DB_PASSWORD = os.getenv('DB_PASSWORD', '99Rules!')
DB_NAME_PRICES = os.getenv('DB_NAME_PRICES', 'crypto_prices')

def get_db_connection():
    try:
        return mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME_PRICES
        )
    except Exception as e:
        print(f'Database connection error: {e}')
        return None

def check_pending_recommendations():
    conn = get_db_connection()
    if not conn:
        return

    cursor = conn.cursor(dictionary=True)

    print('=== PENDING RECOMMENDATIONS CHECK ===')
    cursor.execute('''
        SELECT id, symbol, signal_type, confidence, execution_status, llm_validation, created_at 
        FROM trade_recommendations 
        WHERE execution_status = 'PENDING'
        ORDER BY created_at DESC 
        LIMIT 10
    ''')
    pending = cursor.fetchall()

    print(f'Pending recommendations: {len(pending)}')
    for rec in pending:
        print(f'  ID: {rec["id"]}, Symbol: {rec["symbol"]}, Type: {rec["signal_type"]}, Status: {rec["execution_status"]}, LLM: {rec["llm_validation"]}')

    print()
    print('=== RECENT RECOMMENDATIONS STATUS ===')
    cursor.execute('''
        SELECT execution_status, COUNT(*) as count 
        FROM trade_recommendations 
        WHERE created_at >= DATE_SUB(NOW(), INTERVAL 2 HOUR)
        GROUP BY execution_status
    ''')
    status_breakdown = cursor.fetchall()
    for status in status_breakdown:
        print(f'  {status["execution_status"]}: {status["count"]}')

    print()
    print('=== RECENT RECOMMENDATIONS WITH LLM VALIDATION ===')
    cursor.execute('''
        SELECT id, symbol, signal_type, execution_status, llm_validation, llm_confidence, llm_reasoning, created_at 
        FROM trade_recommendations 
        WHERE created_at >= DATE_SUB(NOW(), INTERVAL 2 HOUR)
        AND llm_validation IS NOT NULL
        ORDER BY created_at DESC 
        LIMIT 10
    ''')
    llm_validated = cursor.fetchall()
    
    print(f'LLM validated recommendations: {len(llm_validated)}')
    for rec in llm_validated:
        print(f'  ID: {rec["id"]}, Symbol: {rec["symbol"]}, Type: {rec["signal_type"]}, Status: {rec["execution_status"]}')
        print(f'    LLM Validated: {rec["llm_validation"]}, Confidence: {rec["llm_confidence"]}')
        print(f'    Reasoning: {rec["llm_reasoning"]}')

    cursor.close()
    conn.close()

if __name__ == '__main__':
    check_pending_recommendations()
