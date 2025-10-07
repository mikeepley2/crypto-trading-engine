#!/usr/bin/env python3
"""
Test LLM Validation Integration
"""

import mysql.connector
import os

def test_llm_validation():
    try:
        conn = mysql.connector.connect(
            host=os.getenv('DB_HOST', '172.22.32.1'),
            user=os.getenv('DB_USER', 'news_collector'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME_PRICES', 'crypto_prices')
        )
        
        cursor = conn.cursor(dictionary=True)
        
        # Check for recommendations with LLM validation
        cursor.execute('''
            SELECT id, symbol, signal_type, confidence, llm_validation, llm_confidence, llm_reasoning
            FROM trade_recommendations 
            WHERE created_at >= NOW() - INTERVAL 10 MINUTE
            AND llm_validation IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 5
        ''')
        
        results = cursor.fetchall()
        
        if results:
            print('✅ LLM VALIDATION WORKING!')
            print('Recent LLM-validated recommendations:')
            for rec in results:
                print(f'  ID: {rec["id"]}, {rec["symbol"]} {rec["signal_type"]} - {rec["llm_validation"]} (confidence: {rec["llm_confidence"]:.3f})')
                print(f'    Reasoning: {rec["llm_reasoning"]}')
        else:
            print('⏳ No LLM-validated recommendations found in last 10 minutes')
            print('Checking pending recommendations...')
            
            cursor.execute('''
                SELECT COUNT(*) as count FROM trade_recommendations 
                WHERE execution_status = 'PENDING'
                AND (llm_validation IS NULL OR llm_validation = '')
                AND created_at >= NOW() - INTERVAL 30 MINUTE
            ''')
            
            pending = cursor.fetchone()
            print(f'Pending recommendations awaiting LLM validation: {pending["count"]}')
        
        conn.close()
        
    except Exception as e:
        print(f'❌ Error: {e}')

if __name__ == '__main__':
    test_llm_validation()
