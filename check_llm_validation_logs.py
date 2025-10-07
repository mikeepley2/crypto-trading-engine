#!/usr/bin/env python3
import mysql.connector
import os

def check_llm_validation_logs():
    try:
        conn = mysql.connector.connect(
            host='172.22.32.1',
            user='news_collector', 
            password=os.getenv('DB_PASSWORD'),
            database='crypto_prices'
        )
        cursor = conn.cursor()
        
        # Check total LLM validated recommendations
        cursor.execute('SELECT COUNT(*) FROM trade_recommendations WHERE llm_validation IS NOT NULL')
        count = cursor.fetchone()[0]
        print(f'Total LLM validated recommendations: {count}')
        
        # Check pending recommendations
        cursor.execute('SELECT COUNT(*) FROM trade_recommendations WHERE llm_validation IS NULL AND execution_status = "PENDING"')
        pending = cursor.fetchone()[0]
        print(f'Pending recommendations awaiting LLM validation: {pending}')
        
        # Check recent LLM validations
        cursor.execute('''
            SELECT id, symbol, signal_type, llm_validation, llm_confidence, 
                   llm_reasoning, risk_assessment, validation_timestamp
            FROM trade_recommendations 
            WHERE llm_validation IS NOT NULL 
            ORDER BY validation_timestamp DESC 
            LIMIT 5
        ''')
        recent = cursor.fetchall()
        
        print('\nRecent LLM validations:')
        for rec in recent:
            print(f'  ID: {rec[0]}, {rec[1]} {rec[2]} - {rec[3]} (confidence: {rec[4]})')
            print(f'    Risk: {rec[6]}, Validated: {rec[7]}')
            if rec[5]:
                print(f'    Reasoning: {rec[5][:100]}...')
            print()
        
        conn.close()
        
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    check_llm_validation_logs()
