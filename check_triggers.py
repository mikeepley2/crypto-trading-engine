#!/usr/bin/env python3

import mysql.connector

def check_triggers():
    try:
        conn = mysql.connector.connect(
            host='172.22.32.1',
            user='news_collector',
            password='99Rules!',
            database='crypto_prices'
        )
        
        cursor = conn.cursor()
        
        print('=== CHECKING DATABASE TRIGGERS ===')
        
        # Check for triggers on trading_signals table
        cursor.execute('''
            SELECT TRIGGER_NAME, EVENT_MANIPULATION, ACTION_TIMING, ACTION_STATEMENT
            FROM INFORMATION_SCHEMA.TRIGGERS 
            WHERE EVENT_OBJECT_TABLE = 'trading_signals'
        ''')
        
        triggers = cursor.fetchall()
        
        if triggers:
            print(f'Found {len(triggers)} triggers on trading_signals table:')
            for trigger in triggers:
                print(f'  {trigger[0]}: {trigger[1]} {trigger[2]} - {trigger[3]}')
        else:
            print('No triggers found on trading_signals table')
        
        print()
        
        # Check table structure
        cursor.execute('DESCRIBE trading_signals')
        columns = cursor.fetchall()
        
        print('trading_signals table structure:')
        for column in columns:
            print(f'  {column[0]}: {column[1]} {column[2]} {column[3]} {column[4]} {column[5]}')
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    check_triggers()
