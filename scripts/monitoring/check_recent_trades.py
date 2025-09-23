#!/usr/bin/env python3

import mysql.connector
from datetime import datetime, timedelta

# Database connection
db_config = {
    'host': 'host.docker.internal',
    'user': 'news_collector',
    'password': '99Rules!',
    'database': 'crypto_transactions',
    'charset': 'utf8mb4'
}

try:
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    
    # Check recent trades in last 24 hours
    print('=== RECENT TRADING ACTIVITY (Last 24 Hours) ===')
    cursor.execute('''
        SELECT COUNT(*) as count 
        FROM trades 
        WHERE timestamp > DATE_SUB(NOW(), INTERVAL 24 HOURS)
    ''')
    recent_trades = cursor.fetchone()
    print(f'Trades in last 24h: {recent_trades["count"]}')
    
    # Check last few trades
    print('\n=== LAST 5 TRADES ===')
    cursor.execute('''
        SELECT symbol, side, quantity, price, timestamp, total_value
        FROM trades 
        ORDER BY timestamp DESC 
        LIMIT 5
    ''')
    last_trades = cursor.fetchall()
    
    for trade in last_trades:
        print(f'{trade["timestamp"]} | {trade["side"]} {trade["quantity"]:.4f} {trade["symbol"]} @ ${trade["price"]:.4f} = ${trade["total_value"]:.2f}')
    
    # Check total trades
    print('\n=== OVERALL STATISTICS ===')
    cursor.execute('SELECT COUNT(*) as total FROM trades')
    total = cursor.fetchone()
    print(f'Total trades ever: {total["total"]}')
    
    # Check recent recommendations
    cursor.execute('''
        SELECT COUNT(*) as count 
        FROM trade_recommendations 
        WHERE created_at > DATE_SUB(NOW(), INTERVAL 24 HOURS)
    ''')
    recent_recs = cursor.fetchone()
    print(f'Trade recommendations in last 24h: {recent_recs["count"]}')
    
    # Check recent signals
    print('\n=== RECENT ML SIGNALS ===')
    cursor.execute('''
        SELECT COUNT(*) as count 
        FROM crypto_prices.trading_signals 
        WHERE timestamp > DATE_SUB(NOW(), INTERVAL 24 HOURS)
    ''')
    recent_signals = cursor.fetchone()
    print(f'ML signals in last 24h: {recent_signals["count"]}')
    
    conn.close()
    
except Exception as e:
    print(f'Error: {e}')
