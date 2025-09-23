#!/usr/bin/env python3
"""
Check Trade Execution Status
Queries the database to see if any trades have been executed
"""

import mysql.connector
from datetime import datetime

# Database configuration
DB_CONFIG = {
    'host': 'host.docker.internal',
    'user': 'news_collector',
    'password': '99Rules!',
    'database': 'crypto_transactions'
}

def check_trade_status():
    """Check if any trades have been executed"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor(dictionary=True)
        
        print("🔍 TRADE EXECUTION STATUS CHECK")
        print("=" * 50)
        
        # Check total trades
        cursor.execute("""
            SELECT 
                COUNT(*) as total_recommendations,
                COUNT(CASE WHEN execution_status = 'EXECUTED' THEN 1 END) as executed_trades,
                COUNT(CASE WHEN execution_status = 'REJECTED' THEN 1 END) as rejected_trades,
                COUNT(CASE WHEN execution_status = 'PENDING' THEN 1 END) as pending_trades,
                MAX(executed_at) as latest_execution
            FROM trade_recommendations 
            WHERE is_mock = 0
        """)
        
        stats = cursor.fetchone()
        
        print(f"📊 OVERALL TRADE STATISTICS:")
        print(f"   Total Recommendations: {stats['total_recommendations']}")
        print(f"   ✅ Executed Trades: {stats['executed_trades']}")
        print(f"   ❌ Rejected Trades: {stats['rejected_trades']}")
        print(f"   ⏳ Pending Trades: {stats['pending_trades']}")
        print(f"   🕐 Latest Execution: {stats['latest_execution']}")
        
        # Get recent executed trades
        cursor.execute("""
            SELECT id, symbol, action, entry_price, executed_at, execution_price, execution_notes
            FROM trade_recommendations 
            WHERE is_mock = 0 AND execution_status = 'EXECUTED'
            ORDER BY executed_at DESC
            LIMIT 10
        """)
        
        executed_trades = cursor.fetchall()
        
        if executed_trades:
            print(f"\n✅ RECENT EXECUTED TRADES ({len(executed_trades)}):")
            for trade in executed_trades:
                print(f"   🎯 {trade['symbol']} {trade['action']} @ ${trade['execution_price']} (ID: {trade['id']}) - {trade['executed_at']}")
        else:
            print(f"\n❌ NO TRADES HAVE BEEN EXECUTED YET")
        
        # Get recent rejected trades
        cursor.execute("""
            SELECT id, symbol, action, entry_price, executed_at, execution_notes
            FROM trade_recommendations 
            WHERE is_mock = 0 AND execution_status = 'REJECTED'
            ORDER BY executed_at DESC
            LIMIT 5
        """)
        
        rejected_trades = cursor.fetchall()
        
        if rejected_trades:
            print(f"\n❌ RECENT REJECTED TRADES ({len(rejected_trades)}):")
            for trade in rejected_trades:
                print(f"   ⚠️  {trade['symbol']} {trade['action']} - {trade['execution_notes']} (ID: {trade['id']})")
        
        # Check fresh recommendations from today
        cursor.execute("""
            SELECT COUNT(*) as fresh_count
            FROM trade_recommendations 
            WHERE is_mock = 0 
            AND generated_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
            AND status = 'pending'
        """)
        
        fresh = cursor.fetchone()
        print(f"\n🆕 FRESH RECOMMENDATIONS (last hour): {fresh['fresh_count']}")
        
        cursor.close()
        connection.close()
        
        # Summary
        print("\n" + "=" * 50)
        if stats['executed_trades'] > 0:
            print("✅ RESULT: TRADES HAVE BEEN EXECUTED!")
        else:
            print("❌ RESULT: NO TRADES EXECUTED YET")
            if stats['rejected_trades'] > 0:
                print("⚠️  Some trades were rejected - check execution engine health")
        
    except Exception as e:
        print(f"❌ Error checking trade status: {e}")

if __name__ == "__main__":
    check_trade_status()
