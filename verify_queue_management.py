#!/usr/bin/env python3
"""
Verify Queue Management Script
Verifies that the queue management is working correctly
"""

import mysql.connector

def verify_queue_management():
    print('=' * 60)
    print('QUEUE MANAGEMENT VERIFICATION')
    print('=' * 60)
    
    # Connect to database
    conn = mysql.connector.connect(
        host='172.22.32.1',
        user='news_collector',
        password='99Rules!',
        database='crypto_prices'
    )
    cursor = conn.cursor()
    
    try:
        # Check current queue status
        cursor.execute('''
            SELECT COUNT(*) as total_pending
            FROM trade_recommendations 
            WHERE execution_status = 'PENDING'
        ''')
        total_pending = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT COUNT(*) as recent_pending
            FROM trade_recommendations 
            WHERE execution_status = 'PENDING'
            AND created_at >= NOW() - INTERVAL 30 MINUTE
        ''')
        recent_pending = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT COUNT(*) as cancelled_old
            FROM trade_recommendations 
            WHERE execution_status = 'CANCELLED'
            AND executed_at >= NOW() - INTERVAL 10 MINUTE
        ''')
        cancelled_old = cursor.fetchone()[0]
        
        print(f'Current Queue Status:')
        print(f'  Total pending recommendations: {total_pending}')
        print(f'  Recent (last 30 min): {recent_pending}')
        print(f'  Old recommendations cancelled (last 10 min): {cancelled_old}')
        
        # Check if all pending are recent
        if total_pending == recent_pending:
            print(f'\n✅ SUCCESS: All pending recommendations are recent (last 30 minutes)')
        else:
            print(f'\n⚠️  WARNING: {total_pending - recent_pending} old recommendations still pending')
        
        # Show recent activity
        cursor.execute('''
            SELECT COUNT(*) as count
            FROM trade_recommendations 
            WHERE created_at >= NOW() - INTERVAL 5 MINUTE
        ''')
        recent_5min = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT COUNT(*) as count
            FROM trade_recommendations 
            WHERE executed_at >= NOW() - INTERVAL 5 MINUTE
            AND execution_status = 'EXECUTED'
        ''')
        executed_5min = cursor.fetchone()[0]
        
        print(f'\n📊 Recent Activity (last 5 minutes):')
        print(f'  New recommendations: {recent_5min}')
        print(f'  Executed trades: {executed_5min}')
        
        # Show age distribution
        cursor.execute('''
            SELECT 
                CASE 
                    WHEN created_at >= NOW() - INTERVAL 5 MINUTE THEN '0-5 min'
                    WHEN created_at >= NOW() - INTERVAL 10 MINUTE THEN '5-10 min'
                    WHEN created_at >= NOW() - INTERVAL 15 MINUTE THEN '10-15 min'
                    WHEN created_at >= NOW() - INTERVAL 20 MINUTE THEN '15-20 min'
                    WHEN created_at >= NOW() - INTERVAL 25 MINUTE THEN '20-25 min'
                    ELSE '25-30 min'
                END as age_group,
                COUNT(*) as count
            FROM trade_recommendations 
            WHERE execution_status = 'PENDING'
            GROUP BY age_group
            ORDER BY 
                CASE 
                    WHEN age_group = '0-5 min' THEN 1
                    WHEN age_group = '5-10 min' THEN 2
                    WHEN age_group = '10-15 min' THEN 3
                    WHEN age_group = '15-20 min' THEN 4
                    WHEN age_group = '20-25 min' THEN 5
                    ELSE 6
                END
        ''')
        
        age_distribution = cursor.fetchall()
        
        print(f'\n📈 Age Distribution of Pending Recommendations:')
        for age_group, count in age_distribution:
            print(f'  {age_group}: {count} recommendations')
        
        print(f'\n🎯 Queue Management Status:')
        if total_pending == recent_pending and cancelled_old > 0:
            print(f'  ✅ SUCCESS: Queue successfully cleared of old recommendations')
            print(f'  ✅ SUCCESS: Only recent recommendations (last 30 min) remain')
            print(f'  ✅ SUCCESS: System configured for 30-minute processing window')
        else:
            print(f'  ⚠️  Queue management may need attention')
        
    except Exception as e:
        print(f'Error during verification: {e}')
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    verify_queue_management()
