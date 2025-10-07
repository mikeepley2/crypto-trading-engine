#!/usr/bin/env python3
"""
Clear Recommendation Queue Script
Clears old recommendations to only process recent ones (last 30 minutes)
"""

import mysql.connector
from datetime import datetime

def clear_recommendation_queue():
    print('=' * 60)
    print('CLEARING RECOMMENDATION QUEUE')
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
        
        old_pending = total_pending - recent_pending
        
        print(f'Current Queue Status:')
        print(f'  Total pending recommendations: {total_pending}')
        print(f'  Recent (last 30 min): {recent_pending}')
        print(f'  Old (older than 30 min): {old_pending}')
        
        if old_pending > 0:
            print(f'\n🗑️  Clearing {old_pending} old recommendations...')
            
            # Update old recommendations to CANCELLED status
            cursor.execute('''
                UPDATE trade_recommendations 
                SET execution_status = 'CANCELLED',
                    executed_at = NOW()
                WHERE execution_status = 'PENDING'
                AND created_at < NOW() - INTERVAL 30 MINUTE
            ''')
            
            affected_rows = cursor.rowcount
            conn.commit()
            
            print(f'✅ Successfully cancelled {affected_rows} old recommendations')
            
            # Verify the cleanup
            cursor.execute('''
                SELECT COUNT(*) as remaining_pending
                FROM trade_recommendations 
                WHERE execution_status = 'PENDING'
            ''')
            remaining_pending = cursor.fetchone()[0]
            
            print(f'\n📊 Updated Queue Status:')
            print(f'  Remaining pending recommendations: {remaining_pending}')
            print(f'  All remaining recommendations are from the last 30 minutes')
            
            # Show age distribution of remaining recommendations
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
            
            print(f'\n📈 Age Distribution of Remaining Recommendations:')
            for age_group, count in age_distribution:
                print(f'  {age_group}: {count} recommendations')
            
        else:
            print(f'\n✅ No old recommendations to clear')
            print(f'  All {total_pending} pending recommendations are recent (last 30 minutes)')
        
        # Show recent activity
        print(f'\n📊 Recent Activity Summary:')
        cursor.execute('''
            SELECT COUNT(*) as count
            FROM trade_recommendations 
            WHERE created_at >= NOW() - INTERVAL 10 MINUTE
        ''')
        recent_10min = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT COUNT(*) as count
            FROM trade_recommendations 
            WHERE executed_at >= NOW() - INTERVAL 10 MINUTE
            AND execution_status = 'EXECUTED'
        ''')
        executed_10min = cursor.fetchone()[0]
        
        print(f'  Recommendations created (last 10 min): {recent_10min}')
        print(f'  Trades executed (last 10 min): {executed_10min}')
        
        print(f'\n🎯 Queue Management Complete!')
        print(f'  System will now only process recommendations from the last 30 minutes')
        
    except Exception as e:
        print(f'Error during queue clearing: {e}')
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    clear_recommendation_queue()
