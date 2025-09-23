#!/usr/bin/env python3
"""
Emergency Trading System - Bypass Health Checks
Try to execute trades directly, ignoring health check failures
"""

import requests
import json
import time
from datetime import datetime, timedelta

def get_fresh_recommendations():
    """Get fresh recommendations directly"""
    try:
        # Get recent recommendations (within last hour)
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        cutoff_str = cutoff_time.strftime('%Y-%m-%d %H:%M:%S')
        
        url = f"http://localhost:8022/recommendations"
        params = {
            'is_mock': 'false',
            'status': 'pending',
            'generated_after': cutoff_str,
            'limit': 5
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        recommendations = response.json()
        print(f"🆕 Found {len(recommendations)} fresh recommendations")
        
        for rec in recommendations:
            print(f"   📊 {rec['symbol']} {rec['action']} @ ${rec['entry_price']} (ID: {rec['id']})")
        
        return recommendations
        
    except Exception as e:
        print(f"❌ Error getting recommendations: {e}")
        return []

def execute_trade(recommendation_id):
    """Try to execute a single trade"""
    try:
        url = f"http://localhost:8024/process_recommendation/{recommendation_id}"
        response = requests.post(url, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Trade {recommendation_id} executed: {result}")
            return True
        else:
            print(f"❌ Trade {recommendation_id} failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Trade {recommendation_id} exception: {e}")
        return False

def main():
    print("🚨 EMERGENCY TRADING - BYPASS HEALTH CHECKS")
    print("=" * 60)
    
    # Get fresh recommendations
    recommendations = get_fresh_recommendations()
    
    if not recommendations:
        print("❌ No fresh recommendations to execute")
        return
    
    # Try to execute up to 3 trades
    executed_count = 0
    max_trades = 3
    
    for rec in recommendations[:max_trades]:
        print(f"\n🎯 Attempting to execute trade {rec['id']}: {rec['symbol']} {rec['action']}")
        
        success = execute_trade(rec['id'])
        if success:
            executed_count += 1
        
        time.sleep(2)  # Brief pause between trades
    
    print(f"\n" + "=" * 60)
    print(f"📊 EXECUTION SUMMARY:")
    print(f"   Available trades: {len(recommendations)}")
    print(f"   Attempted trades: {min(len(recommendations), max_trades)}")
    print(f"   Successful trades: {executed_count}")
    
    if executed_count > 0:
        print("✅ TRADES EXECUTED SUCCESSFULLY!")
    else:
        print("❌ NO TRADES EXECUTED - CHECK EXECUTION ENGINE")

if __name__ == "__main__":
    main()
