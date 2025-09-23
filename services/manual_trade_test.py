#!/usr/bin/env python3
"""
Manual Trade Execution
Execute trades manually while system issues are resolved
"""

import requests
import json
import time
from datetime import datetime

def execute_single_trade():
    """Try to execute one trade manually"""
    
    print("🎯 MANUAL TRADE EXECUTION")
    print("=" * 40)
    
    # Create a simple trade recommendation
    trade_data = {
        "symbol": "BTC",
        "action": "BUY", 
        "entry_price": 50000.00,
        "position_size": 0.001,  # Small amount for testing
        "stop_loss": 49000.00,
        "take_profit": 51000.00
    }
    
    print(f"📊 Trade Details:")
    print(f"   Symbol: {trade_data['symbol']}")
    print(f"   Action: {trade_data['action']}")
    print(f"   Price: ${trade_data['entry_price']}")
    print(f"   Size: {trade_data['position_size']}")
    
    # Try to submit trade directly to execution engine
    try:
        print(f"\n🔄 Submitting trade to execution engine...")
        
        url = "http://localhost:8024/execute_trade"
        response = requests.post(url, json=trade_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ TRADE EXECUTED SUCCESSFULLY!")
            print(f"   Result: {result}")
            return True
        else:
            print(f"❌ Trade failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Trade execution error: {e}")
        return False

def check_balance():
    """Check current account balance"""
    try:
        response = requests.get("http://localhost:8024/balance", timeout=10)
        if response.status_code == 200:
            balance = response.json()
            print(f"💰 Current Balance: ${balance.get('usd_balance', 'Unknown')}")
        else:
            print(f"❌ Balance check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Balance error: {e}")

def main():
    print("🚨 MANUAL TRADING MODE")
    print("Testing direct trade execution while automated system is being fixed")
    print("=" * 60)
    
    # Check balance first
    check_balance()
    
    # Try manual trade execution
    success = execute_single_trade()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ MANUAL TRADE EXECUTION SUCCESSFUL!")
        print("🔄 The trading system is working - automated trader should work once connectivity is fixed")
    else:
        print("❌ MANUAL TRADE EXECUTION FAILED")
        print("🔧 Need to fix execution engine connectivity or API credentials")

if __name__ == "__main__":
    main()
