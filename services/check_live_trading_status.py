#!/usr/bin/env python3
"""
Live Trading Status Checker
Monitors automated trading system and fresh signals
"""

import requests
import json
import subprocess
from datetime import datetime, timedelta

def check_service_health():
    """Check if all services are running"""
    print("🔍 Checking Service Health...")
    
    # Check recommendations service
    try:
        response = requests.get("http://localhost:8022/health", timeout=5)
        print(f"✅ Recommendations Service: {response.status_code}")
    except Exception as e:
        print(f"❌ Recommendations Service: {e}")
    
    # Check execution service
    try:
        response = requests.get("http://localhost:8024/health", timeout=5)
        print(f"✅ Execution Service: {response.status_code}")
    except Exception as e:
        print(f"❌ Execution Service: {e}")

def check_fresh_signals():
    """Check for fresh trading signals from today"""
    print("\n📊 Checking Fresh Trading Signals...")
    
    try:
        # Get recent recommendations
        cutoff_time = (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
        url = f"http://localhost:8022/recommendations?is_mock=false&status=pending&generated_after={cutoff_time}&limit=10"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            recommendations = response.json()
            
            if recommendations:
                print(f"✅ Found {len(recommendations)} fresh recommendations:")
                for rec in recommendations:
                    print(f"   {rec['symbol']} {rec['action']} @ ${rec['entry_price']} (conf: {rec['confidence']}) - {rec['generated_at']}")
            else:
                print("⚠️  No fresh recommendations found")
                
                # Try to generate some
                print("🔄 Generating fresh signals...")
                subprocess.run(["python", "generate_fresh_signals.py"], check=True)
                
        else:
            print(f"❌ Error fetching recommendations: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error checking signals: {e}")

def check_container_status():
    """Check Docker container status"""
    print("\n🐳 Checking Container Status...")
    
    try:
        result = subprocess.run(["docker-compose", "ps"], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"❌ Error checking containers: {e}")

def check_live_balance():
    """Check current portfolio balance"""
    print("\n💰 Checking Live Balance...")
    
    try:
        response = requests.get("http://localhost:8026/portfolio/balance", timeout=5)
        if response.status_code == 200:
            balance = response.json()
            print(f"✅ Available Balance: ${balance.get('usd_balance', 'N/A')}")
        else:
            print(f"❌ Error fetching balance: {response.status_code}")
    except Exception as e:
        print(f"❌ Error checking balance: {e}")

def main():
    print("🚀 Live Trading System Status Check")
    print("=" * 50)
    
    check_service_health()
    check_fresh_signals()
    check_container_status()
    check_live_balance()
    
    print("\n" + "=" * 50)
    print("✅ Status check complete!")
    print("📝 To view automated trader logs: docker-compose logs -f automated-live-trader")

if __name__ == "__main__":
    main()
