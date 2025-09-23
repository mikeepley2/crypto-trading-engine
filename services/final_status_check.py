#!/usr/bin/env python3
"""
Comprehensive Trading System Status
Final status check after fixes
"""

import requests
import subprocess
import json
from datetime import datetime, timedelta

def check_containers():
    """Check Docker container status"""
    print("🐳 CONTAINER STATUS:")
    try:
        result = subprocess.run(["docker-compose", "ps"], 
                              capture_output=True, text=True, 
                              cwd="e:\\git\\aitest\\backend\\services\\trading")
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines[1:]:  # Skip header
                if line.strip():
                    print(f"   {line}")
        else:
            print("   ❌ Error checking containers")
    except Exception as e:
        print(f"   ❌ Exception: {e}")

def check_services():
    """Check service health"""
    print("\n🔍 SERVICE HEALTH:")
    
    # Recommendations service
    try:
        response = requests.get("http://localhost:8022/health", timeout=5)
        print(f"   ✅ Recommendations: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Recommendations: {e}")
    
    # Execution service
    try:
        response = requests.get("http://localhost:8024/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            mode = data.get('mode', 'unknown')
            trading = data.get('trading_enabled', False)
            print(f"   ✅ Execution: {response.status_code} (Mode: {mode}, Trading: {trading})")
        else:
            print(f"   ⚠️  Execution: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Execution: {e}")

def check_fresh_recommendations():
    """Check for fresh recommendations"""
    print("\n📊 FRESH RECOMMENDATIONS:")
    try:
        cutoff = (datetime.utcnow() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
        url = "http://localhost:8022/recommendations"
        params = {
            'is_mock': 'false',
            'status': 'pending',
            'generated_after': cutoff,
            'limit': 5
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            recs = response.json()
            print(f"   ✅ Found {len(recs)} fresh recommendations")
            for rec in recs[:3]:
                print(f"      📈 {rec['symbol']} {rec['action']} @ ${rec['entry_price']} (ID: {rec['id']})")
        else:
            print(f"   ❌ Status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def check_automated_trader():
    """Check automated trader logs"""
    print("\n🤖 AUTOMATED TRADER:")
    try:
        result = subprocess.run(
            ["docker-compose", "logs", "automated-live-trader", "--tail=5"],
            capture_output=True, text=True,
            cwd="e:\\git\\aitest\\backend\\services\\trading"
        )
        if result.returncode == 0:
            logs = result.stdout.strip()
            if logs:
                # Get just the last few lines without docker-compose warnings
                lines = [line for line in logs.split('\n') if 'automated-live-trader' in line]
                for line in lines[-3:]:
                    print(f"   {line}")
            else:
                print("   ⚠️  No recent logs")
        else:
            print("   ❌ Error getting logs")
    except Exception as e:
        print(f"   ❌ Exception: {e}")

def main():
    print("🔬 COMPREHENSIVE TRADING SYSTEM STATUS")
    print("=" * 60)
    
    check_containers()
    check_services()
    check_fresh_recommendations()
    check_automated_trader()
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY:")
    print("   1. If services are healthy and fresh recommendations exist:")
    print("      - Automated trader should execute trades automatically")
    print("   2. If execution service is accessible:")
    print("      - Manual trade execution should work")
    print("   3. Monitor: docker-compose logs -f automated-live-trader")

if __name__ == "__main__":
    main()
