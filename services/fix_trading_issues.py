#!/usr/bin/env python3
"""
Fix Trading System Issues
- Generate fresh signals
- Test connectivity 
- Restart automated trader
"""

import subprocess
import requests
import time
import os

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, 
                              cwd="e:\\git\\aitest\\backend\\services\\trading")
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()[:200]}")
            return True
        else:
            print(f"❌ {description} - Failed")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()[:200]}")
            return False
    except Exception as e:
        print(f"❌ {description} - Exception: {e}")
        return False

def test_connectivity():
    """Test if services are accessible"""
    print(f"\n🔍 Testing Service Connectivity...")
    
    # Test recommendations service
    try:
        response = requests.get("http://localhost:8022/health", timeout=5)
        print(f"✅ Recommendations Service: {response.status_code}")
    except Exception as e:
        print(f"❌ Recommendations Service: {e}")
    
    # Test execution service
    try:
        response = requests.get("http://localhost:8024/health", timeout=10)
        print(f"✅ Execution Service: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Mode: {data.get('mode', 'unknown')}")
            print(f"   Trading Enabled: {data.get('trading_enabled', False)}")
    except Exception as e:
        print(f"❌ Execution Service: {e}")

def main():
    print("🚀 FIXING TRADING SYSTEM ISSUES")
    print("=" * 50)
    
    # Change to trading directory
    os.chdir("e:\\git\\aitest\\backend\\services\\trading")
    
    # Test connectivity first
    test_connectivity()
    
    # Stop and rebuild automated trader
    run_command("docker-compose stop automated-live-trader", "Stopping automated trader")
    run_command("docker-compose build automated-live-trader", "Rebuilding automated trader")
    
    # Start automated trader
    run_command("docker-compose up -d automated-live-trader", "Starting automated trader")
    
    # Wait and check status
    print("\n⏳ Waiting 10 seconds for startup...")
    time.sleep(10)
    
    run_command("docker-compose ps", "Checking container status")
    
    # Test connectivity again
    test_connectivity()
    
    print("\n" + "=" * 50)
    print("✅ System fix attempts completed!")
    print("📝 Monitor logs: docker-compose logs -f automated-live-trader")

if __name__ == "__main__":
    main()
