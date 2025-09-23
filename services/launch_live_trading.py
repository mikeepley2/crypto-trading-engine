#!/usr/bin/env python3
"""
Simple Live Trading Launcher
Restarts the automated trading system
"""

import subprocess
import os
import time

def run_command(cmd, description):
    """Run a command and print results"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="e:\\git\\aitest\\backend\\services\\trading")
        
        if result.stdout:
            print(f"✅ Output: {result.stdout.strip()}")
        if result.stderr:
            print(f"⚠️  Error: {result.stderr.strip()}")
            
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    print("🚀 Starting Automated Live Trading System")
    print("=" * 50)
    
    # Change to trading directory
    os.chdir("e:\\git\\aitest\\backend\\services\\trading")
    
    # Stop existing containers
    run_command("docker-compose down", "Stopping existing containers")
    
    # Generate fresh signals
    run_command("python generate_fresh_signals.py", "Generating fresh trading signals")
    
    # Start recommendations service
    run_command("docker-compose up -d trade-recommendations", "Starting recommendations service")
    
    # Wait a moment
    print("⏳ Waiting 10 seconds for service to start...")
    time.sleep(10)
    
    # Start automated trader
    run_command("docker-compose up -d automated-live-trader", "Starting automated trader")
    
    # Check status
    run_command("docker-compose ps", "Checking container status")
    
    print("\n" + "=" * 50)
    print("✅ Live trading system should now be running!")
    print("📊 Check recommendations: curl http://localhost:8022/recommendations?limit=5")
    print("📝 Monitor logs: docker-compose logs -f automated-live-trader")

if __name__ == "__main__":
    main()
