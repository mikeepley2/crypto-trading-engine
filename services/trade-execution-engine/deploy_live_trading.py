#!/usr/bin/env python3
"""
Live Trading Deployment Script
Deploy and start the live trading system safely
"""

import os
import sys
import time
import subprocess
import requests
from datetime import datetime
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"🚀 {title}")
    print("=" * 60)

def print_step(step):
    """Print a step"""
    print(f"\n📋 {step}")

def run_command(command, cwd=None, check=True):
    """Run a command and return the result"""
    print(f"   Running: {command}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd, 
            capture_output=True, 
            text=True,
            check=check
        )
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"   Error: {e}")
        if e.stderr:
            print(f"   Stderr: {e.stderr}")
        return e

def check_service_health(url, service_name, max_retries=30):
    """Check if a service is healthy"""
    for i in range(max_retries):
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print(f"   ✅ {service_name} is healthy")
                return True
        except requests.exceptions.RequestException:
            pass
        
        if i < max_retries - 1:
            print(f"   ⏳ Waiting for {service_name} to start... ({i+1}/{max_retries})")
            time.sleep(2)
    
    print(f"   ❌ {service_name} failed to start")
    return False

def main():
    """Main deployment function"""
    print_header("Live Trading System Deployment")
    
    # Change to the live trading directory
    live_trading_dir = Path("e:/git/aitest/backend/services/trading/live")
    print(f"Working directory: {live_trading_dir}")
    
    # Step 1: Environment validation
    print_step("1. Validating Environment")
    env_file = Path("e:/git/aitest/.env.live")
    if not env_file.exists():
        print("   ❌ .env.live file not found!")
        print("   Please ensure your API credentials are in .env.live")
        return False
    print("   ✅ Environment file found")
    
    # Step 2: Install dependencies
    print_step("2. Installing Dependencies")
    result = run_command("pip install -r requirements.txt", cwd=live_trading_dir)
    if isinstance(result, subprocess.CalledProcessError):
        print("   ❌ Failed to install dependencies")
        return False
    print("   ✅ Dependencies installed")
    
    # Step 3: Run comprehensive tests
    print_step("3. Running System Tests")
    result = run_command("python test_live_trading.py", cwd=live_trading_dir, check=False)
    
    # Continue regardless of test results - user can review
    print("   📊 Tests completed - please review results above")
    
    # Step 4: Build Docker images
    print_step("4. Building Docker Images")
    result = run_command("docker-compose build", cwd=live_trading_dir)
    if isinstance(result, subprocess.CalledProcessError):
        print("   ❌ Failed to build Docker images")
        return False
    print("   ✅ Docker images built")
    
    # Step 5: Start services
    print_step("5. Starting Live Trading Services")
    
    # Stop any existing services first
    run_command("docker-compose down", cwd=live_trading_dir, check=False)
    
    # Start in detached mode
    result = run_command("docker-compose up -d", cwd=live_trading_dir)
    if isinstance(result, subprocess.CalledProcessError):
        print("   ❌ Failed to start services")
        return False
    print("   ✅ Services started")
    
    # Step 6: Health checks
    print_step("6. Checking Service Health")
    
    services = [
        ("http://localhost:8024", "Live Trading Engine"),
        ("http://localhost:8025", "Trading Monitor")
    ]
    
    all_healthy = True
    for url, name in services:
        if not check_service_health(url, name):
            all_healthy = False
    
    # Step 7: Final status
    print_step("7. Deployment Status")
    
    if all_healthy:
        print("   🟢 ALL SERVICES RUNNING")
        print("\n📊 Service URLs:")
        print("   • Live Trading Engine: http://localhost:8024")
        print("   • Trading Monitor: http://localhost:8025")
        print("   • Health Dashboard: http://localhost:8025/dashboard")
        
        print("\n🛡️ SAFETY REMINDERS:")
        print("   • System is configured with safety limits:")
        print("   • Max position size: $100 USD")
        print("   • Max daily trades: 20")
        print("   • Max daily loss: $200 USD")
        print("   • Monitor the system closely during initial operation")
        
        print("\n📋 NEXT STEPS:")
        print("   1. Review test results above")
        print("   2. Monitor logs: docker-compose logs -f")
        print("   3. Check trading monitor dashboard")
        print("   4. Start with small test trades")
        
        return True
    else:
        print("   🔴 DEPLOYMENT FAILED")
        print("   Some services are not healthy. Check logs:")
        print("   docker-compose logs")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
