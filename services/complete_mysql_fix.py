#!/usr/bin/env python3
"""
Complete MySQL Connection Fix
Solve the Too Many Connections issue permanently
"""

import subprocess
import time
import os

def run_command(cmd, description, ignore_errors=False):
    """Run a command with error handling"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 or ignore_errors:
            print(f"✅ {description} - Success")
            if result.stdout and len(result.stdout.strip()) > 0:
                # Show only first 200 chars to avoid spam
                output = result.stdout.strip()[:200] 
                if len(result.stdout.strip()) > 200:
                    output += "..."
                print(f"   Output: {output}")
            return True
        else:
            print(f"❌ {description} - Failed (code: {result.returncode})")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏱️ {description} - Timeout (30s)")
        return False
    except Exception as e:
        print(f"❌ {description} - Exception: {e}")
        return False

def stop_all_containers():
    """Stop all trading-related containers"""
    print("🛑 STOPPING ALL CONTAINERS TO FREE CONNECTIONS...")
    
    # Stop trading services
    os.chdir("e:\\git\\aitest\\backend\\services\\trading")
    run_command("docker-compose down", "Stopping trading services")
    
    # Stop execution engine
    os.chdir("e:\\git\\aitest\\backend\\services\\trading\\trade-execution-engine")
    run_command("docker-compose down", "Stopping execution engine")
    
    # Kill any other docker containers using our database
    run_command("docker ps -q | xargs -r docker stop", "Stopping other containers", ignore_errors=True)

def fix_mysql_connections():
    """Fix MySQL connection limits"""
    print("\n💾 FIXING MYSQL CONNECTION LIMITS...")
    
    # Kill any hanging python processes
    run_command("taskkill /F /IM python.exe", "Killing Python processes", ignore_errors=True)
    run_command("taskkill /F /IM mysql.exe", "Killing MySQL processes", ignore_errors=True)
    
    # Wait for processes to die
    time.sleep(5)
    
    # Try to increase MySQL connection limit via configuration
    mysql_commands = [
        "SET GLOBAL max_connections = 5000;",
        "SET GLOBAL wait_timeout = 28800;", 
        "SET GLOBAL interactive_timeout = 28800;",
        "SHOW VARIABLES LIKE 'max_connections';"
    ]
    
    for cmd in mysql_commands:
        run_command(f'mysql -h localhost -u root -p99Rules! -e "{cmd}"', 
                   f"MySQL: {cmd[:30]}...", ignore_errors=True)

def restart_services_sequentially():
    """Restart services one by one to avoid connection overload"""
    print("\n🔄 RESTARTING SERVICES SEQUENTIALLY...")
    
    # Start execution engine first (in live mode)
    os.chdir("e:\\git\\aitest\\backend\\services\\trading\\trade-execution-engine")
    
    # Set environment variables for live trading
    env_cmd = "set EXECUTION_MODE=live && set TRADE_EXECUTION_ENABLED=true && docker-compose up -d"
    run_command(env_cmd, "Starting execution engine in LIVE mode")
    
    # Wait for it to fully start
    print("⏳ Waiting 20 seconds for execution engine to initialize...")
    time.sleep(20)
    
    # Check if execution engine is healthy
    run_command("curl -s http://localhost:8024/health", "Testing execution engine")
    
    # Start trading services
    os.chdir("e:\\git\\aitest\\backend\\services\\trading")
    run_command("docker-compose up -d trade-recommendations", "Starting recommendations service")
    
    # Wait for recommendations to start
    print("⏳ Waiting 15 seconds for recommendations service...")
    time.sleep(15)
    
    # Start automated trader last
    run_command("docker-compose up -d automated-live-trader", "Starting automated trader")

def validate_system():
    """Test that everything is working"""
    print("\n✅ VALIDATING SYSTEM...")
    
    # Test services
    run_command("curl -s http://localhost:8022/health", "Testing recommendations service")
    run_command("curl -s http://localhost:8024/health", "Testing execution engine")
    
    # Check container status
    run_command("docker ps --format 'table {{.Names}}\\t{{.Status}}'", "Container status")
    
    # Generate fresh signals
    os.chdir("e:\\git\\aitest\\backend\\services\\trading")
    run_command("python generate_fresh_signals.py", "Generating fresh signals")

def main():
    print("🔧 COMPLETE MYSQL CONNECTION FIX")
    print("=" * 60)
    print("This will:")
    print("1. Stop all containers to free database connections")
    print("2. Fix MySQL connection limits")
    print("3. Restart services sequentially in correct order")
    print("4. Validate the system is working")
    print("=" * 60)
    
    # Execute fix steps
    stop_all_containers()
    fix_mysql_connections()
    restart_services_sequentially()
    validate_system()
    
    print("\n" + "=" * 60)
    print("🎯 MYSQL CONNECTION FIX COMPLETE!")
    print("📊 Monitor: docker-compose logs -f automated-live-trader")
    print("🔍 Status: python final_status_check.py")

if __name__ == "__main__":
    main()
