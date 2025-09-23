#!/usr/bin/env python3
"""
Fix MySQL Connection Issues
Kill idle connections and restart services
"""

import subprocess
import time
import os

def run_command(cmd, description):
    """Run a command safely"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            if result.stdout and len(result.stdout.strip()) > 0:
                print(f"   Output: {result.stdout.strip()[:300]}")
            return True
        else:
            print(f"❌ {description} - Failed (code: {result.returncode})")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()[:300]}")
            return False
    except Exception as e:
        print(f"❌ {description} - Exception: {e}")
        return False

def main():
    print("🔧 FIXING MYSQL CONNECTION ISSUES")
    print("=" * 50)
    
    # Kill any hanging Python processes that might be holding connections
    run_command("taskkill /F /IM python.exe 2>nul || echo 'No python processes'", "Killing hanging Python processes")
    
    # Restart MySQL service (Windows)
    run_command("net stop mysql80 && net start mysql80", "Restarting MySQL service")
    
    # Wait for MySQL to restart
    print("\n⏳ Waiting 15 seconds for MySQL to restart...")
    time.sleep(15)
    
    # Restart trading services
    os.chdir("e:\\git\\aitest\\backend\\services\\trading")
    
    run_command("docker-compose restart trade-recommendations", "Restarting recommendations service")
    
    # Wait for services to start
    print("\n⏳ Waiting 10 seconds for services to start...")
    time.sleep(10)
    
    # Test connectivity
    run_command("curl -s http://localhost:8022/health", "Testing recommendations service")
    
    print("\n" + "=" * 50)
    print("✅ MySQL connection fix completed!")
    print("🔄 Now try generating fresh signals and executing trades")

if __name__ == "__main__":
    main()
