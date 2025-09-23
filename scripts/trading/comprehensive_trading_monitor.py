#!/usr/bin/env python3
"""
Comprehensive Trading System Monitor
Continuously monitors all trading services, processes, and operations
"""

import os
import sys
import time
import json
import requests
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Any, List
import mysql.connector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - MONITOR - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TradingSystemMonitor:
    def __init__(self):
        self.services = {
            'Trade Recommendations': 'http://localhost:8022',
            'Trade Execution': 'http://localhost:8024',
            'System Monitoring': 'http://localhost:8080'
        }
        
        self.db_config = {
            'host': 'host.docker.internal',
            'user': 'news_collector',
            'password': '99Rules!',
            'database': 'crypto_transactions'
        }
        
        self.last_stats = {
            'recommendations_count': 0,
            'executed_trades': 0,
            'portfolio_value': 0.0,
            'last_signal_time': None
        }
        
        logger.info("Trading System Monitor initialized")
    
    def check_service_health(self, service_name: str, url: str) -> Dict[str, Any]:
        """Check health of a specific service"""
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                return {
                    'service': service_name,
                    'status': 'healthy',
                    'url': url,
                    'response_time': response.elapsed.total_seconds(),
                    'data': response.json()
                }
            else:
                return {
                    'service': service_name,
                    'status': 'unhealthy',
                    'url': url,
                    'error': f'HTTP {response.status_code}'
                }
        except Exception as e:
            return {
                'service': service_name,
                'status': 'unreachable',
                'url': url,
                'error': str(e)
            }
    
    def check_docker_services(self) -> Dict[str, Any]:
        """Check Docker container status"""
        try:
            result = subprocess.run(['docker', 'ps', '--format', 'json'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                containers = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        containers.append(json.loads(line))
                
                trading_containers = [c for c in containers if 
                                    'trade' in c.get('Names', '').lower() or
                                    'recommendation' in c.get('Names', '').lower()]
                
                return {
                    'total_containers': len(containers),
                    'trading_containers': len(trading_containers),
                    'status': 'healthy' if len(containers) >= 25 else 'degraded',
                    'containers': trading_containers
                }
            else:
                return {'status': 'error', 'error': result.stderr}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def check_database_activity(self) -> Dict[str, Any]:
        """Check recent database activity"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Check recent recommendations
            cursor.execute("""
                SELECT COUNT(*) as count, MAX(generated_at) as latest
                FROM trade_recommendations
                WHERE generated_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
            """)
            rec_data = cursor.fetchone()
            
            # Check recent trades (if table exists)
            try:
                cursor.execute("""
                    SELECT COUNT(*) as count, MAX(executed_at) as latest
                    FROM trade_recommendations
                    WHERE executed_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
                    AND execution_status = 'EXECUTED'
                """)
                trade_data = cursor.fetchone()
            except:
                trade_data = (0, None)
            
            cursor.close()
            conn.close()
            
            return {
                'status': 'connected',
                'recent_recommendations': rec_data[0] if rec_data else 0,
                'latest_recommendation': rec_data[1] if rec_data else None,
                'recent_executions': trade_data[0] if trade_data else 0,
                'latest_execution': trade_data[1] if trade_data else None
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def check_active_processes(self) -> Dict[str, Any]:
        """Check for running trading processes"""
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                trading_processes = []
                
                for line in lines:
                    if any(keyword in line.lower() for keyword in 
                          ['automated_live_trader', 'signal_generator', 'trading']):
                        trading_processes.append(line.strip())
                
                return {
                    'status': 'ok',
                    'trading_processes': len(trading_processes),
                    'processes': trading_processes
                }
            else:
                return {'status': 'error', 'error': result.stderr}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def get_trading_stats(self) -> Dict[str, Any]:
        """Get current trading statistics"""
        stats = {}
        
        # Get recommendations count
        try:
            response = requests.get(f"{self.services['Trade Recommendations']}/recommendations?status=pending", timeout=5)
            if response.status_code == 200:
                recommendations = response.json()
                stats['pending_recommendations'] = len(recommendations)
            else:
                stats['pending_recommendations'] = 'error'
        except:
            stats['pending_recommendations'] = 'error'
        
        # Get portfolio status
        try:
            response = requests.get(f"{self.services['Trade Execution']}/portfolio", timeout=5)
            if response.status_code == 200:
                portfolio = response.json()
                stats['portfolio'] = portfolio
            else:
                stats['portfolio'] = 'error'
        except:
            stats['portfolio'] = 'error'
        
        return stats
    
    def log_status_summary(self):
        """Log comprehensive status summary"""
        logger.info("=" * 80)
        logger.info("🚀 COMPREHENSIVE TRADING SYSTEM STATUS REPORT")
        logger.info("=" * 80)
        
        # Check all services
        logger.info("📊 SERVICE HEALTH CHECK:")
        all_healthy = True
        for name, url in self.services.items():
            health = self.check_service_health(name, url)
            if health['status'] == 'healthy':
                logger.info(f"  ✅ {name}: {health['status']} ({health['response_time']:.3f}s)")
            else:
                logger.info(f"  ❌ {name}: {health['status']} - {health.get('error', 'Unknown error')}")
                all_healthy = False
        
        # Check Docker containers
        logger.info("🐳 DOCKER CONTAINERS:")
        docker_status = self.check_docker_services()
        if docker_status['status'] == 'healthy':
            logger.info(f"  ✅ {docker_status['total_containers']} total containers running")
            logger.info(f"  📦 {docker_status['trading_containers']} trading containers active")
        else:
            logger.info(f"  ❌ Docker status: {docker_status.get('error', 'Unknown error')}")
        
        # Check database activity
        logger.info("💾 DATABASE ACTIVITY:")
        db_status = self.check_database_activity()
        if db_status['status'] == 'connected':
            logger.info(f"  ✅ Database connected")
            logger.info(f"  📈 {db_status['recent_recommendations']} recommendations in last hour")
            logger.info(f"  🔄 {db_status['recent_executions']} executions in last hour")
            if db_status['latest_recommendation']:
                logger.info(f"  🕐 Latest recommendation: {db_status['latest_recommendation']}")
        else:
            logger.info(f"  ❌ Database error: {db_status.get('error', 'Unknown error')}")
        
        # Check active processes
        logger.info("⚙️  ACTIVE PROCESSES:")
        process_status = self.check_active_processes()
        if process_status['status'] == 'ok':
            logger.info(f"  ✅ {process_status['trading_processes']} trading processes running")
            for proc in process_status['processes']:
                if 'automated_live_trader' in proc:
                    logger.info(f"  🤖 Live Trader: Active")
                elif 'signal_generator' in proc:
                    logger.info(f"  📊 Signal Generator: Active")
        else:
            logger.info(f"  ❌ Process check error: {process_status.get('error', 'Unknown error')}")
        
        # Get trading statistics
        logger.info("💰 TRADING STATISTICS:")
        stats = self.get_trading_stats()
        if isinstance(stats['pending_recommendations'], int):
            logger.info(f"  📋 Pending recommendations: {stats['pending_recommendations']}")
        else:
            logger.info(f"  ❌ Could not fetch recommendations: {stats['pending_recommendations']}")
        
        if stats['portfolio'] != 'error':
            logger.info(f"  💼 Portfolio status: Available")
        else:
            logger.info(f"  ❌ Portfolio status: Unavailable")
        
        # Overall system status
        logger.info("🎯 OVERALL STATUS:")
        if all_healthy and docker_status['status'] == 'healthy' and db_status['status'] == 'connected':
            logger.info("  🟢 SYSTEM FULLY OPERATIONAL - All systems go!")
        else:
            logger.info("  🟡 SYSTEM DEGRADED - Some issues detected")
        
        logger.info("=" * 80)
    
    def run_continuous_monitoring(self, interval_seconds: int = 300):
        """Run continuous monitoring with periodic status reports"""
        logger.info(f"Starting continuous monitoring (reports every {interval_seconds}s)")
        
        while True:
            try:
                self.log_status_summary()
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(30)

def main():
    monitor = TradingSystemMonitor()
    
    # Run initial status check
    monitor.log_status_summary()
    
    # Ask user if they want continuous monitoring
    print("\n" + "="*50)
    print("TRADING SYSTEM MONITOR")
    print("="*50)
    print("1. One-time status check (completed above)")
    print("2. Continuous monitoring (every 5 minutes)")
    print("3. Exit")
    
    choice = input("\nSelect option (1/2/3): ").strip()
    
    if choice == '2':
        print("Starting continuous monitoring...")
        monitor.run_continuous_monitoring(300)  # 5 minutes
    elif choice == '1':
        print("Status check completed.")
    else:
        print("Exiting...")

if __name__ == "__main__":
    main()
