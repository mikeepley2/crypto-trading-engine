#!/usr/bin/env python3
"""
Live Trading Monitor and Alert System
Comprehensive monitoring, logging, and alerting for live trading operations
"""

import os
import json
import time
import logging
import smtplib
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import mysql.connector
from fastapi import FastAPI, BackgroundTasks
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv('../../../.env.live')

logger = logging.getLogger(__name__)

class TradingMonitor:
    """Comprehensive trading monitoring and alerting system"""
    
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'host.docker.internal'),
            'user': os.getenv('DB_USER', 'news_collector'),
            'password': os.getenv('DB_PASSWORD', '99Rules!'),
            'database': os.getenv('DB_NAME_TRANSACTIONS', 'crypto_transactions')
        }
        
        # Alert configuration
        self.alerts_enabled = os.getenv('ENABLE_TRADING_ALERTS', 'true').lower() == 'true'
        self.alert_email = os.getenv('ALERT_EMAIL')
        self.alert_webhook = os.getenv('ALERT_WEBHOOK_URL')
        
        # Monitoring thresholds
        self.max_daily_loss = float(os.getenv('MAX_DAILY_LOSS_USD', '50.0'))
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE_USD', '100.0'))
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_volume': 0.0,
            'realized_pnl': 0.0,
            'unrealized_pnl': 0.0,
            'win_rate': 0.0,
            'last_updated': datetime.now()
        }
        
        logger.info("✅ Trading Monitor initialized")
    
    def check_system_health(self) -> Dict:
        """Check overall system health"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'services': {},
            'alerts': []
        }
        
        # Check trading engine
        try:
            response = requests.get('http://localhost:8024/health', timeout=5)
            health_status['services']['live_trading_engine'] = {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'response_time': response.elapsed.total_seconds()
            }
        except Exception as e:
            health_status['services']['live_trading_engine'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['alerts'].append('Live Trading Engine unreachable')
        
        # Check recommendation service
        try:
            response = requests.get('http://localhost:8022/health', timeout=5)
            health_status['services']['recommendation_engine'] = {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'response_time': response.elapsed.total_seconds()
            }
        except Exception as e:
            health_status['services']['recommendation_engine'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['alerts'].append('Recommendation Engine unreachable')
        
        # Check database connectivity
        try:
            conn = mysql.connector.connect(**self.db_config)
            conn.close()
            health_status['services']['database'] = {'status': 'healthy'}
        except Exception as e:
            health_status['services']['database'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['alerts'].append('Database connection failed')
        
        # Check for any critical alerts
        if health_status['alerts']:
            health_status['overall_status'] = 'degraded'
        
        return health_status
    
    def analyze_trading_performance(self) -> Dict:
        """Analyze trading performance metrics"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor(dictionary=True)
            
            # Get today's trades
            cursor.execute("""
                SELECT COUNT(*) as total_trades,
                       SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) as successful_trades,
                       SUM(size_usd) as total_volume
                FROM live_trades 
                WHERE DATE(timestamp) = CURDATE()
            """)
            daily_stats = cursor.fetchone()
            
            # Get recent trade history for performance calculation
            cursor.execute("""
                SELECT symbol, action, size_usd, price, timestamp, status
                FROM live_trades 
                WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
                ORDER BY timestamp DESC
            """)
            recent_trades = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            # Calculate performance metrics
            total_trades = daily_stats['total_trades'] or 0
            successful_trades = daily_stats['successful_trades'] or 0
            total_volume = float(daily_stats['total_volume'] or 0)
            
            win_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
            
            performance = {
                'daily_stats': {
                    'total_trades': total_trades,
                    'successful_trades': successful_trades,
                    'failed_trades': total_trades - successful_trades,
                    'total_volume': total_volume,
                    'win_rate': round(win_rate, 2)
                },
                'recent_trades': recent_trades[:10],  # Last 10 trades
                'alerts': []
            }
            
            # Check for performance alerts
            if total_volume > self.max_daily_loss:
                performance['alerts'].append(f"Daily volume ${total_volume:.2f} exceeds threshold ${self.max_daily_loss}")
            
            if win_rate < 30 and total_trades >= 5:
                performance['alerts'].append(f"Low win rate: {win_rate:.1f}% (minimum 30%)")
            
            return performance
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze trading performance: {e}")
            return {'error': str(e)}
    
    def check_risk_limits(self) -> Dict:
        """Check if trading is within risk limits"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor(dictionary=True)
            
            # Check daily trading volume
            cursor.execute("""
                SELECT SUM(size_usd) as daily_volume,
                       COUNT(*) as daily_trades
                FROM live_trades 
                WHERE DATE(timestamp) = CURDATE()
            """)
            daily_limits = cursor.fetchone()
            
            # Check largest single position
            cursor.execute("""
                SELECT MAX(size_usd) as max_position
                FROM live_trades 
                WHERE DATE(timestamp) = CURDATE()
            """)
            position_check = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            daily_volume = float(daily_limits['daily_volume'] or 0)
            daily_trades = daily_limits['daily_trades'] or 0
            max_position = float(position_check['max_position'] or 0)
            
            risk_status = {
                'within_limits': True,
                'daily_volume': daily_volume,
                'daily_trades': daily_trades,
                'max_position': max_position,
                'alerts': []
            }
            
            # Check limits
            if daily_volume > self.max_daily_loss:
                risk_status['within_limits'] = False
                risk_status['alerts'].append(f"Daily volume ${daily_volume:.2f} exceeds limit ${self.max_daily_loss}")
            
            if max_position > self.max_position_size:
                risk_status['within_limits'] = False
                risk_status['alerts'].append(f"Position size ${max_position:.2f} exceeds limit ${self.max_position_size}")
            
            return risk_status
            
        except Exception as e:
            logger.error(f"❌ Failed to check risk limits: {e}")
            return {'error': str(e)}
    
    def send_alert(self, alert_type: str, message: str, severity: str = 'INFO'):
        """Send alert via email or webhook"""
        if not self.alerts_enabled:
            return
        
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'severity': severity,
            'service': 'live_trading_monitor'
        }
        
        # Send email alert
        if self.alert_email:
            self._send_email_alert(alert_data)
        
        # Send webhook alert
        if self.alert_webhook:
            self._send_webhook_alert(alert_data)
        
        logger.info(f"🚨 Alert sent: {alert_type} - {message}")
    
    def _send_email_alert(self, alert_data: Dict):
        """Send email alert"""
        try:
            # This is a placeholder - implement with your email service
            # For production, use services like SendGrid, SES, etc.
            logger.info(f"📧 Email alert: {alert_data['message']}")
        except Exception as e:
            logger.error(f"❌ Failed to send email alert: {e}")
    
    def _send_webhook_alert(self, alert_data: Dict):
        """Send webhook alert"""
        try:
            response = requests.post(
                self.alert_webhook,
                json=alert_data,
                timeout=10
            )
            response.raise_for_status()
            logger.info("✅ Webhook alert sent successfully")
        except Exception as e:
            logger.error(f"❌ Failed to send webhook alert: {e}")
    
    def monitor_trading_session(self):
        """Continuous monitoring of trading session"""
        while True:
            try:
                # Check system health
                health = self.check_system_health()
                if health['alerts']:
                    for alert in health['alerts']:
                        self.send_alert('SYSTEM_HEALTH', alert, 'WARNING')
                
                # Check performance
                performance = self.analyze_trading_performance()
                if performance.get('alerts'):
                    for alert in performance['alerts']:
                        self.send_alert('PERFORMANCE', alert, 'WARNING')
                
                # Check risk limits
                risk_status = self.check_risk_limits()
                if not risk_status.get('within_limits', True):
                    for alert in risk_status.get('alerts', []):
                        self.send_alert('RISK_LIMIT', alert, 'CRITICAL')
                
                # Log monitoring status
                logger.info(f"📊 Monitoring check completed - Health: {health['overall_status']}")
                
                # Wait before next check
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
                time.sleep(60)
    
    def generate_daily_report(self) -> Dict:
        """Generate daily trading report"""
        try:
            performance = self.analyze_trading_performance()
            risk_status = self.check_risk_limits()
            health = self.check_system_health()
            
            report = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'performance': performance,
                'risk_status': risk_status,
                'system_health': health,
                'summary': {
                    'total_trades': performance.get('daily_stats', {}).get('total_trades', 0),
                    'win_rate': performance.get('daily_stats', {}).get('win_rate', 0),
                    'total_volume': performance.get('daily_stats', {}).get('total_volume', 0),
                    'within_risk_limits': risk_status.get('within_limits', True),
                    'system_status': health.get('overall_status', 'unknown')
                }
            }
            
            # Save report to file
            report_file = f"reports/daily_report_{datetime.now().strftime('%Y%m%d')}.json"
            os.makedirs('reports', exist_ok=True)
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📊 Daily report generated: {report_file}")
            return report
            
        except Exception as e:
            logger.error(f"❌ Failed to generate daily report: {e}")
            return {'error': str(e)}

# FastAPI application for monitoring dashboard
app = FastAPI(title="Live Trading Monitor", version="1.0.0")

monitor = TradingMonitor()

@app.get("/health")
async def health_check():
    """Monitor health check"""
    return {
        "status": "healthy",
        "service": "trading_monitor",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/system_health")
async def get_system_health():
    """Get comprehensive system health"""
    return monitor.check_system_health()

@app.get("/performance")
async def get_performance():
    """Get trading performance metrics"""
    return monitor.analyze_trading_performance()

@app.get("/risk_status")
async def get_risk_status():
    """Get current risk status"""
    return monitor.check_risk_limits()

@app.get("/daily_report")
async def get_daily_report():
    """Get daily trading report"""
    return monitor.generate_daily_report()

@app.post("/alert")
async def send_manual_alert(alert_type: str, message: str, severity: str = "INFO"):
    """Send manual alert"""
    monitor.send_alert(alert_type, message, severity)
    return {"status": "alert_sent", "type": alert_type, "message": message}

@app.on_event("startup")
async def startup_event():
    """Start background monitoring"""
    # Start monitoring in background
    asyncio.create_task(run_monitoring())

async def run_monitoring():
    """Run monitoring in background"""
    while True:
        try:
            # Perform monitoring checks
            health = monitor.check_system_health()
            performance = monitor.analyze_trading_performance()
            risk_status = monitor.check_risk_limits()
            
            # Send alerts if needed
            if health['alerts']:
                for alert in health['alerts']:
                    monitor.send_alert('SYSTEM_HEALTH', alert, 'WARNING')
            
            if performance.get('alerts'):
                for alert in performance['alerts']:
                    monitor.send_alert('PERFORMANCE', alert, 'WARNING')
            
            if not risk_status.get('within_limits', True):
                for alert in risk_status.get('alerts', []):
                    monitor.send_alert('RISK_LIMIT', alert, 'CRITICAL')
            
            # Wait before next check
            await asyncio.sleep(60)  # Check every minute
            
        except Exception as e:
            logger.error(f"❌ Background monitoring error: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    import uvicorn
    
    # Start the monitoring server
    port = int(os.getenv('TRADING_MONITOR_PORT', 8025))
    uvicorn.run(app, host="0.0.0.0", port=port)
