#!/usr/bin/env python3
"""
LLM Quota Notification Service
Monitors LLM API usage and sends alerts when quotas are exceeded.
"""

import os
import json
import smtplib
import requests
import logging
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NotificationConfig:
    """Notification configuration."""
    webhook_url: Optional[str] = None
    email_smtp_server: Optional[str] = None
    email_smtp_port: int = 587
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    admin_emails: List[str] = None
    slack_webhook: Optional[str] = None
    discord_webhook: Optional[str] = None

class LLMQuotaNotificationService:
    """
    Service for sending notifications about LLM quota issues.
    """
    
    def __init__(self, config: NotificationConfig = None):
        self.config = config or self._load_config_from_env()
        self.notification_log = Path("temp/notifications.log")
        self.notification_log.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("📢 LLM Quota Notification Service initialized")
    
    def _load_config_from_env(self) -> NotificationConfig:
        """Load notification configuration from environment variables."""
        
        admin_emails = []
        if os.getenv('ADMIN_EMAIL'):
            admin_emails = [email.strip() for email in os.getenv('ADMIN_EMAIL').split(',')]
        
        return NotificationConfig(
            webhook_url=os.getenv('NOTIFICATION_WEBHOOK_URL'),
            email_smtp_server=os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com'),
            email_smtp_port=int(os.getenv('EMAIL_SMTP_PORT', '587')),
            email_username=os.getenv('EMAIL_USERNAME'),
            email_password=os.getenv('EMAIL_PASSWORD'),
            admin_emails=admin_emails,
            slack_webhook=os.getenv('SLACK_WEBHOOK_URL'),
            discord_webhook=os.getenv('DISCORD_WEBHOOK_URL')
        )
    
    async def send_quota_exceeded_alert(self, provider: str, error_message: str, quota_details: Dict = None):
        """Send comprehensive quota exceeded alert through all configured channels."""
        
        alert_data = {
            "alert_type": "quota_exceeded",
            "provider": provider,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat(),
            "system": "CryptoAI Trading System",
            "severity": "HIGH",
            "quota_details": quota_details or {},
            "action_required": f"Increase {provider} API quota or check billing",
            "fallback_status": "Local rule-based analysis activated",
            "impact": "LLM-based risk analysis temporarily unavailable"
        }
        
        # Log the alert
        self._log_notification(alert_data)
        
        # Send through all available channels
        results = {}
        
        if self.config.webhook_url:
            results['webhook'] = await self._send_webhook_notification(alert_data)
        
        if self.config.admin_emails:
            results['email'] = await self._send_email_notification(alert_data)
        
        if self.config.slack_webhook:
            results['slack'] = await self._send_slack_notification(alert_data)
        
        if self.config.discord_webhook:
            results['discord'] = await self._send_discord_notification(alert_data)
        
        # Console alert (always available)
        self._print_console_alert(alert_data)
        
        logger.info(f"📢 Quota alert sent through {len([r for r in results.values() if r])} channels")
        
        return results
    
    async def send_quota_warning(self, provider: str, usage_percentage: float, quota_details: Dict = None):
        """Send warning when quota usage reaches threshold (e.g., 80%)."""
        
        alert_data = {
            "alert_type": "quota_warning",
            "provider": provider,
            "usage_percentage": usage_percentage,
            "timestamp": datetime.now().isoformat(),
            "system": "CryptoAI Trading System",
            "severity": "MEDIUM",
            "quota_details": quota_details or {},
            "action_required": f"Monitor {provider} API usage - approaching quota limit",
            "threshold": "80%",
            "current_usage": f"{usage_percentage:.1f}%"
        }
        
        # Log the warning
        self._log_notification(alert_data)
        
        # Send lighter notifications for warnings
        results = {}
        
        if self.config.webhook_url:
            results['webhook'] = await self._send_webhook_notification(alert_data)
        
        # Only send email if usage is very high (90%+)
        if usage_percentage >= 90 and self.config.admin_emails:
            results['email'] = await self._send_email_notification(alert_data)
        
        logger.info(f"⚠️ Quota warning sent for {provider} at {usage_percentage:.1f}% usage")
        
        return results
    
    async def send_system_recovery_notification(self, provider: str, recovery_details: Dict = None):
        """Send notification when LLM service recovers."""
        
        alert_data = {
            "alert_type": "system_recovery",
            "provider": provider,
            "timestamp": datetime.now().isoformat(),
            "system": "CryptoAI Trading System",
            "severity": "INFO",
            "recovery_details": recovery_details or {},
            "status": f"{provider} API service restored",
            "impact": "Full LLM-based risk analysis restored"
        }
        
        # Log the recovery
        self._log_notification(alert_data)
        
        # Send recovery notification
        if self.config.webhook_url:
            await self._send_webhook_notification(alert_data)
        
        logger.info(f"✅ Recovery notification sent for {provider}")
    
    async def _send_webhook_notification(self, alert_data: Dict) -> bool:
        """Send notification to generic webhook."""
        try:
            response = requests.post(
                self.config.webhook_url,
                json=alert_data,
                timeout=10,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code < 300:
                logger.info(f"✅ Webhook notification sent successfully")
                return True
            else:
                logger.error(f"❌ Webhook notification failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Webhook notification error: {e}")
            return False
    
    async def _send_email_notification(self, alert_data: Dict) -> bool:
        """Send email notification."""
        if not self.config.admin_emails or not self.config.email_username:
            return False
        
        try:
            # Create email content
            subject = f"🚨 CryptoAI Alert: {alert_data['alert_type'].title()} - {alert_data['provider']}"
            
            body = self._format_email_body(alert_data)
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config.email_username
            msg['To'] = ', '.join(self.config.admin_emails)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(self.config.email_smtp_server, self.config.email_smtp_port)
            server.starttls()
            server.login(self.config.email_username, self.config.email_password)
            
            for email in self.config.admin_emails:
                server.sendmail(self.config.email_username, email, msg.as_string())
            
            server.quit()
            
            logger.info(f"✅ Email notification sent to {len(self.config.admin_emails)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"❌ Email notification error: {e}")
            return False
    
    async def _send_slack_notification(self, alert_data: Dict) -> bool:
        """Send Slack notification."""
        try:
            slack_payload = {
                "text": f"🚨 CryptoAI Alert: {alert_data['alert_type'].title()}",
                "attachments": [
                    {
                        "color": self._get_alert_color(alert_data['severity']),
                        "fields": [
                            {
                                "title": "Provider",
                                "value": alert_data['provider'],
                                "short": True
                            },
                            {
                                "title": "Severity",
                                "value": alert_data['severity'],
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert_data['timestamp'],
                                "short": True
                            },
                            {
                                "title": "Action Required",
                                "value": alert_data.get('action_required', 'None'),
                                "short": False
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(
                self.config.slack_webhook,
                json=slack_payload,
                timeout=10
            )
            
            if response.status_code < 300:
                logger.info(f"✅ Slack notification sent successfully")
                return True
            else:
                logger.error(f"❌ Slack notification failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Slack notification error: {e}")
            return False
    
    async def _send_discord_notification(self, alert_data: Dict) -> bool:
        """Send Discord notification."""
        try:
            discord_payload = {
                "embeds": [
                    {
                        "title": f"🚨 CryptoAI Alert: {alert_data['alert_type'].title()}",
                        "description": f"**Provider:** {alert_data['provider']}\n**Severity:** {alert_data['severity']}",
                        "color": self._get_discord_color(alert_data['severity']),
                        "fields": [
                            {
                                "name": "Action Required",
                                "value": alert_data.get('action_required', 'None'),
                                "inline": False
                            },
                            {
                                "name": "Time",
                                "value": alert_data['timestamp'],
                                "inline": True
                            }
                        ],
                        "footer": {
                            "text": "CryptoAI Trading System"
                        }
                    }
                ]
            }
            
            response = requests.post(
                self.config.discord_webhook,
                json=discord_payload,
                timeout=10
            )
            
            if response.status_code < 300:
                logger.info(f"✅ Discord notification sent successfully")
                return True
            else:
                logger.error(f"❌ Discord notification failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Discord notification error: {e}")
            return False
    
    def _format_email_body(self, alert_data: Dict) -> str:
        """Format HTML email body."""
        
        severity_color = {
            'HIGH': '#FF4444',
            'MEDIUM': '#FF8800',
            'LOW': '#44FF44',
            'INFO': '#4488FF'
        }.get(alert_data['severity'], '#888888')
        
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 20px;">
            <div style="border-left: 5px solid {severity_color}; padding-left: 15px;">
                <h2 style="color: {severity_color};">🚨 CryptoAI Trading System Alert</h2>
                
                <table style="border-collapse: collapse; width: 100%; margin: 20px 0;">
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px; background-color: #f9f9f9; font-weight: bold;">Alert Type</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{alert_data['alert_type'].title()}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px; background-color: #f9f9f9; font-weight: bold;">Provider</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{alert_data['provider']}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px; background-color: #f9f9f9; font-weight: bold;">Severity</td>
                        <td style="border: 1px solid #ddd; padding: 8px; color: {severity_color}; font-weight: bold;">{alert_data['severity']}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px; background-color: #f9f9f9; font-weight: bold;">Time</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{alert_data['timestamp']}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px; background-color: #f9f9f9; font-weight: bold;">Action Required</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{alert_data.get('action_required', 'None')}</td>
                    </tr>
                </table>
        """
        
        if 'error_message' in alert_data:
            html_body += f"""
                <h3>Error Details:</h3>
                <div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; font-family: monospace;">
                    {alert_data['error_message']}
                </div>
            """
        
        if 'quota_details' in alert_data and alert_data['quota_details']:
            html_body += f"""
                <h3>Quota Details:</h3>
                <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px;">
                    <pre>{json.dumps(alert_data['quota_details'], indent=2)}</pre>
                </div>
            """
        
        html_body += """
                <hr style="margin: 20px 0;">
                <p style="color: #666; font-size: 12px;">
                    This alert was generated by the CryptoAI Trading System LLM Quota Monitor.<br>
                    Please take appropriate action to maintain system reliability.
                </p>
            </div>
        </body>
        </html>
        """
        
        return html_body
    
    def _get_alert_color(self, severity: str) -> str:
        """Get color for Slack attachments."""
        colors = {
            'HIGH': 'danger',
            'MEDIUM': 'warning',
            'LOW': 'good',
            'INFO': '#36a64f'
        }
        return colors.get(severity, '#808080')
    
    def _get_discord_color(self, severity: str) -> int:
        """Get color for Discord embeds (decimal)."""
        colors = {
            'HIGH': 0xFF4444,
            'MEDIUM': 0xFF8800,
            'LOW': 0x44FF44,
            'INFO': 0x4488FF
        }
        return colors.get(severity, 0x808080)
    
    def _print_console_alert(self, alert_data: Dict):
        """Print alert to console with formatting."""
        
        severity_symbols = {
            'HIGH': '🚨',
            'MEDIUM': '⚠️',
            'LOW': '💡',
            'INFO': 'ℹ️'
        }
        
        symbol = severity_symbols.get(alert_data['severity'], '📢')
        
        print(f"\n{'='*60}")
        print(f"{symbol} CRYPTOAI TRADING SYSTEM ALERT {symbol}")
        print(f"{'='*60}")
        print(f"Alert Type: {alert_data['alert_type'].title()}")
        print(f"Provider: {alert_data['provider']}")
        print(f"Severity: {alert_data['severity']}")
        print(f"Time: {alert_data['timestamp']}")
        
        if 'action_required' in alert_data:
            print(f"Action Required: {alert_data['action_required']}")
        
        if 'error_message' in alert_data:
            print(f"Error: {alert_data['error_message']}")
        
        if 'fallback_status' in alert_data:
            print(f"Fallback Status: {alert_data['fallback_status']}")
        
        print(f"{'='*60}\n")
    
    def _log_notification(self, alert_data: Dict):
        """Log notification to file."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "alert_data": alert_data
            }
            
            with open(self.notification_log, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            logger.error(f"Failed to log notification: {e}")
    
    def get_notification_history(self, hours: int = 24) -> List[Dict]:
        """Get recent notification history."""
        
        if not self.notification_log.exists():
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_notifications = []
        
        try:
            with open(self.notification_log, "r") as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        entry_time = datetime.fromisoformat(log_entry['timestamp'])
                        
                        if entry_time >= cutoff_time:
                            recent_notifications.append(log_entry)
                            
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to read notification history: {e}")
        
        return recent_notifications

# Example usage
async def main():
    """Example usage of notification service."""
    
    # Create configuration
    config = NotificationConfig(
        webhook_url="https://your-webhook-url.com/alerts",
        admin_emails=["admin@yourcompany.com"],
        email_smtp_server="smtp.gmail.com",
        email_username="alerts@yourcompany.com",
        email_password="your-app-password"
    )
    
    # Initialize service
    notification_service = LLMQuotaNotificationService(config)
    
    # Send test quota exceeded alert
    await notification_service.send_quota_exceeded_alert(
        provider="grok",
        error_message="Rate limit exceeded: 429 Too Many Requests",
        quota_details={
            "requests_today": 1000,
            "tokens_today": 50000,
            "quota_limit": 1000,
            "reset_time": "2025-01-02T00:00:00Z"
        }
    )
    
    # Send quota warning
    await notification_service.send_quota_warning(
        provider="openai",
        usage_percentage=85.5,
        quota_details={
            "requests_today": 855,
            "quota_limit": 1000
        }
    )
    
    # Get notification history
    history = notification_service.get_notification_history(hours=24)
    print(f"Recent notifications: {len(history)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
