#!/usr/bin/env python3
"""
Simple notification helper for trading services
Sends email notifications when trades are executed
"""

import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class SimpleNotificationService:
    """Simple notification service for trade alerts"""
    
    def __init__(self):
        # Email configuration
        self.smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.environ.get('SMTP_PORT', 587))
        self.smtp_username = os.environ.get('SMTP_USERNAME', 'aicrypto.alerts@gmail.com')
        self.smtp_password = os.environ.get('SMTP_PASSWORD', '')
        self.from_email = os.environ.get('FROM_EMAIL', 'aicrypto.alerts@gmail.com')
        self.to_email = os.environ.get('NOTIFICATION_EMAIL', 'epley.mike@gmail.com')
        
        # Notification settings
        self.notifications_enabled = os.environ.get('ENABLE_NOTIFICATIONS', 'true').lower() == 'true'
        
        logger.info(f"Notification service initialized - enabled: {self.notifications_enabled}")
    
    def is_configured(self) -> bool:
        """Check if email notifications are properly configured"""
        return bool(self.smtp_username and self.smtp_password and self.to_email)
    
    def send_trade_notification(self, trade_data: Dict) -> bool:
        """Send notification for trade execution"""
        if not self.notifications_enabled or not self.is_configured():
            logger.debug("Notifications disabled or not configured")
            return False
            
        try:
            # Determine if trade was successful
            success = trade_data.get('success', False)
            symbol = trade_data.get('symbol', 'UNKNOWN')
            action = trade_data.get('action', 'UNKNOWN').upper()
            amount = trade_data.get('size_usd', 0)
            price = trade_data.get('price', 0)
            mode = trade_data.get('mode', 'mock')
            
            # Create subject and message
            if success:
                subject = f"✅ {mode.upper()} Trade Executed: {action} ${amount:.2f} {symbol}"
                priority_color = "#28a745"  # Green
                emoji = "✅"
            else:
                subject = f"❌ {mode.upper()} Trade Failed: {action} ${amount:.2f} {symbol}"
                priority_color = "#dc3545"  # Red  
                emoji = "❌"
                
            # Create detailed message
            message_body = self._create_trade_email_body(trade_data, emoji, priority_color)
            
            # Send email
            return self._send_email(subject, message_body)
            
        except Exception as e:
            logger.error(f"Failed to send trade notification: {e}")
            return False
    
    def send_portfolio_update(self, portfolio_data: Dict) -> bool:
        """Send portfolio status update"""
        if not self.notifications_enabled or not self.is_configured():
            return False
            
        try:
            total_value = portfolio_data.get('total_value_usd', 0)
            positions_count = len(portfolio_data.get('positions', []))
            cash_balance = portfolio_data.get('usd_balance', 0)
            
            subject = f"📊 Portfolio Update: ${total_value:.2f} total value"
            
            message_body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background: #007bff; color: white; padding: 15px; border-radius: 5px;">
                    <h2 style="margin: 0;">📊 Portfolio Status Update</h2>
                    <p style="margin: 5px 0 0 0;">AI Crypto Trading System</p>
                </div>
                
                <div style="padding: 20px; background: #f8f9fa; margin: 10px 0; border-radius: 5px;">
                    <h3>Current Portfolio Status</h3>
                    <p><strong>Total Value:</strong> ${total_value:.2f} USD</p>
                    <p><strong>Cash Balance:</strong> ${cash_balance:.2f} USD</p>
                    <p><strong>Active Positions:</strong> {positions_count}</p>
                    <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                </div>
                
                <div style="padding: 10px; font-size: 12px; color: #666;">
                    <p>AI Crypto Trading System - Automated Portfolio Management</p>
                </div>
            </body>
            </html>
            """
            
            return self._send_email(subject, message_body)
            
        except Exception as e:
            logger.error(f"Failed to send portfolio update: {e}")
            return False
    
    def send_system_alert(self, alert_type: str, message: str, severity: str = "warning") -> bool:
        """Send system alert notification"""
        if not self.notifications_enabled or not self.is_configured():
            return False
            
        try:
            severity_config = {
                "critical": {"emoji": "🔥", "color": "#dc3545"},
                "warning": {"emoji": "⚠️", "color": "#ffc107"},
                "info": {"emoji": "ℹ️", "color": "#007bff"}
            }
            
            config = severity_config.get(severity.lower(), severity_config["warning"])
            subject = f"{config['emoji']} System Alert: {alert_type}"
            
            message_body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background: {config['color']}; color: white; padding: 15px; border-radius: 5px;">
                    <h2 style="margin: 0;">{config['emoji']} System Alert</h2>
                    <p style="margin: 5px 0 0 0;">Severity: {severity.upper()}</p>
                </div>
                
                <div style="padding: 20px; background: #f8f9fa; margin: 10px 0; border-radius: 5px;">
                    <h3>{alert_type}</h3>
                    <p>{message}</p>
                    <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                </div>
                
                <div style="padding: 10px; font-size: 12px; color: #666;">
                    <p>AI Crypto Trading System - System Monitoring</p>
                </div>
            </body>
            </html>
            """
            
            return self._send_email(subject, message_body)
            
        except Exception as e:
            logger.error(f"Failed to send system alert: {e}")
            return False
    
    def _create_trade_email_body(self, trade_data: Dict, emoji: str, color: str) -> str:
        """Create HTML email body for trade notification"""
        symbol = trade_data.get('symbol', 'UNKNOWN')
        action = trade_data.get('action', 'UNKNOWN').upper()
        amount = trade_data.get('size_usd', 0)
        price = trade_data.get('price', 0)
        mode = trade_data.get('mode', 'mock')
        success = trade_data.get('success', False)
        order_id = trade_data.get('order_id', 'N/A')
        error = trade_data.get('error', '')
        
        success_text = "Trade executed successfully!" if success else f"Trade failed: {error}"
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: {color}; color: white; padding: 15px; border-radius: 5px;">
                <h2 style="margin: 0;">{emoji} {mode.upper()} Trade Alert</h2>
                <p style="margin: 5px 0 0 0;">{success_text}</p>
            </div>
            
            <div style="padding: 20px; background: #f8f9fa; margin: 10px 0; border-radius: 5px;">
                <h3>Trade Details</h3>
                <p><strong>Symbol:</strong> {symbol}</p>
                <p><strong>Action:</strong> {action}</p>
                <p><strong>Amount:</strong> ${amount:.2f} USD</p>
                <p><strong>Price:</strong> ${price:.6f}</p>
                <p><strong>Mode:</strong> {mode.upper()}</p>
                <p><strong>Order ID:</strong> {order_id}</p>
                <p><strong>Status:</strong> {'SUCCESSFUL' if success else 'FAILED'}</p>
                <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>
            
            <div style="padding: 10px; font-size: 12px; color: #666;">
                <p>AI Crypto Trading System - Automated Trading Notifications</p>
                <p>Portfolio value updated in real-time. 🚀</p>
            </div>
        </body>
        </html>
        """
    
    def _send_email(self, subject: str, html_body: str) -> bool:
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = self.to_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(html_body, 'html'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"✅ Email notification sent: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send email: {e}")
            return False

# Singleton instance
notification_service = SimpleNotificationService()