#!/usr/bin/env python3
"""
Standalone Automated Live Trader
Connects to existing running services and executes trades based on recommendations
"""

import os
import sys
import time
import json
import logging
import requests
import mysql.connector
from datetime import datetime
from typing import Dict, Any, Optional, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automated_live_trader.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AutomatedLiveTrader:
    def __init__(self):
        self.trade_execution_url = "http://host.docker.internal:8024"
        self.polling_interval = 30  # seconds
        self.running = True
        
        # Database configuration
        self.db_config = {
            'host': 'host.docker.internal',
            'user': 'news_collector',
            'password': '99Rules!',
            'database': 'crypto_transactions'
        }
        
        # Safety limits
        self.max_daily_trades = 200  # Increased for active trading
        self.max_position_size_usd = 100
        self.max_daily_loss_usd = 1000  # Increased for active trading
        
        # Track daily statistics
        self.daily_trades = 0
        self.daily_loss = 0.0
        self.last_reset_date = datetime.now().date()
        
        logger.info("Automated Live Trader initialized")
        logger.info(f"Trade Execution Service: {self.trade_execution_url}")
        logger.info(f"Database: {self.db_config['host']}")
    
    def reset_daily_counters(self):
        """Reset daily counters if new day"""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            logger.info(f"New day detected. Resetting daily counters. Previous: {self.daily_trades} trades, ${self.daily_loss:.2f} loss")
            self.daily_trades = 0
            self.daily_loss = 0.0
            self.last_reset_date = current_date
    
    def check_service_health(self, service_url: str, service_name: str) -> bool:
        """Check if a service is healthy"""
        try:
            response = requests.get(f"{service_url}/health", timeout=5)
            if response.status_code == 200:
                return True
            else:
                logger.warning(f"{service_name} health check failed with status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Failed to check {service_name} health: {e}")
            return False
    
    def get_trade_recommendations(self) -> List[Dict[str, Any]]:
        """Get pending trade recommendations with high confidence directly from database"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor(dictionary=True)
            
            # Get pending recommendations with confidence >= 0.5
            query = """
                SELECT id, symbol, action, confidence, position_size_percent, execution_status, generated_at
                FROM trade_recommendations 
                WHERE execution_status = 'PENDING' 
                AND confidence >= 0.5
                ORDER BY confidence DESC, generated_at DESC
                LIMIT 50
            """
            
            cursor.execute(query)
            recommendations = cursor.fetchall()
            
            # Convert decimal/datetime objects to proper types
            for rec in recommendations:
                if rec['confidence']:
                    rec['confidence'] = float(rec['confidence'])
                if rec['position_size_percent']:
                    rec['position_size_percent'] = float(rec['position_size_percent'])
            
            conn.close()
            logger.info(f"Retrieved {len(recommendations)} high-confidence PENDING recommendations from database")
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get recommendations from database: {e}")
            return []
    
    def execute_trade(self, recommendation: Dict[str, Any]) -> bool:
        """Execute a trade based on recommendation"""
        try:
            # Check safety limits
            if self.daily_trades >= self.max_daily_trades:
                logger.warning("Daily trade limit reached, skipping trade")
                return False
            
            if self.daily_loss >= self.max_daily_loss_usd:
                logger.warning("Daily loss limit reached, skipping trade")
                return False
            
            # Prepare trade request according to the API schema
            position_size = recommendation.get("position_size_percent", "1.0")
            if isinstance(position_size, str):
                try:
                    position_size = float(position_size)
                except (ValueError, TypeError):
                    position_size = 1.0
            
            trade_request = {
                "symbol": recommendation.get("symbol"),
                "action": recommendation.get("action", "").upper(),  # BUY/SELL
                "size_usd": min(position_size * 100, self.max_position_size_usd),
                "order_type": "MARKET"
            }
            
            logger.info(f"Executing trade: {trade_request}")
            
            # Execute trade
            response = requests.post(
                f"{self.trade_execution_url}/execute_trade",
                json=trade_request,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Trade executed successfully: {result}")
                self.daily_trades += 1
                
                # Update recommendation status
                self.update_recommendation_status(recommendation.get("id"), "executed")
                return True
            else:
                logger.error(f"Trade execution failed: status {response.status_code}, response: {response.text}")
                self.update_recommendation_status(recommendation.get("id"), "failed")
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            self.update_recommendation_status(recommendation.get("id"), "error")
            return False
    
    def update_recommendation_status(self, recommendation_id: int, status: str):
        """Update recommendation status directly in database"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            update_query = """
                UPDATE trade_recommendations 
                SET execution_status = %s, updated_at = NOW() 
                WHERE id = %s
            """
            
            cursor.execute(update_query, (status.upper(), recommendation_id))
            conn.commit()
            conn.close()
            
            logger.info(f"Updated recommendation {recommendation_id} status to {status}")
            
        except Exception as e:
            logger.error(f"Failed to update recommendation {recommendation_id} status: {e}")
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        try:
            response = requests.get(f"{self.trade_execution_url}/portfolio", timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get portfolio status: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Error getting portfolio status: {e}")
            return {}
    
    def log_status(self):
        """Log current system status"""
        portfolio = self.get_portfolio_status()
        
        logger.info("=== AUTOMATED TRADER STATUS ===")
        logger.info(f"Daily trades executed: {self.daily_trades}/{self.max_daily_trades}")
        logger.info(f"Daily loss: ${self.daily_loss:.2f}/${self.max_daily_loss_usd}")
        logger.info(f"Portfolio value: ${portfolio.get('total_value', 0):.2f}")
        logger.info(f"Available balance: ${portfolio.get('available_balance', 0):.2f}")
        logger.info("===============================")
    
    def run(self):
        """Main trading loop"""
        logger.info("Starting automated live trader")
        
        while self.running:
            try:
                # Reset daily counters if new day
                self.reset_daily_counters()
                
                # Check trade execution service health
                if not self.check_service_health(self.trade_execution_url, "Trade Execution"):
                    logger.warning("Trade execution service not healthy, waiting before next check")
                    time.sleep(self.polling_interval)
                    continue
                
                # Get and process recommendations
                recommendations = self.get_trade_recommendations()
                
                if recommendations:
                    logger.info(f"Processing {len(recommendations)} recommendations")
                    
                    for recommendation in recommendations:
                        # Comprehensive confidence validation and handling
                        confidence = recommendation.get("confidence", 0)
                        rec_id = recommendation.get("id", "unknown")
                        symbol = recommendation.get("symbol", "unknown")
                        action = recommendation.get("action", "unknown")
                        
                        # Robust confidence parsing
                        if confidence is None:
                            logger.warning(f"Recommendation {rec_id} ({symbol} {action}) has NULL confidence - marking as corrupted")
                            self.update_recommendation_status(rec_id, "corrupted_confidence")
                            continue
                        
                        if isinstance(confidence, str):
                            try:
                                confidence = float(confidence)
                            except (ValueError, TypeError):
                                logger.warning(f"Recommendation {rec_id} ({symbol} {action}) has invalid confidence string '{confidence}' - marking as corrupted")
                                self.update_recommendation_status(rec_id, "corrupted_confidence")
                                continue
                        
                        # Ensure confidence is a valid number
                        try:
                            confidence = float(confidence)
                        except (ValueError, TypeError):
                            logger.warning(f"Recommendation {rec_id} ({symbol} {action}) has non-numeric confidence '{confidence}' - marking as corrupted")
                            self.update_recommendation_status(rec_id, "corrupted_confidence")
                            continue
                        
                        # Validate confidence range
                        if confidence < 0 or confidence > 1:
                            logger.warning(f"Recommendation {rec_id} ({symbol} {action}) has out-of-range confidence {confidence} - marking as invalid")
                            self.update_recommendation_status(rec_id, "invalid_confidence")
                            continue
                        
                        # Log all recommendations for debugging
                        logger.info(f"Processing recommendation {rec_id}: {symbol} {action} with confidence {confidence:.4f}")
                        
                        if confidence >= 0.5:  # Execute recommendations with confidence >= 50%
                            logger.info(f"Executing high-confidence trade: {symbol} {action} (confidence: {confidence:.4f})")
                            success = self.execute_trade(recommendation)
                            if success:
                                logger.info(f"✅ Successfully executed trade for {symbol} {action}")
                            else:
                                logger.warning(f"❌ Failed to execute trade for {symbol} {action}")
                        else:
                            logger.info(f"⏭️ Skipping low confidence recommendation: {symbol} {action} (confidence: {confidence:.4f})")
                            self.update_recommendation_status(rec_id, "skipped_low_confidence")
                else:
                    logger.info("No pending recommendations to process")
                
                # Log status every few iterations
                if (time.time() // self.polling_interval) % 10 == 0:  # Every 5 minutes
                    self.log_status()
                
                # Wait before next iteration
                logger.info(f"Waiting {self.polling_interval} seconds before next check...")
                time.sleep(self.polling_interval)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                time.sleep(self.polling_interval)
        
        logger.info("Automated live trader stopped")

def main():
    """Main entry point"""
    trader = AutomatedLiveTrader()
    trader.run()

if __name__ == "__main__":
    main()
