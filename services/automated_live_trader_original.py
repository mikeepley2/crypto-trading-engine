#!/usr/bin/env python3
"""
Automated Live Trading Controller
- Only processes fresh recommendations (within last hour)
- Runs completely within container environment
- Continuous automated execution
"""

import os
import time
import requests
import logging
from datetime import datetime, timedelta
import json
import threading
from fastapi import FastAPI
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/automated_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# FastAPI app for health endpoint
app = FastAPI(title="Automated Live Trader")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "automated_live_trader",
        "timestamp": datetime.now().isoformat()
    }

class AutomatedLiveTrader:
    def __init__(self):
        self.recommendation_service_url = os.getenv('RECOMMENDATION_SERVICE_URL', 'http://localhost:8022')
        self.execution_service_url = os.getenv('EXECUTION_SERVICE_URL', 'http://localhost:8024')
        self.max_age_hours = int(os.getenv('MAX_AGE_HOURS', '1'))  # Only process recommendations within last hour
        self.check_interval = int(os.getenv('CHECK_INTERVAL', '30'))  # Check every 30 seconds
        self.max_trades_per_cycle = int(os.getenv('MAX_TRADES_PER_CYCLE', '3'))  # Limit trades per execution cycle
        
    def get_fresh_recommendations(self):
        """Get only fresh recommendations from the database within the last hour"""
        try:
            import mysql.connector
            
            # Connect to database
            db_config = {
                'host': os.getenv('DB_HOST', 'host.docker.internal'),
                'user': os.getenv('DB_USER', 'news_collector'),
                'password': os.getenv('DB_PASSWORD', '99Rules!'),
                'database': 'crypto_transactions'
            }
            
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor(dictionary=True)
            
            # Query for fresh pending recommendations using database timezone
            # Fixed: use database NOW() instead of Python UTC time to avoid timezone mismatch
            query = """
                SELECT * FROM trade_recommendations 
                WHERE status = 'pending' 
                AND is_mock = 0
                AND generated_at >= (NOW() - INTERVAL %s HOUR)
                ORDER BY generated_at DESC 
                LIMIT %s
            """
            
            cursor.execute(query, (self.max_age_hours, self.max_trades_per_cycle))
            recommendations = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            logger.info(f"Found {len(recommendations)} fresh recommendations (within {self.max_age_hours}h)")
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get fresh recommendations: {e}")
            return []
    
    def execute_recommendation(self, recommendation_id):
        """Execute a single recommendation via the trade execution engine"""
        try:
            url = f"{self.execution_service_url}/process_recommendation/{recommendation_id}"
            response = requests.post(url, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Trade execution result for {recommendation_id}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute recommendation {recommendation_id}: {e}")
            return {'success': False, 'error': str(e)}
    
    def check_system_health(self):
        """Verify execution service is healthy before trading"""
        try:
            # Check execution service
            exec_response = requests.get(f"{self.execution_service_url}/health", timeout=10)
            exec_healthy = exec_response.status_code == 200
            
            if exec_healthy:
                exec_data = exec_response.json()
                trading_enabled = exec_data.get('trading_enabled', False)
                mode = exec_data.get('mode', 'unknown')
                
                logger.info(f"System health: ✅ Services healthy, Trading: {trading_enabled}, Mode: {mode}")
                return trading_enabled and mode == 'live'
            else:
                logger.warning("System health: ❌ Trade execution service unhealthy")
                return False
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def run_trading_cycle(self):
        """Execute one complete trading cycle"""
        logger.info("🔄 Starting trading cycle...")
        
        # Check system health
        if not self.check_system_health():
            logger.warning("⚠️ System not ready for trading - skipping cycle")
            return
        
        # Get fresh recommendations
        fresh_recommendations = self.get_fresh_recommendations()
        
        if not fresh_recommendations:
            logger.info("📭 No fresh recommendations to process")
            return
        
        # Execute recommendations
        executed_count = 0
        for rec in fresh_recommendations:
            rec_id = rec['id']
            symbol = rec['symbol']
            action = rec['action']
            confidence = rec['confidence']
            
            logger.info(f"💰 Executing {action} {symbol} (ID: {rec_id}, Confidence: {confidence})")
            
            result = self.execute_recommendation(rec_id)
            
            if result.get('success'):
                logger.info(f"✅ Successfully executed trade for {symbol}")
                executed_count += 1
            else:
                logger.warning(f"❌ Failed to execute trade for {symbol}: {result.get('error')}")
            
            # Small delay between trades
            time.sleep(2)
        
        logger.info(f"🎯 Trading cycle complete: {executed_count}/{len(fresh_recommendations)} trades executed")
    
    def run_forever(self):
        """Main loop - run automated trading continuously"""
        logger.info("🚀 Starting Automated Live Trading System")
        logger.info(f"📊 Configuration:")
        logger.info(f"   - Max recommendation age: {self.max_age_hours} hours")
        logger.info(f"   - Check interval: {self.check_interval} seconds")
        logger.info(f"   - Max trades per cycle: {self.max_trades_per_cycle}")
        
        cycle_count = 0
        
        while True:
            try:
                cycle_count += 1
                logger.info(f"\n🔁 === Trading Cycle #{cycle_count} ===")
                
                self.run_trading_cycle()
                
                logger.info(f"⏰ Waiting {self.check_interval} seconds until next cycle...")
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 Received stop signal - shutting down automated trader")
                break
            except Exception as e:
                logger.error(f"❌ Unexpected error in trading cycle: {e}")
                logger.info(f"⏰ Waiting {self.check_interval} seconds before retry...")
                time.sleep(self.check_interval)

if __name__ == "__main__":
    # Start FastAPI server in a separate thread
    def run_api():
        uvicorn.run(app, host="0.0.0.0", port=8023, log_level="error")
    
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    
    # Give the API server time to start
    time.sleep(2)
    
    # Start the automated trader
    trader = AutomatedLiveTrader()
    trader.run_forever()
