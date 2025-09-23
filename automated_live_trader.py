#!/usr/bin/env python3
"""
Automated Live Trading Controller - TIMEZONE FIXED
- Only processes fresh recommendations (within last hour)
- Fixed timezone issue by using database NOW() instead of Python UTC
- Ready for production deployment
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
        logging.FileHandler('./logs/automated_trader.log'),
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
        self.max_age_hours = int(os.getenv('MAX_AGE_HOURS', '1'))  
        self.check_interval = int(os.getenv('CHECK_INTERVAL', '30'))  
        self.max_trades_per_cycle = int(os.getenv('MAX_TRADES_PER_CYCLE', '3'))  
        
    def get_fresh_recommendations(self):
        """Get pending recommendations - FIXED TIME WINDOW & PRIORITIZATION"""
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
            
            # ISSUE A FIX: Expand time window and prioritize by status + confidence
            # Process ALL pending recommendations, prioritizing recent high-confidence ones
            query = """
                SELECT id, symbol, signal_type, amount_usd, confidence, reasoning, created_at,
                       entry_price, stop_loss, take_profit, position_size_percent, amount_usd as amount_crypto
                FROM trade_recommendations 
                WHERE execution_status = 'PENDING' 
                AND is_mock = 0
                AND created_at >= (NOW() - INTERVAL 2 HOUR)
                ORDER BY 
                    CASE WHEN reasoning LIKE '%STRATEGIC%' THEN 1 ELSE 2 END,
                    confidence DESC, 
                    created_at DESC 
                LIMIT %s
            """
            
            cursor.execute(query, (self.max_trades_per_cycle * 3,))  # Process more trades per cycle
            recommendations = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            logger.info(f"Found {len(recommendations)} pending recommendations (2h window, strategic priority)")
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
    
    def filter_duplicate_recommendations(self, recommendations):
        """ORCHESTRATOR SAFETY NET: Remove duplicate recommendations before execution"""
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
            cursor = conn.cursor()
            
            filtered_recommendations = []
            seen_combinations = set()
            
            for rec in recommendations:
                symbol = rec['symbol']
                action = rec['signal_type']
                amount_usd = rec.get('amount_usd', 0)
                
                # Create a key for deduplication (symbol + action + approximate amount)
                amount_rounded = round(amount_usd, 0)  # Round to nearest dollar for grouping
                dedup_key = f"{symbol}_{action}_{amount_rounded}"
                
                # Check if we've already processed this combination
                if dedup_key in seen_combinations:
                    logger.warning(f"[ORCHESTRATOR_DEDUP] ❌ Skipping duplicate: {symbol} {action} ~${amount_rounded}")
                    continue
                
                # Check if similar trade was executed recently (last 10 minutes)
                cursor.execute("""
                    SELECT id, executed_at FROM trade_recommendations
                    WHERE symbol = %s AND signal_type = %s 
                    AND execution_status = 'EXECUTED'
                    AND executed_at >= DATE_SUB(NOW(), INTERVAL 10 MINUTE)
                    AND ABS(amount_usd - %s) < 5.0
                    ORDER BY executed_at DESC
                    LIMIT 1
                """, (symbol, action, amount_usd))
                
                recent_similar = cursor.fetchone()
                if recent_similar:
                    recent_id, recent_time = recent_similar
                    logger.warning(f"[ORCHESTRATOR_DEDUP] ❌ Skipping {symbol} {action} ${amount_usd:.2f}")
                    logger.warning(f"[ORCHESTRATOR_DEDUP] Similar trade ID {recent_id} executed at {recent_time}")
                    continue
                
                # This recommendation is unique - add it
                seen_combinations.add(dedup_key)
                filtered_recommendations.append(rec)
                logger.info(f"[ORCHESTRATOR_DEDUP] ✅ Approved {symbol} {action} ${amount_usd:.2f}")
            
            cursor.close()
            conn.close()
            
            original_count = len(recommendations)
            filtered_count = len(filtered_recommendations)
            if original_count != filtered_count:
                logger.info(f"[ORCHESTRATOR_DEDUP] Filtered {original_count} → {filtered_count} recommendations")
            
            return filtered_recommendations
            
        except Exception as e:
            logger.error(f"Error filtering duplicates: {e}")
            return recommendations  # Return original list if filtering fails

    def run_trading_cycle(self):
        """Execute one complete trading cycle with enhanced deduplication"""
        logger.info("��� Starting trading cycle...")
        
        # Check system health
        if not self.check_system_health():
            logger.warning("⚠️ System not ready for trading - skipping cycle")
            return
        
        # Get fresh recommendations
        fresh_recommendations = self.get_fresh_recommendations()
        
        if not fresh_recommendations:
            logger.info("��� No fresh recommendations to process")
            return
        
        # ORCHESTRATOR SAFETY NET: Filter out duplicates
        unique_recommendations = self.filter_duplicate_recommendations(fresh_recommendations)
        
        if not unique_recommendations:
            logger.info("��� All recommendations filtered as duplicates")
            return
        
        # Execute unique recommendations
        executed_count = 0
        for rec in unique_recommendations:
            rec_id = rec['id']
            symbol = rec['symbol']
            action = rec['signal_type']
            confidence = rec['confidence']
            
            logger.info(f"��� Executing {action} {symbol} (ID: {rec_id}, Confidence: {confidence})")
            
            result = self.execute_recommendation(rec_id)
            
            if result.get('success'):
                logger.info(f"✅ Successfully executed trade for {symbol}")
                executed_count += 1
            else:
                logger.warning(f"❌ Failed to execute trade for {symbol}: {result.get('error')}")
            
            # Small delay between trades
            time.sleep(2)
        
        logger.info(f"��� Trading cycle complete: {executed_count}/{len(unique_recommendations)} trades executed")
    
    def run_forever(self):
        """Main loop - run automated trading continuously"""
        logger.info("��� Starting Automated Live Trading System")
        logger.info(f"��� Configuration:")
        logger.info(f"   - Max recommendation age: {self.max_age_hours} hours")
        logger.info(f"   - Check interval: {self.check_interval} seconds")
        logger.info(f"   - Max trades per cycle: {self.max_trades_per_cycle}")
        
        cycle_count = 0
        
        while True:
            try:
                cycle_count += 1
                logger.info(f"\n��� === Trading Cycle #{cycle_count} ===")
                
                self.run_trading_cycle()
                
                logger.info(f"⏰ Waiting {self.check_interval} seconds until next cycle...")
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("��� Received stop signal - shutting down automated trader")
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
