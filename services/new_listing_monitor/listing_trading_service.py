#!/usr/bin/env python3
"""
Coinbase Listing Trading Service Integration
Main orchestrator that combines monitoring, trading, risk management, and exit strategy

Features:
- Unified service orchestration
- Real-time listing detection and trading
- Comprehensive risk management
- Intelligent exit strategy execution
- Database integration and logging
- Integration with existing trading infrastructure
"""

import asyncio
import time
import logging
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import mysql.connector
from mysql.connector import Error
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import threading

# Import our modules
from coinbase_listing_monitor import CoinbaseListingMonitor
from listing_trading_engine import CoinbaseListingTrader, ListingTrade
from exit_strategy_engine import ExitStrategyEngine
from risk_management_engine import RiskManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CoinbaseListingTradingService:
    """Main service orchestrator for Coinbase listing trading strategy"""
    
    def __init__(self):
        # Initialize all components
        self.listing_monitor = CoinbaseListingMonitor()
        self.trading_engine = CoinbaseListingTrader()
        self.exit_engine = ExitStrategyEngine()
        self.risk_manager = RiskManager()
        
        # Service configuration
        self.monitoring_interval = 60     # Check for new listings every minute
        self.position_monitor_interval = 30  # Monitor positions every 30 seconds
        self.max_concurrent_trades = 5    # Maximum simultaneous listing trades
        
        # Integration with existing trading infrastructure
        self.signal_bridge_url = "http://localhost:8022"
        self.trading_engine_url = "http://localhost:8024"
        self.enhanced_signals_url = "http://localhost:8025"
        
        # FastAPI for health endpoint
        self.app = FastAPI(title="Coinbase Listing Trading Service")
        self.setup_endpoints()
        
        # Background task management
        self.background_tasks = []
        self.service_running = False
        
        # Performance tracking
        self.trades_today = 0
        self.successful_trades = 0
        self.total_pnl = 0.0
        self.start_time = datetime.now()
        
        self.db_config = {
            'host': '192.168.230.163',
            'user': 'news_collector',
            'password': '99Rules!',
            'database': 'crypto_transactions'
        }
        
        # Initialize database tables
        self.initialize_service_tables()
    
    def setup_endpoints(self):
        """Setup FastAPI endpoints for health checks and integration"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            uptime = datetime.now() - self.start_time
            return {
                "status": "healthy",
                "service": "coinbase-listing-trader",
                "uptime_seconds": int(uptime.total_seconds()),
                "trades_today": self.trades_today,
                "active_positions": len(self.trading_engine.get_active_listing_trades()),
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/status")
        async def service_status():
            """Detailed service status"""
            portfolio_risk = self.risk_manager.get_portfolio_risk_metrics()
            uptime = datetime.now() - self.start_time
            
            return {
                "service_info": {
                    "name": "Coinbase Listing Trading Service",
                    "version": "1.0.0",
                    "uptime_hours": round(uptime.total_seconds() / 3600, 2),
                    "status": "running" if self.service_running else "stopped"
                },
                "performance": {
                    "trades_today": self.trades_today,
                    "successful_trades": self.successful_trades,
                    "success_rate_pct": round((self.successful_trades / max(self.trades_today, 1)) * 100, 1),
                    "total_pnl_usd": round(self.total_pnl, 2)
                },
                "portfolio": {
                    "active_listing_trades": portfolio_risk.active_listing_count,
                    "listing_allocation_pct": round(portfolio_risk.listing_allocation_pct * 100, 1),
                    "max_position_pct": round(portfolio_risk.max_single_position_pct * 100, 1),
                    "risk_score": round(portfolio_risk.overall_risk_score, 2)
                },
                "integration": {
                    "signal_bridge": self.check_service_health(self.signal_bridge_url),
                    "trading_engine": self.check_service_health(self.trading_engine_url),
                    "enhanced_signals": self.check_service_health(self.enhanced_signals_url)
                }
            }
        
        @self.app.post("/manual-listing")
        async def manual_listing_trigger(listing_data: dict):
            """Manually trigger listing opportunity processing"""
            try:
                symbol = listing_data.get("symbol")
                if not symbol:
                    raise HTTPException(status_code=400, detail="Symbol required")
                
                # Process the manual listing
                approved, reason, risk_metrics = self.risk_manager.validate_trade_opportunity(symbol, listing_data)
                
                if approved:
                    trade_result = self.trading_engine.execute_listing_buy(listing_data)
                    if trade_result:
                        return {"status": "success", "message": f"Trade executed for {symbol}"}
                    else:
                        return {"status": "failed", "message": f"Trade execution failed for {symbol}"}
                else:
                    return {"status": "rejected", "reason": reason}
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    def check_service_health(self, service_url: str) -> Dict:
        """Check health of dependent services"""
        try:
            response = requests.get(f"{service_url}/health", timeout=5)
            if response.status_code == 200:
                return {"status": "healthy", "response_time_ms": response.elapsed.total_seconds() * 1000}
            else:
                return {"status": "unhealthy", "status_code": response.status_code}
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}
    
    def initialize_service_tables(self):
        """Initialize database tables for the listing trading service"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor()
            
            # Service performance tracking table
            create_performance_table = """
            CREATE TABLE IF NOT EXISTS listing_service_performance (
                id INT AUTO_INCREMENT PRIMARY KEY,
                date DATE NOT NULL,
                trades_executed INT DEFAULT 0,
                successful_trades INT DEFAULT 0,
                total_pnl_usd DECIMAL(15, 4) DEFAULT 0,
                avg_hold_time_minutes INT DEFAULT 0,
                best_performer VARCHAR(20),
                best_performance_pct DECIMAL(8, 4),
                service_uptime_hours DECIMAL(8, 2) DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY unique_date (date)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
            
            # Listing processing log table
            create_processing_log_table = """
            CREATE TABLE IF NOT EXISTS listing_processing_log (
                id INT AUTO_INCREMENT PRIMARY KEY,
                listing_id VARCHAR(100) NOT NULL,
                processing_status VARCHAR(50) NOT NULL,
                processing_notes TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY unique_listing (listing_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
            
            cursor.execute(create_performance_table)
            cursor.execute(create_processing_log_table)
            connection.commit()
            logger.info("✅ Service performance and processing tables ready")
            
        except Error as e:
            logger.error(f"❌ Service table initialization error: {e}")
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def send_to_signal_bridge(self, listing_data: Dict) -> bool:
        """Send listing opportunity to existing signal bridge"""
        try:
            # Format as trading signal for the existing infrastructure
            signal_payload = {
                "symbol": listing_data["symbol"],
                "signal_type": "LISTING_OPPORTUNITY",
                "strength": listing_data.get("confidence", 0.8),
                "direction": "BUY",
                "metadata": {
                    "listing_source": listing_data.get("source", "unknown"),
                    "listing_type": listing_data.get("listing_type", "announced"),
                    "confidence": listing_data.get("confidence", 0.8),
                    "detection_time": datetime.now().isoformat(),
                    "strategy": "coinbase_effect"
                }
            }
            
            # Send to signal bridge
            response = requests.post(
                f"{self.signal_bridge_url}/signals",
                json=signal_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Sent listing signal to bridge: {listing_data['symbol']}")
                return True
            else:
                logger.error(f"❌ Signal bridge rejected signal: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error sending to signal bridge: {e}")
            return False
    
    def generate_listing_signal(self, listing_data: Dict) -> bool:
        """Generate trading signal in our system for listing opportunity"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor()
            
            # Insert into trading_signals table (existing infrastructure)
            insert_signal = """
            INSERT INTO trading_signals 
            (symbol, signal_type, strength, direction, metadata, processed, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            metadata = json.dumps({
                "listing_source": listing_data.get("source", "unknown"),
                "listing_type": listing_data.get("listing_type", "announced"),
                "confidence": listing_data.get("confidence", 0.8),
                "detection_time": datetime.now().isoformat(),
                "strategy": "coinbase_effect",
                "expected_gain_pct": 91,  # Coinbase Effect target
                "max_hold_hours": 72
            })
            
            values = (
                listing_data["symbol"],
                "LISTING_OPPORTUNITY", 
                listing_data.get("confidence", 0.8),
                "BUY",
                metadata,
                False,  # Not processed yet
                datetime.now()
            )
            
            cursor.execute(insert_signal, values)
            connection.commit()
            
            logger.info(f"✅ Generated listing signal: {listing_data['symbol']}")
            return True
            
        except Error as e:
            logger.error(f"❌ Error generating listing signal: {e}")
            return False
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    async def process_new_listings(self):
        """Process newly detected listing opportunities"""
        try:
            # Get unprocessed listing opportunities from monitoring system
            new_listings = await self.listing_monitor.get_unprocessed_listings()
            
            for listing in new_listings:
                try:
                    symbol = listing["symbol"]
                    logger.info(f"🎯 Processing new listing opportunity: {symbol}")
                    
                    # Risk assessment
                    approved, reason, risk_metrics = self.risk_manager.validate_trade_opportunity(symbol, listing)
                    
                    if not approved:
                        logger.warning(f"⚠️ Trade rejected for {symbol}: {reason}")
                        await self.mark_listing_processed(listing["id"], "REJECTED", reason)
                        continue
                    
                    # Send to existing signal bridge for integration
                    bridge_success = self.send_to_signal_bridge(listing)
                    
                    # Also generate signal in our database
                    signal_success = self.generate_listing_signal(listing)
                    
                    # Execute trade directly through our trading engine
                    trade_result = self.trading_engine.execute_listing_buy(listing)
                    
                    if trade_result or bridge_success or signal_success:
                        logger.info(f"✅ Successfully processed listing for {symbol}")
                        await self.mark_listing_processed(listing["id"], "EXECUTED", "Trade successful")
                        self.trades_today += 1
                        
                        # Log risk decision
                        self.risk_manager.log_risk_decision(symbol, True, reason, risk_metrics)
                    else:
                        logger.error(f"❌ Failed to process listing for {symbol}")
                        await self.mark_listing_processed(listing["id"], "FAILED", "All execution methods failed")
                
                except Exception as e:
                    logger.error(f"❌ Error processing listing {listing.get('symbol', 'unknown')}: {e}")
        
        except Exception as e:
            logger.error(f"❌ Error in process_new_listings: {e}")
    
    async def mark_listing_processed(self, listing_id: str, status: str, notes: str):
        """Mark a listing opportunity as processed"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor()
            
            # Update the listing record or create processing log
            update_query = """
            INSERT INTO listing_processing_log 
            (listing_id, processing_status, processing_notes, processed_at)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            processing_status = VALUES(processing_status),
            processing_notes = VALUES(processing_notes),
            processed_at = VALUES(processed_at)
            """
            
            cursor.execute(update_query, (listing_id, status, notes, datetime.now()))
            connection.commit()
            
        except Error as e:
            logger.error(f"❌ Error marking listing processed: {e}")
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def monitor_active_positions(self):
        """Monitor active listing positions for exit opportunities"""
        try:
            # Get active listing trades
            active_trades = self.trading_engine.get_active_listing_trades()
            
            for trade_data in active_trades:
                try:
                    symbol = trade_data["symbol"]
                    
                    # Get current price
                    current_price = self.trading_engine.get_current_price(symbol)
                    if not current_price:
                        continue
                    
                    # Update price history for exit analysis
                    self.trading_engine.update_price_history(symbol, current_price)
                    
                    # Analyze exit opportunity
                    exit_signal = self.exit_engine.analyze_exit_opportunity(
                        symbol=symbol,
                        entry_price=float(trade_data["entry_price"]),
                        current_price=current_price,
                        entry_time=trade_data["entry_time"],
                        price_history=self.trading_engine.price_history.get(symbol, [])
                    )
                    
                    if exit_signal:
                        logger.info(f"🚨 EXIT SIGNAL: {symbol} - {exit_signal.signal_type}")
                        logger.info(f"   Confidence: {exit_signal.confidence:.2f}")
                        logger.info(f"   Action: {exit_signal.recommended_action}")
                        logger.info(f"   Reasoning: {exit_signal.reasoning}")
                        
                        # Execute exit if recommended
                        if exit_signal.recommended_action in ["FULL_EXIT", "PARTIAL_EXIT"]:
                            trade = ListingTrade(
                                symbol=trade_data["symbol"],
                                entry_price=float(trade_data["entry_price"]),
                                entry_time=trade_data["entry_time"],
                                position_size_usd=float(trade_data["position_size_usd"]),
                                listing_source=trade_data["listing_source"],
                                confidence_score=trade_data["confidence_score"]
                            )
                            
                            success = self.trading_engine.execute_listing_sell(
                                trade, current_price, exit_signal.reasoning
                            )
                            
                            if success:
                                self.successful_trades += 1
                                
                                # Calculate P&L for tracking
                                pnl_pct = (current_price - trade.entry_price) / trade.entry_price
                                pnl_usd = (current_price - trade.entry_price) * (trade.position_size_usd / trade.entry_price)
                                self.total_pnl += pnl_usd
                    else:
                        # Log position status
                        pnl_pct = (current_price - float(trade_data["entry_price"])) / float(trade_data["entry_price"])
                        logger.info(f"📊 {symbol}: ${current_price:.4f} ({pnl_pct:+.1%}) - Holding")
                
                except Exception as e:
                    logger.error(f"❌ Error monitoring position {symbol}: {e}")
        
        except Exception as e:
            logger.error(f"❌ Error in monitor_active_positions: {e}")
    
    def update_daily_performance(self):
        """Update daily performance metrics"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor()
            
            # Calculate service uptime
            uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
            
            # Get best performer today
            cursor.execute("""
                SELECT symbol, 
                       ((exit_price - entry_price) / entry_price * 100) as performance_pct
                FROM listing_trades 
                WHERE DATE(entry_time) = CURDATE() 
                AND status = 'EXITED'
                ORDER BY performance_pct DESC 
                LIMIT 1
            """)
            
            best_result = cursor.fetchone()
            best_performer = best_result[0] if best_result else None
            best_performance = float(best_result[1]) if best_result else 0.0
            
            # Calculate average hold time
            cursor.execute("""
                SELECT AVG(TIMESTAMPDIFF(MINUTE, entry_time, exit_time)) as avg_hold_minutes
                FROM listing_trades 
                WHERE DATE(entry_time) = CURDATE() 
                AND status = 'EXITED'
            """)
            
            avg_hold_result = cursor.fetchone()
            avg_hold_minutes = int(avg_hold_result[0]) if avg_hold_result and avg_hold_result[0] else 0
            
            # Update performance record
            update_query = """
            INSERT INTO listing_service_performance 
            (date, trades_executed, successful_trades, total_pnl_usd, avg_hold_time_minutes,
             best_performer, best_performance_pct, service_uptime_hours)
            VALUES (CURDATE(), %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            trades_executed = VALUES(trades_executed),
            successful_trades = VALUES(successful_trades),
            total_pnl_usd = VALUES(total_pnl_usd),
            avg_hold_time_minutes = VALUES(avg_hold_time_minutes),
            best_performer = VALUES(best_performer),
            best_performance_pct = VALUES(best_performance_pct),
            service_uptime_hours = VALUES(service_uptime_hours)
            """
            
            cursor.execute(update_query, (
                self.trades_today, self.successful_trades, self.total_pnl,
                avg_hold_minutes, best_performer, best_performance, uptime_hours
            ))
            connection.commit()
            
        except Error as e:
            logger.error(f"❌ Error updating performance metrics: {e}")
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def log_service_status(self):
        """Log current service status and performance"""
        uptime = datetime.now() - self.start_time
        uptime_hours = uptime.total_seconds() / 3600
        
        # Get portfolio risk metrics
        portfolio_risk = self.risk_manager.get_portfolio_risk_metrics()
        
        logger.info("📊 SERVICE STATUS REPORT")
        logger.info(f"   Uptime: {uptime_hours:.1f} hours")
        logger.info(f"   Trades Today: {self.trades_today}")
        logger.info(f"   Successful Trades: {self.successful_trades}")
        logger.info(f"   Success Rate: {(self.successful_trades/max(self.trades_today,1)*100):.1f}%")
        logger.info(f"   Total P&L: ${self.total_pnl:.2f}")
        logger.info(f"   Active Listing Trades: {portfolio_risk.active_listing_count}")
        logger.info(f"   Listing Allocation: {portfolio_risk.listing_allocation_pct:.1%}")
        logger.info(f"   Portfolio Risk Score: {portfolio_risk.overall_risk_score:.2f}")
    
    async def run_monitoring_cycle(self):
        """Run one complete monitoring cycle"""
        try:
            # Process new listing opportunities
            await self.process_new_listings()
            
            # Monitor active positions
            self.monitor_active_positions()
            
            # Update performance metrics
            self.update_daily_performance()
            
        except Exception as e:
            logger.error(f"❌ Error in monitoring cycle: {e}")
    
    async def start_service(self):
        """Start the listing trading service"""
        logger.info("🚀 COINBASE LISTING TRADING SERVICE STARTING...")
        logger.info("🎯 Strategy: Capture Coinbase Effect (91% average gain)")
        logger.info(f"⚡ Monitoring Interval: {self.monitoring_interval}s")
        logger.info(f"📊 Position Monitor: {self.position_monitor_interval}s")
        logger.info(f"🔗 Integration: Signal Bridge ({self.signal_bridge_url})")
        
        self.service_running = True
        
        # Start FastAPI server in background
        def run_fastapi():
            uvicorn.run(self.app, host="0.0.0.0", port=8033, log_level="info")
        
        fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
        fastapi_thread.start()
        
        # Start background monitoring
        listing_monitor_task = asyncio.create_task(self.listing_monitor.start_monitoring())
        
        cycle_count = 0
        last_status_log = datetime.now()
        
        try:
            # Wait a moment for FastAPI to start
            await asyncio.sleep(2)
            logger.info("✅ Health endpoint available at http://localhost:8033/health")
            
            while True:
                # Run monitoring cycle
                await self.run_monitoring_cycle()
                
                cycle_count += 1
                
                # Log status every 10 minutes
                if datetime.now() - last_status_log > timedelta(minutes=10):
                    self.log_service_status()
                    last_status_log = datetime.now()
                
                # Wait for next cycle
                await asyncio.sleep(self.position_monitor_interval)
        
        except KeyboardInterrupt:
            logger.info("🛑 Service stopped by user")
            self.service_running = False
            listing_monitor_task.cancel()
        except Exception as e:
            logger.error(f"❌ Service error: {e}")
            self.service_running = False
            listing_monitor_task.cancel()

def main():
    """Main entry point for the listing trading service"""
    service = CoinbaseListingTradingService()
    
    # Run the async service
    asyncio.run(service.start_service())

if __name__ == "__main__":
    main()
