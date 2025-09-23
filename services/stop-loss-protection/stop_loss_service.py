#!/usr/bin/env python3
"""
Stop-Loss Protection Service
Monitors positions and automatically executes stop-loss orders
Provides trailing stop functionality and risk protection
"""

import os
import sys
import asyncio
import logging
import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import aiohttp
import requests
import json
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Stop-Loss Protection Service", version="1.0.0")

class StopLossOrder(BaseModel):
    symbol: str
    position_value: float
    stop_loss_percent: float = 10.0
    trailing_stop: bool = False
    
class StopLossStatus(BaseModel):
    symbol: str
    status: str
    current_price: float
    stop_loss_price: float
    position_value: float
    unrealized_pnl: float
    protection_active: bool

class StopLossProtectionService:
    def __init__(self):
        self.db_config = {
            'host': os.environ.get('DATABASE_HOST', 'host.docker.internal'),
            'user': os.environ.get('DATABASE_USER', 'news_collector'),
            'password': os.environ.get('DATABASE_PASSWORD', '99Rules!'),
            'database': os.environ.get('DATABASE_NAME', 'crypto_transactions'),
            'port': int(os.environ.get('DATABASE_PORT', 3306))
        }
        
        # Trading engine configuration
        self.trade_execution_url = os.environ.get('TRADE_EXECUTION_URL', 'http://localhost:8024')
        self.portfolio_optimization_url = os.environ.get('PORTFOLIO_OPTIMIZATION_URL', 'http://localhost:8026')
        
        # Stop-loss configuration
        self.default_stop_loss_percent = float(os.environ.get('DEFAULT_STOP_LOSS_PERCENT', '10.0'))
        self.min_position_value = float(os.environ.get('MIN_STOP_LOSS_POSITION_VALUE', '100.0'))
        self.trailing_stop_trigger = float(os.environ.get('TRAILING_STOP_TRIGGER', '5.0'))  # Start trailing after 5% gain
        
        # Active stop-loss orders
        self.active_stop_losses = {}
        self.trailing_stops = {}
        
        # Price cache for monitoring
        self.price_cache = {}
        self.last_price_update = {}
        
        logger.info(f"🛡️ Stop-loss protection initialized")
        logger.info(f"   Default stop-loss: {self.default_stop_loss_percent}%")
        logger.info(f"   Min position value: ${self.min_position_value}")
        
    async def get_current_portfolio(self) -> Dict:
        """Get current portfolio from trading engine"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.trade_execution_url}/portfolio") as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        logger.warning(f"Failed to get portfolio: {response.status}")
                        return {}
        except Exception as e:
            logger.error(f"Error getting portfolio: {e}")
            return {}
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Get latest price from ml_features_materialized
            query = """
            SELECT current_price, timestamp_iso
            FROM crypto_prices.ml_features_materialized 
            WHERE symbol = %s 
            ORDER BY timestamp_iso DESC 
            LIMIT 1
            """
            
            cursor.execute(query, (symbol,))
            result = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            if result:
                price, timestamp = result
                self.price_cache[symbol] = float(price)
                self.last_price_update[symbol] = timestamp
                return float(price)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def calculate_stop_loss_price(self, symbol: str, entry_price: float, stop_loss_percent: float) -> float:
        """Calculate stop-loss price"""
        return entry_price * (1 - stop_loss_percent / 100)
    
    def calculate_trailing_stop_price(self, symbol: str, highest_price: float, trailing_percent: float) -> float:
        """Calculate trailing stop price"""
        return highest_price * (1 - trailing_percent / 100)
    
    async def create_stop_loss_order(self, symbol: str, position_data: Dict, stop_loss_percent: float = None, trailing: bool = False) -> bool:
        """Create stop-loss order for position"""
        try:
            if stop_loss_percent is None:
                stop_loss_percent = self.default_stop_loss_percent
            
            position_value = position_data.get('value_usd', 0)
            balance = position_data.get('balance', 0)
            
            if position_value < self.min_position_value:
                logger.debug(f"Position value ${position_value:.2f} below minimum ${self.min_position_value}")
                return False
            
            # Get current price to calculate entry price
            current_price = await self.get_current_price(symbol)
            if not current_price:
                logger.warning(f"Cannot get current price for {symbol}")
                return False
            
            # Estimate entry price from position value and balance
            entry_price = position_value / balance if balance > 0 else current_price
            
            # Calculate stop-loss price
            stop_loss_price = self.calculate_stop_loss_price(symbol, entry_price, stop_loss_percent)
            
            # Create stop-loss order record
            stop_loss_order = {
                'symbol': symbol,
                'entry_price': entry_price,
                'stop_loss_price': stop_loss_price,
                'stop_loss_percent': stop_loss_percent,
                'position_value': position_value,
                'balance': balance,
                'trailing_stop': trailing,
                'highest_price': current_price if trailing else entry_price,
                'created_at': datetime.now(),
                'status': 'ACTIVE'
            }
            
            self.active_stop_losses[symbol] = stop_loss_order
            
            if trailing:
                self.trailing_stops[symbol] = {
                    'highest_price': current_price,
                    'trailing_percent': stop_loss_percent
                }
            
            logger.info(f"🛡️ Created stop-loss order for {symbol}: entry=${entry_price:.4f}, stop=${stop_loss_price:.4f} (-{stop_loss_percent}%)")
            
            # Save to database
            await self.save_stop_loss_to_db(stop_loss_order)
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating stop-loss order for {symbol}: {e}")
            return False
    
    async def save_stop_loss_to_db(self, stop_loss_order: Dict):
        """Save stop-loss order to database"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Create table if not exists
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS stop_loss_orders (
                id INT AUTO_INCREMENT PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                entry_price DECIMAL(20, 8),
                stop_loss_price DECIMAL(20, 8),
                stop_loss_percent DECIMAL(5, 2),
                position_value DECIMAL(20, 2),
                balance DECIMAL(20, 8),
                trailing_stop BOOLEAN DEFAULT FALSE,
                highest_price DECIMAL(20, 8),
                status VARCHAR(20) DEFAULT 'ACTIVE',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                executed_at TIMESTAMP NULL,
                execution_price DECIMAL(20, 8) NULL,
                INDEX idx_symbol_status (symbol, status)
            )
            """
            cursor.execute(create_table_sql)
            
            # Insert stop-loss order
            insert_sql = """
            INSERT INTO stop_loss_orders (
                symbol, entry_price, stop_loss_price, stop_loss_percent,
                position_value, balance, trailing_stop, highest_price, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(insert_sql, (
                stop_loss_order['symbol'],
                stop_loss_order['entry_price'],
                stop_loss_order['stop_loss_price'],
                stop_loss_order['stop_loss_percent'],
                stop_loss_order['position_value'],
                stop_loss_order['balance'],
                stop_loss_order['trailing_stop'],
                stop_loss_order['highest_price'],
                stop_loss_order['status']
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving stop-loss to database: {e}")
    
    async def execute_stop_loss(self, symbol: str, current_price: float) -> bool:
        """Execute stop-loss order"""
        try:
            if symbol not in self.active_stop_losses:
                return False
            
            stop_loss_order = self.active_stop_losses[symbol]
            
            # Execute sell order
            trade_data = {
                'symbol': symbol,
                'action': 'sell',
                'order_type': 'MARKET',
                'size_percentage': 100  # Sell entire position
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.trade_execution_url}/execute_trade",
                    json=trade_data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        if result.get('success', False):
                            # Update stop-loss order status
                            stop_loss_order['status'] = 'EXECUTED'
                            stop_loss_order['executed_at'] = datetime.now()
                            stop_loss_order['execution_price'] = current_price
                            
                            # Remove from active orders
                            del self.active_stop_losses[symbol]
                            if symbol in self.trailing_stops:
                                del self.trailing_stops[symbol]
                            
                            # Update database
                            await self.update_stop_loss_status(symbol, 'EXECUTED', current_price)
                            
                            # Calculate loss
                            entry_price = stop_loss_order['entry_price']
                            loss_percent = ((current_price - entry_price) / entry_price) * 100
                            position_value = stop_loss_order['position_value']
                            
                            logger.warning(f"🔴 STOP-LOSS EXECUTED: {symbol} at ${current_price:.4f} ({loss_percent:.1f}% loss, ~${position_value * abs(loss_percent) / 100:.2f})")
                            
                            return True
                        else:
                            logger.error(f"Stop-loss execution failed for {symbol}: {result}")
                            return False
                    else:
                        logger.error(f"Stop-loss execution request failed: {response.status}")
                        return False
            
        except Exception as e:
            logger.error(f"Error executing stop-loss for {symbol}: {e}")
            return False
    
    async def update_stop_loss_status(self, symbol: str, status: str, execution_price: float = None):
        """Update stop-loss order status in database"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            if execution_price:
                update_sql = """
                UPDATE stop_loss_orders 
                SET status = %s, executed_at = NOW(), execution_price = %s
                WHERE symbol = %s AND status = 'ACTIVE'
                """
                cursor.execute(update_sql, (status, execution_price, symbol))
            else:
                update_sql = """
                UPDATE stop_loss_orders 
                SET status = %s
                WHERE symbol = %s AND status = 'ACTIVE'
                """
                cursor.execute(update_sql, (status, symbol))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating stop-loss status: {e}")
    
    async def update_trailing_stop(self, symbol: str, current_price: float):
        """Update trailing stop price if profit increases"""
        try:
            if symbol not in self.trailing_stops or symbol not in self.active_stop_losses:
                return
            
            trailing_data = self.trailing_stops[symbol]
            stop_loss_order = self.active_stop_losses[symbol]
            
            # Check if current price is higher than recorded highest
            if current_price > trailing_data['highest_price']:
                trailing_data['highest_price'] = current_price
                
                # Update trailing stop price
                new_stop_price = self.calculate_trailing_stop_price(
                    symbol,
                    current_price,
                    trailing_data['trailing_percent']
                )
                
                # Only update if new stop price is higher (more protective)
                if new_stop_price > stop_loss_order['stop_loss_price']:
                    old_stop_price = stop_loss_order['stop_loss_price']
                    stop_loss_order['stop_loss_price'] = new_stop_price
                    stop_loss_order['highest_price'] = current_price
                    
                    logger.info(f"📈 Trailing stop updated for {symbol}: ${old_stop_price:.4f} → ${new_stop_price:.4f}")
                    
        except Exception as e:
            logger.error(f"Error updating trailing stop for {symbol}: {e}")
    
    async def monitor_positions(self):
        """Monitor all positions for stop-loss conditions"""
        try:
            # Get current portfolio
            portfolio = await self.get_current_portfolio()
            positions = portfolio.get('positions', [])
            
            if not positions:
                return
            
            # Check each position (positions is a list, not a dict)
            for position_data in positions:
                if not isinstance(position_data, dict):
                    continue
                
                symbol = position_data.get('currency')
                if not symbol or symbol == 'USD':  # Skip USD
                    continue
                
                current_price = await self.get_current_price(symbol)
                if not current_price:
                    continue
                
                # Create stop-loss order if position doesn't have one
                if symbol not in self.active_stop_losses:
                    position_value = position_data.get('value_usd', 0)
                    if position_value >= self.min_position_value:
                        # Create trailing stop for larger positions
                        trailing = position_value > 200
                        await self.create_stop_loss_order(symbol, position_data, trailing=trailing)
                
                # Check stop-loss conditions
                if symbol in self.active_stop_losses:
                    stop_loss_order = self.active_stop_losses[symbol]
                    stop_loss_price = stop_loss_order['stop_loss_price']
                    
                    # Update trailing stop if applicable
                    if stop_loss_order.get('trailing_stop', False):
                        await self.update_trailing_stop(symbol, current_price)
                        # Refresh stop_loss_price after potential trailing update
                        stop_loss_price = stop_loss_order['stop_loss_price']
                    
                    # Check if stop-loss should be triggered
                    if current_price <= stop_loss_price:
                        logger.warning(f"⚠️ Stop-loss triggered for {symbol}: ${current_price:.4f} <= ${stop_loss_price:.4f}")
                        await self.execute_stop_loss(symbol, current_price)
            
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
    
    async def get_stop_loss_status(self, symbol: str = None) -> Dict:
        """Get status of stop-loss orders"""
        try:
            if symbol:
                if symbol in self.active_stop_losses:
                    order = self.active_stop_losses[symbol]
                    current_price = await self.get_current_price(symbol)
                    
                    return {
                        'symbol': symbol,
                        'status': 'ACTIVE',
                        'entry_price': order['entry_price'],
                        'stop_loss_price': order['stop_loss_price'],
                        'current_price': current_price,
                        'position_value': order['position_value'],
                        'trailing_stop': order.get('trailing_stop', False),
                        'created_at': order['created_at'].isoformat()
                    }
                else:
                    return {
                        'symbol': symbol,
                        'status': 'NO_ACTIVE_ORDER'
                    }
            else:
                # Return all active orders
                status = {}
                for sym, order in self.active_stop_losses.items():
                    current_price = await self.get_current_price(sym)
                    status[sym] = {
                        'status': 'ACTIVE',
                        'entry_price': order['entry_price'],
                        'stop_loss_price': order['stop_loss_price'],
                        'current_price': current_price,
                        'position_value': order['position_value'],
                        'trailing_stop': order.get('trailing_stop', False),
                        'created_at': order['created_at'].isoformat()
                    }
                
                return status
                
        except Exception as e:
            logger.error(f"Error getting stop-loss status: {e}")
            return {}
    
    async def cancel_stop_loss(self, symbol: str) -> bool:
        """Cancel stop-loss order for symbol"""
        try:
            if symbol in self.active_stop_losses:
                del self.active_stop_losses[symbol]
                
                if symbol in self.trailing_stops:
                    del self.trailing_stops[symbol]
                
                # Update database
                await self.update_stop_loss_status(symbol, 'CANCELLED')
                
                logger.info(f"🚫 Cancelled stop-loss order for {symbol}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling stop-loss for {symbol}: {e}")
            return False

# Global stop-loss service instance
stop_loss_service = StopLossProtectionService()

# Background monitoring task
async def background_monitoring():
    """Background task to monitor positions"""
    while True:
        try:
            await stop_loss_service.monitor_positions()
            await asyncio.sleep(30)  # Check every 30 seconds
        except Exception as e:
            logger.error(f"Error in background monitoring: {e}")
            await asyncio.sleep(60)  # Wait longer on error

@app.on_event("startup")
async def startup_event():
    """Start background monitoring on startup"""
    asyncio.create_task(background_monitoring())
    logger.info("🛡️ Stop-loss protection service started")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "stop-loss-protection", "timestamp": datetime.now().isoformat()}

@app.post("/create_stop_loss")
async def create_stop_loss_endpoint(order: StopLossOrder):
    """Create stop-loss order for position"""
    try:
        portfolio = await stop_loss_service.get_current_portfolio()
        positions = portfolio.get('positions', {})
        
        if order.symbol not in positions:
            raise HTTPException(status_code=404, detail=f"No position found for {order.symbol}")
        
        position_data = positions[order.symbol]
        success = await stop_loss_service.create_stop_loss_order(
            order.symbol,
            position_data,
            order.stop_loss_percent,
            order.trailing_stop
        )
        
        if success:
            return {"message": f"Stop-loss order created for {order.symbol}", "success": True}
        else:
            raise HTTPException(status_code=400, detail="Failed to create stop-loss order")
        
    except Exception as e:
        logger.error(f"Error creating stop-loss: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stop_loss_status")
async def get_stop_loss_status_endpoint(symbol: str = None):
    """Get stop-loss status"""
    try:
        status = await stop_loss_service.get_stop_loss_status(symbol)
        return status
        
    except Exception as e:
        logger.error(f"Error getting stop-loss status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cancel_stop_loss/{symbol}")
async def cancel_stop_loss_endpoint(symbol: str):
    """Cancel stop-loss order"""
    try:
        success = await stop_loss_service.cancel_stop_loss(symbol.upper())
        
        if success:
            return {"message": f"Stop-loss order cancelled for {symbol}", "success": True}
        else:
            raise HTTPException(status_code=404, detail=f"No active stop-loss order for {symbol}")
        
    except Exception as e:
        logger.error(f"Error cancelling stop-loss: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/monitor_now")
async def monitor_now_endpoint():
    """Trigger immediate position monitoring"""
    try:
        await stop_loss_service.monitor_positions()
        return {"message": "Position monitoring completed", "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        logger.error(f"Error in manual monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/protection_stats")
async def get_protection_stats():
    """Get stop-loss protection statistics"""
    try:
        conn = mysql.connector.connect(**stop_loss_service.db_config)
        cursor = conn.cursor()
        
        # Get statistics
        stats_query = """
        SELECT 
            status,
            COUNT(*) as count,
            AVG(stop_loss_percent) as avg_stop_loss_percent,
            SUM(position_value) as total_position_value
        FROM stop_loss_orders 
        WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
        GROUP BY status
        """
        
        cursor.execute(stats_query)
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        stats = {}
        for status, count, avg_percent, total_value in results:
            stats[status] = {
                'count': count,
                'avg_stop_loss_percent': float(avg_percent) if avg_percent else 0,
                'total_position_value': float(total_value) if total_value else 0
            }
        
        return {
            'stats': stats,
            'active_orders': len(stop_loss_service.active_stop_losses),
            'trailing_stops': len(stop_loss_service.trailing_stops),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting protection stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "stop_loss_service:app",
        host="0.0.0.0",
        port=8030,
        log_level="info"
    )
