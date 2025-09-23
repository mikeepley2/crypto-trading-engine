#!/usr/bin/env python3
"""
Multi-Platform Database Integration
Handles database operations for multi-platform trading data
"""

import mysql.connector
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal

from .platforms import TradingPlatform, OrderResponse, TradeExecution, AssetBalance

logger = logging.getLogger(__name__)

class MultiPlatformDatabase:
    """Database operations for multi-platform trading"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.connection = None
        
    def connect(self) -> bool:
        """Connect to the database"""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            logger.info("Connected to multi-platform trading database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the database"""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def ensure_connection(self) -> bool:
        """Ensure database connection is active"""
        try:
            if not self.connection or not self.connection.is_connected():
                return self.connect()
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False
    
    def get_platform_id(self, platform: TradingPlatform) -> Optional[int]:
        """Get platform ID from database"""
        if not self.ensure_connection():
            return None
        
        try:
            cursor = self.connection.cursor()
            query = "SELECT platform_id FROM trading_platforms WHERE platform_name = %s"
            cursor.execute(query, (platform.value,))
            result = cursor.fetchone()
            cursor.close()
            
            return result[0] if result else None
            
        except Exception as e:
            logger.error(f"Failed to get platform ID for {platform.value}: {e}")
            return None
    
    def ensure_platform_exists(self, platform: TradingPlatform) -> int:
        """Ensure platform exists in database and return its ID"""
        platform_id = self.get_platform_id(platform)
        
        if platform_id is not None:
            return platform_id
        
        # Create platform record
        if not self.ensure_connection():
            raise Exception("Database connection failed")
        
        try:
            cursor = self.connection.cursor()
            
            insert_query = """
                INSERT INTO trading_platforms (platform_name, display_name, is_active, created_at)
                VALUES (%s, %s, %s, %s)
            """
            
            cursor.execute(insert_query, (
                platform.value,
                platform.value.replace('_', ' ').title(),
                True,
                datetime.now()
            ))
            
            platform_id = cursor.lastrowid
            self.connection.commit()
            cursor.close()
            
            logger.info(f"Created platform record for {platform.value} with ID {platform_id}")
            return platform_id
            
        except Exception as e:
            logger.error(f"Failed to create platform record for {platform.value}: {e}")
            raise
    
    def save_trade_record(self, trade: TradeExecution, platform: TradingPlatform) -> bool:
        """Save trade execution record to database"""
        if not self.ensure_connection():
            return False
        
        try:
            platform_id = self.ensure_platform_exists(platform)
            
            cursor = self.connection.cursor()
            
            insert_query = """
                INSERT INTO trades (
                    platform_id, platform_trade_id, platform_order_id, symbol, 
                    side, quantity, price, fee, fee_asset, executed_at, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(insert_query, (
                platform_id,
                trade.trade_id,
                trade.order_id,
                trade.symbol,
                trade.side.value,
                float(trade.quantity),
                float(trade.price),
                float(trade.fee),
                trade.fee_asset,
                trade.executed_at,
                datetime.now()
            ))
            
            self.connection.commit()
            cursor.close()
            
            logger.info(f"Saved trade record {trade.trade_id} for {platform.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save trade record: {e}")
            return False
    
    def save_order_record(self, order: OrderResponse, platform: TradingPlatform) -> bool:
        """Save order record to database"""
        if not self.ensure_connection():
            return False
        
        try:
            platform_id = self.ensure_platform_exists(platform)
            
            cursor = self.connection.cursor()
            
            # Check if order already exists
            check_query = """
                SELECT order_id FROM orders 
                WHERE platform_id = %s AND platform_order_id = %s
            """
            cursor.execute(check_query, (platform_id, order.platform_order_id))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing order
                update_query = """
                    UPDATE orders SET
                        status = %s, filled_quantity = %s, remaining_quantity = %s,
                        average_price = %s, total_fee = %s, updated_at = %s
                    WHERE platform_id = %s AND platform_order_id = %s
                """
                
                cursor.execute(update_query, (
                    order.status.value,
                    float(order.filled_quantity),
                    float(order.remaining_quantity),
                    float(order.average_price) if order.average_price else None,
                    float(order.total_fee),
                    datetime.now(),
                    platform_id,
                    order.platform_order_id
                ))
            else:
                # Insert new order
                insert_query = """
                    INSERT INTO orders (
                        platform_id, platform_order_id, client_order_id, symbol,
                        side, order_type, quantity, price, status, filled_quantity,
                        remaining_quantity, average_price, total_fee, fee_asset,
                        created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                cursor.execute(insert_query, (
                    platform_id,
                    order.platform_order_id,
                    order.client_order_id,
                    order.symbol,
                    order.side.value,
                    order.order_type.value,
                    float(order.quantity),
                    float(order.price) if order.price else None,
                    order.status.value,
                    float(order.filled_quantity),
                    float(order.remaining_quantity),
                    float(order.average_price) if order.average_price else None,
                    float(order.total_fee),
                    order.fee_asset,
                    order.created_at,
                    datetime.now()
                ))
            
            self.connection.commit()
            cursor.close()
            
            logger.info(f"Saved order record {order.platform_order_id} for {platform.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save order record: {e}")
            return False
    
    def update_portfolio_position(self, platform: TradingPlatform, asset: str, 
                                 balance: AssetBalance) -> bool:
        """Update portfolio position for an asset on a platform"""
        if not self.ensure_connection():
            return False
        
        try:
            platform_id = self.ensure_platform_exists(platform)
            
            cursor = self.connection.cursor()
            
            # Check if position already exists
            check_query = """
                SELECT position_id FROM portfolio_positions 
                WHERE platform_id = %s AND asset = %s
            """
            cursor.execute(check_query, (platform_id, asset))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing position
                update_query = """
                    UPDATE portfolio_positions SET
                        free_quantity = %s, locked_quantity = %s, total_quantity = %s,
                        updated_at = %s
                    WHERE platform_id = %s AND asset = %s
                """
                
                cursor.execute(update_query, (
                    float(balance.free),
                    float(balance.locked),
                    float(balance.total),
                    datetime.now(),
                    platform_id,
                    asset
                ))
            else:
                # Insert new position
                insert_query = """
                    INSERT INTO portfolio_positions (
                        platform_id, asset, free_quantity, locked_quantity, 
                        total_quantity, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                
                cursor.execute(insert_query, (
                    platform_id,
                    asset,
                    float(balance.free),
                    float(balance.locked),
                    float(balance.total),
                    datetime.now(),
                    datetime.now()
                ))
            
            self.connection.commit()
            cursor.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update portfolio position: {e}")
            return False
    
    def get_portfolio_positions(self, platform: Optional[TradingPlatform] = None) -> List[Dict[str, Any]]:
        """Get portfolio positions from database"""
        if not self.ensure_connection():
            return []
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            if platform:
                platform_id = self.get_platform_id(platform)
                if not platform_id:
                    return []
                
                query = """
                    SELECT pp.*, tp.platform_name 
                    FROM portfolio_positions pp
                    JOIN trading_platforms tp ON pp.platform_id = tp.platform_id
                    WHERE pp.platform_id = %s AND pp.total_quantity > 0
                    ORDER BY pp.updated_at DESC
                """
                cursor.execute(query, (platform_id,))
            else:
                query = """
                    SELECT pp.*, tp.platform_name 
                    FROM portfolio_positions pp
                    JOIN trading_platforms tp ON pp.platform_id = tp.platform_id
                    WHERE pp.total_quantity > 0
                    ORDER BY tp.platform_name, pp.asset
                """
                cursor.execute(query)
            
            positions = cursor.fetchall()
            cursor.close()
            
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get portfolio positions: {e}")
            return []
    
    def get_recent_trades(self, platform: Optional[TradingPlatform] = None, 
                         limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades from database"""
        if not self.ensure_connection():
            return []
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            if platform:
                platform_id = self.get_platform_id(platform)
                if not platform_id:
                    return []
                
                query = """
                    SELECT t.*, tp.platform_name 
                    FROM trades t
                    JOIN trading_platforms tp ON t.platform_id = tp.platform_id
                    WHERE t.platform_id = %s
                    ORDER BY t.executed_at DESC
                    LIMIT %s
                """
                cursor.execute(query, (platform_id, limit))
            else:
                query = """
                    SELECT t.*, tp.platform_name 
                    FROM trades t
                    JOIN trading_platforms tp ON t.platform_id = tp.platform_id
                    ORDER BY t.executed_at DESC
                    LIMIT %s
                """
                cursor.execute(query, (limit,))
            
            trades = cursor.fetchall()
            cursor.close()
            
            return trades
            
        except Exception as e:
            logger.error(f"Failed to get recent trades: {e}")
            return []
    
    def get_open_orders_from_db(self, platform: Optional[TradingPlatform] = None) -> List[Dict[str, Any]]:
        """Get open orders from database"""
        if not self.ensure_connection():
            return []
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            open_statuses = ['new', 'partially_filled', 'pending_cancel']
            
            if platform:
                platform_id = self.get_platform_id(platform)
                if not platform_id:
                    return []
                
                query = """
                    SELECT o.*, tp.platform_name 
                    FROM orders o
                    JOIN trading_platforms tp ON o.platform_id = tp.platform_id
                    WHERE o.platform_id = %s AND o.status IN ({})
                    ORDER BY o.created_at DESC
                """.format(','.join(['%s'] * len(open_statuses)))
                
                cursor.execute(query, (platform_id, *open_statuses))
            else:
                query = """
                    SELECT o.*, tp.platform_name 
                    FROM orders o
                    JOIN trading_platforms tp ON o.platform_id = tp.platform_id
                    WHERE o.status IN ({})
                    ORDER BY tp.platform_name, o.created_at DESC
                """.format(','.join(['%s'] * len(open_statuses)))
                
                cursor.execute(query, open_statuses)
            
            orders = cursor.fetchall()
            cursor.close()
            
            return orders
            
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []
    
    def get_trading_statistics(self, platform: Optional[TradingPlatform] = None, 
                              days: int = 30) -> Dict[str, Any]:
        """Get trading statistics from database"""
        if not self.ensure_connection():
            return {}
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            # Base conditions
            conditions = ["t.executed_at >= DATE_SUB(NOW(), INTERVAL %s DAY)"]
            params = [days]
            
            if platform:
                platform_id = self.get_platform_id(platform)
                if not platform_id:
                    return {}
                conditions.append("t.platform_id = %s")
                params.append(platform_id)
            
            where_clause = " AND ".join(conditions)
            
            # Get basic statistics
            query = f"""
                SELECT 
                    tp.platform_name,
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN t.side = 'buy' THEN 1 ELSE 0 END) as buy_trades,
                    SUM(CASE WHEN t.side = 'sell' THEN 1 ELSE 0 END) as sell_trades,
                    SUM(t.quantity * t.price) as total_volume,
                    SUM(t.fee) as total_fees,
                    AVG(t.price) as avg_price,
                    MIN(t.executed_at) as first_trade,
                    MAX(t.executed_at) as last_trade
                FROM trades t
                JOIN trading_platforms tp ON t.platform_id = tp.platform_id
                WHERE {where_clause}
                GROUP BY tp.platform_id, tp.platform_name
            """
            
            cursor.execute(query, params)
            stats = cursor.fetchall()
            cursor.close()
            
            return {
                'statistics': stats,
                'period_days': days,
                'generated_at': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to get trading statistics: {e}")
            return {}
    
    def cleanup_old_records(self, days_to_keep: int = 90) -> bool:
        """Clean up old records from database"""
        if not self.ensure_connection():
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # Clean up old completed orders
            cleanup_orders_query = """
                DELETE FROM orders 
                WHERE status IN ('filled', 'canceled', 'expired', 'rejected') 
                AND created_at < DATE_SUB(NOW(), INTERVAL %s DAY)
            """
            cursor.execute(cleanup_orders_query, (days_to_keep,))
            orders_deleted = cursor.rowcount
            
            # Don't clean up trades - they're important for tax records
            
            self.connection.commit()
            cursor.close()
            
            logger.info(f"Cleaned up {orders_deleted} old order records")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup old records: {e}")
            return False
