#!/usr/bin/env python3
"""
Multi-Platform Trading Engine
Unified trading engine that supports multiple cryptocurrency exchanges
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal

from .platforms import (
    TradingPlatformInterface, TradingPlatform, OrderType, OrderSide, OrderStatus,
    AssetBalance, TradingPair, OrderRequest, OrderResponse, TradeExecution,
    PlatformHealthStatus, TradingFee, PlatformManager, ConfigManager
)

logger = logging.getLogger(__name__)

class MultiPlatformTradingEngine:
    """
    Unified trading engine supporting multiple platforms
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_manager = ConfigManager(config_path)
        self.platform_manager: Optional[PlatformManager] = None
        self.is_initialized = False
        self.active_orders: Dict[str, OrderResponse] = {}
        self.trade_history: List[TradeExecution] = []
        
    async def initialize(self) -> bool:
        """Initialize the trading engine"""
        try:
            # Load configuration
            self.config_manager.load_config()
            
            # Initialize platforms
            success = self.config_manager.initialize_platforms()
            if not success:
                logger.error("Failed to initialize platforms")
                return False
            
            self.platform_manager = self.config_manager.get_platform_manager()
            
            # Authenticate all platforms
            auth_results = await self.platform_manager.authenticate_all()
            
            failed_auths = [platform.value for platform, success in auth_results.items() if not success]
            if failed_auths:
                logger.warning(f"Authentication failed for platforms: {failed_auths}")
            
            self.is_initialized = True
            logger.info("Multi-platform trading engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize trading engine: {e}")
            return False
    
    def get_active_platform(self) -> Optional[TradingPlatformInterface]:
        """Get the currently active platform"""
        if not self.platform_manager:
            return None
        return self.platform_manager.get_active_platform()
    
    def get_platform(self, platform: TradingPlatform) -> Optional[TradingPlatformInterface]:
        """Get a specific platform instance"""
        if not self.platform_manager:
            return None
        return self.platform_manager.get_platform(platform)
    
    def set_active_platform(self, platform: TradingPlatform) -> bool:
        """Set the active trading platform"""
        try:
            if not self.platform_manager:
                logger.error("Platform manager not initialized")
                return False
            
            if not self.platform_manager.is_platform_configured(platform):
                logger.error(f"Platform {platform.value} not configured")
                return False
            
            self.platform_manager.set_active_platform(platform)
            self.config_manager.set_active_platform(platform.value)
            
            logger.info(f"Set active platform to {platform.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set active platform: {e}")
            return False
    
    async def get_health_status(self, platform: Optional[TradingPlatform] = None) -> Dict[TradingPlatform, PlatformHealthStatus]:
        """Get health status for platforms"""
        if not self.platform_manager:
            return {}
        
        if platform:
            platform_instance = self.platform_manager.get_platform(platform)
            if platform_instance:
                status = await platform_instance.get_health_status()
                return {platform: status}
            return {}
        
        return await self.platform_manager.get_all_health_status()
    
    async def get_account_balances(self, platform: Optional[TradingPlatform] = None) -> Dict[TradingPlatform, List[AssetBalance]]:
        """Get account balances from platforms"""
        if not self.platform_manager:
            return {}
        
        results = {}
        
        if platform:
            platform_instance = self.platform_manager.get_platform(platform)
            if platform_instance:
                try:
                    balances = await platform_instance.get_account_balances()
                    results[platform] = balances
                except Exception as e:
                    logger.error(f"Failed to get balances for {platform.value}: {e}")
                    results[platform] = []
        else:
            # Get balances from all platforms
            for platform_enum, platform_instance in self.platform_manager.get_all_platforms().items():
                try:
                    balances = await platform_instance.get_account_balances()
                    results[platform_enum] = balances
                except Exception as e:
                    logger.error(f"Failed to get balances for {platform_enum.value}: {e}")
                    results[platform_enum] = []
        
        return results
    
    async def get_trading_pairs(self, platform: Optional[TradingPlatform] = None) -> Dict[TradingPlatform, List[TradingPair]]:
        """Get trading pairs from platforms"""
        if not self.platform_manager:
            return {}
        
        results = {}
        
        if platform:
            platform_instance = self.platform_manager.get_platform(platform)
            if platform_instance:
                try:
                    pairs = await platform_instance.get_trading_pairs()
                    results[platform] = pairs
                except Exception as e:
                    logger.error(f"Failed to get trading pairs for {platform.value}: {e}")
                    results[platform] = []
        else:
            # Get pairs from all platforms
            for platform_enum, platform_instance in self.platform_manager.get_all_platforms().items():
                try:
                    pairs = await platform_instance.get_trading_pairs()
                    results[platform_enum] = pairs
                except Exception as e:
                    logger.error(f"Failed to get trading pairs for {platform_enum.value}: {e}")
                    results[platform_enum] = []
        
        return results
    
    async def get_current_price(self, symbol: str, platform: Optional[TradingPlatform] = None) -> Dict[TradingPlatform, Decimal]:
        """Get current price across platforms"""
        if not self.platform_manager:
            return {}
        
        results = {}
        
        if platform:
            platform_instance = self.platform_manager.get_platform(platform)
            if platform_instance:
                try:
                    # Normalize symbol for the platform
                    normalized_symbol = platform_instance.normalize_symbol(symbol)
                    price = await platform_instance.get_current_price(normalized_symbol)
                    results[platform] = price
                except Exception as e:
                    logger.error(f"Failed to get price for {symbol} on {platform.value}: {e}")
        else:
            # Get price from all platforms
            for platform_enum, platform_instance in self.platform_manager.get_all_platforms().items():
                try:
                    normalized_symbol = platform_instance.normalize_symbol(symbol)
                    price = await platform_instance.get_current_price(normalized_symbol)
                    results[platform_enum] = price
                except Exception as e:
                    logger.error(f"Failed to get price for {symbol} on {platform_enum.value}: {e}")
        
        return results
    
    async def place_order(self, order_request: OrderRequest, platform: Optional[TradingPlatform] = None) -> Optional[OrderResponse]:
        """Place an order on specified platform or active platform"""
        if not self.platform_manager:
            logger.error("Platform manager not initialized")
            return None
        
        # Use specified platform or active platform
        if platform:
            platform_instance = self.platform_manager.get_platform(platform)
        else:
            platform_instance = self.platform_manager.get_active_platform()
            platform = self.platform_manager.active_platform
        
        if not platform_instance:
            logger.error("No platform available for order placement")
            return None
        
        try:
            # Normalize symbol for the platform
            normalized_symbol = platform_instance.normalize_symbol(order_request.symbol)
            order_request.symbol = normalized_symbol
            
            # Place the order
            order_response = await platform_instance.place_order(order_request)
            
            # Store active order
            if order_response:
                self.active_orders[order_response.platform_order_id] = order_response
            
            logger.info(f"Placed order {order_response.platform_order_id} on {platform.value}")
            return order_response
            
        except Exception as e:
            logger.error(f"Failed to place order on {platform.value}: {e}")
            return None
    
    async def cancel_order(self, order_id: str, symbol: str, platform: Optional[TradingPlatform] = None) -> Optional[OrderResponse]:
        """Cancel an order"""
        if not self.platform_manager:
            logger.error("Platform manager not initialized")
            return None
        
        # Use specified platform or find platform from stored orders
        if platform:
            platform_instance = self.platform_manager.get_platform(platform)
        else:
            # Try to find the order in our active orders
            stored_order = self.active_orders.get(order_id)
            if stored_order:
                platform = stored_order.platform
                platform_instance = self.platform_manager.get_platform(platform)
            else:
                platform_instance = self.platform_manager.get_active_platform()
                platform = self.platform_manager.active_platform
        
        if not platform_instance:
            logger.error("No platform available for order cancellation")
            return None
        
        try:
            # Normalize symbol for the platform
            normalized_symbol = platform_instance.normalize_symbol(symbol)
            
            # Cancel the order
            order_response = await platform_instance.cancel_order(order_id, normalized_symbol)
            
            # Update stored order
            if order_response:
                self.active_orders[order_id] = order_response
            
            logger.info(f"Cancelled order {order_id} on {platform.value}")
            return order_response
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id} on {platform.value}: {e}")
            return None
    
    async def get_order_status(self, order_id: str, symbol: str, platform: Optional[TradingPlatform] = None) -> Optional[OrderResponse]:
        """Get order status"""
        if not self.platform_manager:
            logger.error("Platform manager not initialized")
            return None
        
        # Use specified platform or find platform from stored orders
        if platform:
            platform_instance = self.platform_manager.get_platform(platform)
        else:
            # Try to find the order in our active orders
            stored_order = self.active_orders.get(order_id)
            if stored_order:
                platform = stored_order.platform
                platform_instance = self.platform_manager.get_platform(platform)
            else:
                platform_instance = self.platform_manager.get_active_platform()
                platform = self.platform_manager.active_platform
        
        if not platform_instance:
            logger.error("No platform available for order status check")
            return None
        
        try:
            # Normalize symbol for the platform
            normalized_symbol = platform_instance.normalize_symbol(symbol)
            
            # Get order status
            order_response = await platform_instance.get_order(order_id, normalized_symbol)
            
            # Update stored order
            if order_response:
                self.active_orders[order_id] = order_response
            
            return order_response
            
        except Exception as e:
            logger.error(f"Failed to get order status for {order_id} on {platform.value}: {e}")
            return None
    
    async def get_open_orders(self, symbol: Optional[str] = None, platform: Optional[TradingPlatform] = None) -> Dict[TradingPlatform, List[OrderResponse]]:
        """Get open orders from platforms"""
        if not self.platform_manager:
            return {}
        
        results = {}
        
        if platform:
            platform_instance = self.platform_manager.get_platform(platform)
            if platform_instance:
                try:
                    normalized_symbol = platform_instance.normalize_symbol(symbol) if symbol else None
                    orders = await platform_instance.get_open_orders(normalized_symbol)
                    results[platform] = orders
                    
                    # Update stored orders
                    for order in orders:
                        self.active_orders[order.platform_order_id] = order
                        
                except Exception as e:
                    logger.error(f"Failed to get open orders for {platform.value}: {e}")
                    results[platform] = []
        else:
            # Get orders from all platforms
            for platform_enum, platform_instance in self.platform_manager.get_all_platforms().items():
                try:
                    normalized_symbol = platform_instance.normalize_symbol(symbol) if symbol else None
                    orders = await platform_instance.get_open_orders(normalized_symbol)
                    results[platform_enum] = orders
                    
                    # Update stored orders
                    for order in orders:
                        self.active_orders[order.platform_order_id] = order
                        
                except Exception as e:
                    logger.error(f"Failed to get open orders for {platform_enum.value}: {e}")
                    results[platform_enum] = []
        
        return results
    
    async def get_trade_history(self, symbol: Optional[str] = None, 
                               limit: int = 100,
                               platform: Optional[TradingPlatform] = None) -> Dict[TradingPlatform, List[TradeExecution]]:
        """Get trade history from platforms"""
        if not self.platform_manager:
            return {}
        
        results = {}
        
        if platform:
            platform_instance = self.platform_manager.get_platform(platform)
            if platform_instance:
                try:
                    normalized_symbol = platform_instance.normalize_symbol(symbol) if symbol else None
                    trades = await platform_instance.get_trade_history(normalized_symbol, limit=limit)
                    results[platform] = trades
                except Exception as e:
                    logger.error(f"Failed to get trade history for {platform.value}: {e}")
                    results[platform] = []
        else:
            # Get trades from all platforms
            for platform_enum, platform_instance in self.platform_manager.get_all_platforms().items():
                try:
                    normalized_symbol = platform_instance.normalize_symbol(symbol) if symbol else None
                    trades = await platform_instance.get_trade_history(normalized_symbol, limit=limit)
                    results[platform_enum] = trades
                except Exception as e:
                    logger.error(f"Failed to get trade history for {platform_enum.value}: {e}")
                    results[platform_enum] = []
        
        return results
    
    async def get_trading_fees(self, symbol: Optional[str] = None, platform: Optional[TradingPlatform] = None) -> Dict[TradingPlatform, TradingFee]:
        """Get trading fees from platforms"""
        if not self.platform_manager:
            return {}
        
        results = {}
        
        if platform:
            platform_instance = self.platform_manager.get_platform(platform)
            if platform_instance:
                try:
                    fees = await platform_instance.get_trading_fees(symbol)
                    results[platform] = fees
                except Exception as e:
                    logger.error(f"Failed to get trading fees for {platform.value}: {e}")
        else:
            # Get fees from all platforms
            for platform_enum, platform_instance in self.platform_manager.get_all_platforms().items():
                try:
                    fees = await platform_instance.get_trading_fees(symbol)
                    results[platform_enum] = fees
                except Exception as e:
                    logger.error(f"Failed to get trading fees for {platform_enum.value}: {e}")
        
        return results
    
    def get_configured_platforms(self) -> List[TradingPlatform]:
        """Get list of configured platforms"""
        if not self.platform_manager:
            return []
        return self.platform_manager.get_configured_platforms()
    
    def get_platform_status_summary(self) -> Dict[str, Any]:
        """Get summary of platform status"""
        if not self.platform_manager:
            return {"error": "Platform manager not initialized"}
        
        configured = self.platform_manager.get_configured_platforms()
        active = self.platform_manager.active_platform
        
        return {
            "initialized": self.is_initialized,
            "active_platform": active.value if active else None,
            "configured_platforms": [p.value for p in configured],
            "platform_count": len(configured)
        }
