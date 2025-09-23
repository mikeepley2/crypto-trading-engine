#!/usr/bin/env python3
"""
Multi-Platform Extensions for Trade Execution Engine
Adds Binance.US and KuCoin support to existing Coinbase engine
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import HTTPException

# Import platform modules
from platforms.platform_factory import PlatformFactory
from platforms.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class MultiPlatformExtension:
    """Extension class to add multi-platform support to existing trading engine"""
    
    def __init__(self):
        """Initialize multi-platform extension"""
        self.config_manager = ConfigManager()
        self.platform_factory = PlatformFactory()
        self.platforms = {}
        self.active_platforms = []
        
        try:
            # Load configuration
            config_path = "/app/trading_platforms.json"
            if os.path.exists(config_path):
                self.config = self.config_manager.load_config(config_path)
            else:
                logger.warning("Multi-platform config not found, using Coinbase-only mode")
                self.config = self._get_default_config()
            
            # Initialize enabled platforms
            self._initialize_platforms()
            
            logger.info(f"Multi-platform extension initialized with {len(self.active_platforms)} platforms")
            
        except Exception as e:
            logger.error(f"Failed to initialize multi-platform extension: {e}")
            # Fallback to Coinbase-only configuration
            self.config = self._get_default_config()
            self.active_platforms = ['coinbase']
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for Coinbase-only trading"""
        return {
            "enable_live_trading": True,
            "platforms": {
                "coinbase": {
                    "enabled": True,
                    "api_key": os.getenv('COINBASE_API_KEY', ''),
                    "private_key": os.getenv('COINBASE_PRIVATE_KEY', '')
                },
                "binance_us": {"enabled": False},
                "kucoin": {"enabled": False}
            }
        }
    
    def _initialize_platforms(self):
        """Initialize all enabled platforms"""
        for platform_name, platform_config in self.config.get('platforms', {}).items():
            if platform_config.get('enabled', False):
                try:
                    platform = self.platform_factory.create_platform(platform_name, platform_config)
                    if platform:
                        self.platforms[platform_name] = platform
                        self.active_platforms.append(platform_name)
                        logger.info(f"Initialized {platform_name} platform")
                    else:
                        logger.warning(f"Failed to create {platform_name} platform")
                except Exception as e:
                    logger.error(f"Error initializing {platform_name}: {e}")
    
    def get_platforms_status(self) -> Dict:
        """Get status of all platforms"""
        platforms_status = {}
        
        for platform_name, platform_config in self.config.get('platforms', {}).items():
            is_enabled = platform_config.get('enabled', False)
            is_initialized = platform_name in self.platforms
            
            status = {
                "enabled": is_enabled,
                "initialized": is_initialized,
                "status": "active" if is_enabled and is_initialized else "disabled"
            }
            
            # Add connection status for active platforms
            if is_initialized:
                try:
                    platform = self.platforms[platform_name]
                    # Test connection
                    if hasattr(platform, 'get_account_balance'):
                        balance = platform.get_account_balance('USD')
                        status["connected"] = True
                        status["usd_balance"] = balance
                    else:
                        status["connected"] = False
                except Exception as e:
                    status["connected"] = False
                    status["error"] = str(e)
            
            platforms_status[platform_name] = status
        
        return {
            "platforms": platforms_status,
            "active_platforms": self.active_platforms,
            "total_enabled": len(self.active_platforms),
            "live_trading_enabled": self.config.get('enable_live_trading', False),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_platform_portfolios(self) -> Dict:
        """Get portfolio information from all active platforms"""
        portfolios = {}
        
        for platform_name in self.active_platforms:
            if platform_name in self.platforms:
                try:
                    platform = self.platforms[platform_name]
                    portfolio = platform.get_portfolio()
                    portfolios[platform_name] = portfolio
                except Exception as e:
                    portfolios[platform_name] = {
                        "error": f"Failed to get portfolio: {e}",
                        "timestamp": datetime.now().isoformat()
                    }
        
        return portfolios
    
    def get_available_symbols(self) -> Dict:
        """Get available trading symbols from all platforms"""
        symbols_by_platform = {}
        
        for platform_name in self.active_platforms:
            if platform_name in self.platforms:
                try:
                    platform = self.platforms[platform_name]
                    if hasattr(platform, 'get_available_symbols'):
                        symbols = platform.get_available_symbols()
                        symbols_by_platform[platform_name] = symbols
                    else:
                        symbols_by_platform[platform_name] = []
                except Exception as e:
                    symbols_by_platform[platform_name] = {"error": str(e)}
        
        return symbols_by_platform
    
    def execute_multi_platform_trade(self, symbol: str, action: str, size_usd: float, platform: str = None) -> Dict:
        """Execute trade on specified platform or best available platform"""
        
        # If no platform specified, use the first active platform (usually Coinbase)
        if not platform:
            if self.active_platforms:
                platform = self.active_platforms[0]
            else:
                raise HTTPException(status_code=400, detail="No active platforms available")
        
        # Validate platform
        if platform not in self.active_platforms:
            raise HTTPException(status_code=400, detail=f"Platform {platform} not active")
        
        if platform not in self.platforms:
            raise HTTPException(status_code=500, detail=f"Platform {platform} not initialized")
        
        try:
            platform_instance = self.platforms[platform]
            
            # Execute trade using platform-specific implementation
            if hasattr(platform_instance, 'execute_trade'):
                result = platform_instance.execute_trade(symbol, action, size_usd)
                result['platform'] = platform
                return result
            else:
                raise HTTPException(status_code=500, detail=f"Platform {platform} does not support trade execution")
                
        except Exception as e:
            logger.error(f"Multi-platform trade execution failed on {platform}: {e}")
            raise HTTPException(status_code=500, detail=f"Trade execution failed: {e}")
    
    def get_platform_config(self) -> Dict:
        """Get current multi-platform configuration"""
        return self.config
    
    def update_platform_config(self, config: Dict) -> Dict:
        """Update multi-platform configuration"""
        try:
            # Validate configuration
            if 'platforms' not in config:
                raise ValueError("Configuration must include 'platforms' section")
            
            # Save configuration
            config_path = "/app/trading_platforms.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Reload configuration
            self.config = config
            self._initialize_platforms()
            
            return {
                "status": "success",
                "message": "Platform configuration updated",
                "active_platforms": self.active_platforms,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to update platform configuration: {e}")
            raise HTTPException(status_code=500, detail=f"Configuration update failed: {e}")

# Global multi-platform extension instance
multi_platform_extension = None

def initialize_multi_platform_extension():
    """Initialize the global multi-platform extension"""
    global multi_platform_extension
    try:
        multi_platform_extension = MultiPlatformExtension()
        logger.info("Multi-platform extension initialized successfully")
        return multi_platform_extension
    except Exception as e:
        logger.error(f"Failed to initialize multi-platform extension: {e}")
        return None

def get_multi_platform_extension():
    """Get the global multi-platform extension instance"""
    global multi_platform_extension
    if multi_platform_extension is None:
        multi_platform_extension = initialize_multi_platform_extension()
    return multi_platform_extension
