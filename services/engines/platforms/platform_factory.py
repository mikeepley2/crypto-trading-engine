#!/usr/bin/env python3
"""
Platform Factory for Multi-Platform Trading
Handles platform instantiation and configuration management
"""

import logging
from typing import Dict, Any, Optional, List
from .platform_interface import TradingPlatformInterface, TradingPlatform
from .coinbase_platform import CoinbasePlatform
from .binance_us_platform import BinanceUSPlatform
from .kucoin_platform import KuCoinPlatform

logger = logging.getLogger(__name__)

class PlatformFactory:
    """Factory for creating trading platform instances"""
    
    # Platform class mappings
    PLATFORM_CLASSES = {
        TradingPlatform.COINBASE: CoinbasePlatform,
        TradingPlatform.BINANCE_US: BinanceUSPlatform,
        TradingPlatform.KUCOIN: KuCoinPlatform
    }
    
    @classmethod
    def create_platform(cls, platform: TradingPlatform, config: Dict[str, Any]) -> TradingPlatformInterface:
        """
        Create a trading platform instance
        
        Args:
            platform: The platform enum
            config: Platform-specific configuration
            
        Returns:
            Configured platform instance
            
        Raises:
            ValueError: If platform is not supported
            KeyError: If required configuration is missing
        """
        if platform not in cls.PLATFORM_CLASSES:
            raise ValueError(f"Unsupported platform: {platform}")
        
        platform_class = cls.PLATFORM_CLASSES[platform]
        
        # Validate required configuration
        cls._validate_config(platform, config)
        
        try:
            instance = platform_class(config)
            logger.info(f"Created {platform.value} platform instance")
            return instance
        except Exception as e:
            logger.error(f"Failed to create {platform.value} platform: {e}")
            raise
    
    @classmethod
    def _validate_config(cls, platform: TradingPlatform, config: Dict[str, Any]) -> None:
        """Validate platform-specific configuration"""
        
        required_fields = {
            TradingPlatform.COINBASE: ['api_key', 'private_key'],
            TradingPlatform.BINANCE_US: ['api_key', 'secret_key'],
            TradingPlatform.KUCOIN: ['api_key', 'secret_key', 'passphrase']
        }
        
        platform_required = required_fields.get(platform, [])
        
        for field in platform_required:
            if field not in config or not config[field]:
                raise KeyError(f"Missing required configuration field '{field}' for {platform.value}")
    
    @classmethod
    def get_supported_platforms(cls) -> List[TradingPlatform]:
        """Get list of supported platforms"""
        return list(cls.PLATFORM_CLASSES.keys())
    
    @classmethod
    def is_platform_supported(cls, platform: TradingPlatform) -> bool:
        """Check if a platform is supported"""
        return platform in cls.PLATFORM_CLASSES

class PlatformManager:
    """Manages multiple platform instances and configuration"""
    
    def __init__(self):
        self.platforms: Dict[TradingPlatform, TradingPlatformInterface] = {}
        self.active_platform: Optional[TradingPlatform] = None
        
    def add_platform(self, platform: TradingPlatform, config: Dict[str, Any]) -> None:
        """Add a platform configuration"""
        try:
            platform_instance = PlatformFactory.create_platform(platform, config)
            self.platforms[platform] = platform_instance
            
            # Set as active if first platform added
            if self.active_platform is None:
                self.active_platform = platform
                
            logger.info(f"Added {platform.value} platform to manager")
            
        except Exception as e:
            logger.error(f"Failed to add {platform.value} platform: {e}")
            raise
    
    def remove_platform(self, platform: TradingPlatform) -> None:
        """Remove a platform"""
        if platform in self.platforms:
            del self.platforms[platform]
            
            # Update active platform if removed
            if self.active_platform == platform:
                self.active_platform = next(iter(self.platforms.keys())) if self.platforms else None
                
            logger.info(f"Removed {platform.value} platform from manager")
    
    def set_active_platform(self, platform: TradingPlatform) -> None:
        """Set the active platform"""
        if platform not in self.platforms:
            raise ValueError(f"Platform {platform.value} not configured")
        
        self.active_platform = platform
        logger.info(f"Set {platform.value} as active platform")
    
    def get_active_platform(self) -> Optional[TradingPlatformInterface]:
        """Get the active platform instance"""
        if self.active_platform and self.active_platform in self.platforms:
            return self.platforms[self.active_platform]
        return None
    
    def get_platform(self, platform: TradingPlatform) -> Optional[TradingPlatformInterface]:
        """Get a specific platform instance"""
        return self.platforms.get(platform)
    
    def get_all_platforms(self) -> Dict[TradingPlatform, TradingPlatformInterface]:
        """Get all configured platforms"""
        return self.platforms.copy()
    
    def get_configured_platforms(self) -> List[TradingPlatform]:
        """Get list of configured platforms"""
        return list(self.platforms.keys())
    
    async def authenticate_all(self) -> Dict[TradingPlatform, bool]:
        """Authenticate all configured platforms"""
        results = {}
        
        for platform, instance in self.platforms.items():
            try:
                success = await instance.authenticate()
                results[platform] = success
                logger.info(f"{platform.value} authentication: {'success' if success else 'failed'}")
            except Exception as e:
                results[platform] = False
                logger.error(f"{platform.value} authentication failed: {e}")
        
        return results
    
    async def get_all_health_status(self) -> Dict[TradingPlatform, Any]:
        """Get health status for all platforms"""
        results = {}
        
        for platform, instance in self.platforms.items():
            try:
                status = await instance.get_health_status()
                results[platform] = status
            except Exception as e:
                logger.error(f"Failed to get health status for {platform.value}: {e}")
                results[platform] = None
        
        return results
    
    def is_platform_configured(self, platform: TradingPlatform) -> bool:
        """Check if a platform is configured"""
        return platform in self.platforms
    
    def get_platform_count(self) -> int:
        """Get number of configured platforms"""
        return len(self.platforms)

class ConfigurationValidator:
    """Validates platform configurations"""
    
    @staticmethod
    def validate_coinbase_config(config: Dict[str, Any]) -> List[str]:
        """Validate Coinbase configuration"""
        errors = []
        
        if not config.get('api_key'):
            errors.append("Missing 'api_key'")
        
        if not config.get('private_key'):
            errors.append("Missing 'private_key'")
        elif not (config['private_key'].startswith('-----BEGIN') or 
                 len(config['private_key']) > 40):
            errors.append("Invalid 'private_key' format")
        
        base_url = config.get('base_url', 'https://api.coinbase.com')
        if not base_url.startswith('http'):
            errors.append("Invalid 'base_url' format")
        
        return errors
    
    @staticmethod
    def validate_binance_us_config(config: Dict[str, Any]) -> List[str]:
        """Validate Binance.US configuration"""
        errors = []
        
        if not config.get('api_key'):
            errors.append("Missing 'api_key'")
        
        if not config.get('secret_key'):
            errors.append("Missing 'secret_key'")
        
        base_url = config.get('base_url', 'https://api.binance.us')
        if not base_url.startswith('http'):
            errors.append("Invalid 'base_url' format")
        
        return errors
    
    @staticmethod
    def validate_kucoin_config(config: Dict[str, Any]) -> List[str]:
        """Validate KuCoin configuration"""
        errors = []
        
        if not config.get('api_key'):
            errors.append("Missing 'api_key'")
        
        if not config.get('secret_key'):
            errors.append("Missing 'secret_key'")
        
        if not config.get('passphrase'):
            errors.append("Missing 'passphrase'")
        
        base_url = config.get('base_url', 'https://api.kucoin.com')
        if not base_url.startswith('http'):
            errors.append("Invalid 'base_url' format")
        
        return errors
    
    @classmethod
    def validate_platform_config(cls, platform: TradingPlatform, config: Dict[str, Any]) -> List[str]:
        """Validate configuration for any platform"""
        validators = {
            TradingPlatform.COINBASE: cls.validate_coinbase_config,
            TradingPlatform.BINANCE_US: cls.validate_binance_us_config,
            TradingPlatform.KUCOIN: cls.validate_kucoin_config
        }
        
        validator = validators.get(platform)
        if validator:
            return validator(config)
        
        return [f"No validator available for platform: {platform.value}"]
