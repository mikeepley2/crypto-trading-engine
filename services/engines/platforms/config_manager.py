#!/usr/bin/env python3
"""
Multi-Platform Trading Configuration Management
Handles configuration loading, validation, and platform selection
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

from .platform_interface import TradingPlatform
from .platform_factory import PlatformManager, ConfigurationValidator

logger = logging.getLogger(__name__)

@dataclass
class PlatformConfig:
    """Configuration for a single trading platform"""
    enabled: bool = False
    api_key: str = ""
    secret_key: str = ""
    private_key: str = ""  # Coinbase only
    passphrase: str = ""   # KuCoin only
    base_url: str = ""
    sandbox: bool = False
    max_order_size: float = 1000.0
    default_order_type: str = "limit"
    risk_parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.risk_parameters is None:
            self.risk_parameters = {}

@dataclass
class TradingConfig:
    """Complete trading system configuration"""
    active_platform: str = "coinbase"
    enable_live_trading: bool = False
    max_portfolio_risk: float = 0.1
    default_position_size: float = 100.0
    platforms: Dict[str, PlatformConfig] = None
    
    def __post_init__(self):
        if self.platforms is None:
            self.platforms = {
                "coinbase": PlatformConfig(),
                "binance_us": PlatformConfig(),
                "kucoin": PlatformConfig()
            }

class ConfigManager:
    """Manages trading platform configurations"""
    
    DEFAULT_CONFIG_PATH = "config/trading_platforms.json"
    ENV_PREFIX = "TRADING_"
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config: TradingConfig = TradingConfig()
        self.platform_manager = PlatformManager()
        
    def load_config(self, config_path: Optional[str] = None) -> TradingConfig:
        """Load configuration from file"""
        path = config_path or self.config_path
        
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    config_data = json.load(f)
                self.config = self._dict_to_config(config_data)
                logger.info(f"Loaded configuration from {path}")
            else:
                logger.warning(f"Config file {path} not found, using defaults")
                self.config = TradingConfig()
                
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            self.config = TradingConfig()
        
        # Override with environment variables
        self._load_from_environment()
        
        return self.config
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """Save configuration to file"""
        path = config_path or self.config_path
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            config_data = self._config_to_dict(self.config)
            
            with open(path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            logger.info(f"Saved configuration to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save config to {path}: {e}")
            raise
    
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables"""
        
        # Active platform
        active_platform = os.getenv(f"{self.ENV_PREFIX}ACTIVE_PLATFORM")
        if active_platform:
            self.config.active_platform = active_platform
        
        # Live trading
        enable_live = os.getenv(f"{self.ENV_PREFIX}ENABLE_LIVE_TRADING")
        if enable_live:
            self.config.enable_live_trading = enable_live.lower() == 'true'
        
        # Portfolio risk
        max_risk = os.getenv(f"{self.ENV_PREFIX}MAX_PORTFOLIO_RISK")
        if max_risk:
            self.config.max_portfolio_risk = float(max_risk)
        
        # Platform-specific configurations
        for platform_name in ["coinbase", "binance_us", "kucoin"]:
            platform_config = self.config.platforms.get(platform_name)
            if not platform_config:
                continue
                
            prefix = f"{self.ENV_PREFIX}{platform_name.upper()}_"
            
            # API credentials
            api_key = os.getenv(f"{prefix}API_KEY")
            if api_key:
                platform_config.api_key = api_key
            
            secret_key = os.getenv(f"{prefix}SECRET_KEY")
            if secret_key:
                platform_config.secret_key = secret_key
            
            private_key = os.getenv(f"{prefix}PRIVATE_KEY")
            if private_key:
                platform_config.private_key = private_key
            
            passphrase = os.getenv(f"{prefix}PASSPHRASE")
            if passphrase:
                platform_config.passphrase = passphrase
            
            # Other settings
            enabled = os.getenv(f"{prefix}ENABLED")
            if enabled:
                platform_config.enabled = enabled.lower() == 'true'
            
            sandbox = os.getenv(f"{prefix}SANDBOX")
            if sandbox:
                platform_config.sandbox = sandbox.lower() == 'true'
            
            base_url = os.getenv(f"{prefix}BASE_URL")
            if base_url:
                platform_config.base_url = base_url
    
    def validate_config(self) -> Dict[str, List[str]]:
        """Validate all platform configurations"""
        validation_errors = {}
        
        for platform_name, platform_config in self.config.platforms.items():
            if not platform_config.enabled:
                continue
                
            try:
                platform_enum = TradingPlatform(platform_name)
                config_dict = asdict(platform_config)
                
                errors = ConfigurationValidator.validate_platform_config(platform_enum, config_dict)
                if errors:
                    validation_errors[platform_name] = errors
                    
            except ValueError:
                validation_errors[platform_name] = [f"Unknown platform: {platform_name}"]
        
        return validation_errors
    
    def initialize_platforms(self) -> bool:
        """Initialize all enabled platforms"""
        validation_errors = self.validate_config()
        
        if validation_errors:
            logger.error(f"Configuration validation failed: {validation_errors}")
            return False
        
        success = True
        
        for platform_name, platform_config in self.config.platforms.items():
            if not platform_config.enabled:
                continue
                
            try:
                platform_enum = TradingPlatform(platform_name)
                config_dict = asdict(platform_config)
                
                self.platform_manager.add_platform(platform_enum, config_dict)
                logger.info(f"Initialized {platform_name} platform")
                
            except Exception as e:
                logger.error(f"Failed to initialize {platform_name}: {e}")
                success = False
        
        # Set active platform
        try:
            active_platform = TradingPlatform(self.config.active_platform)
            if self.platform_manager.is_platform_configured(active_platform):
                self.platform_manager.set_active_platform(active_platform)
            else:
                logger.warning(f"Active platform {self.config.active_platform} not configured")
                # Set first available platform as active
                configured = self.platform_manager.get_configured_platforms()
                if configured:
                    self.platform_manager.set_active_platform(configured[0])
                    
        except ValueError:
            logger.error(f"Invalid active platform: {self.config.active_platform}")
            success = False
        
        return success
    
    def get_platform_manager(self) -> PlatformManager:
        """Get the platform manager instance"""
        return self.platform_manager
    
    def get_active_platform_config(self) -> Optional[PlatformConfig]:
        """Get configuration for the active platform"""
        return self.config.platforms.get(self.config.active_platform)
    
    def update_platform_config(self, platform_name: str, config_updates: Dict[str, Any]) -> None:
        """Update configuration for a specific platform"""
        if platform_name not in self.config.platforms:
            self.config.platforms[platform_name] = PlatformConfig()
        
        platform_config = self.config.platforms[platform_name]
        
        for key, value in config_updates.items():
            if hasattr(platform_config, key):
                setattr(platform_config, key, value)
            else:
                logger.warning(f"Unknown config key: {key}")
    
    def enable_platform(self, platform_name: str) -> None:
        """Enable a platform"""
        if platform_name in self.config.platforms:
            self.config.platforms[platform_name].enabled = True
            logger.info(f"Enabled platform: {platform_name}")
    
    def disable_platform(self, platform_name: str) -> None:
        """Disable a platform"""
        if platform_name in self.config.platforms:
            self.config.platforms[platform_name].enabled = False
            logger.info(f"Disabled platform: {platform_name}")
    
    def set_active_platform(self, platform_name: str) -> None:
        """Set the active platform"""
        if platform_name in self.config.platforms:
            self.config.active_platform = platform_name
            
            # Update platform manager if initialized
            try:
                platform_enum = TradingPlatform(platform_name)
                if self.platform_manager.is_platform_configured(platform_enum):
                    self.platform_manager.set_active_platform(platform_enum)
            except ValueError:
                logger.error(f"Invalid platform name: {platform_name}")
    
    def get_enabled_platforms(self) -> List[str]:
        """Get list of enabled platforms"""
        return [name for name, config in self.config.platforms.items() if config.enabled]
    
    def _dict_to_config(self, config_data: Dict[str, Any]) -> TradingConfig:
        """Convert dictionary to TradingConfig"""
        config = TradingConfig()
        
        # Top-level settings
        for key in ['active_platform', 'enable_live_trading', 'max_portfolio_risk', 'default_position_size']:
            if key in config_data:
                setattr(config, key, config_data[key])
        
        # Platform configurations
        if 'platforms' in config_data:
            config.platforms = {}
            for platform_name, platform_data in config_data['platforms'].items():
                platform_config = PlatformConfig()
                for key, value in platform_data.items():
                    if hasattr(platform_config, key):
                        setattr(platform_config, key, value)
                config.platforms[platform_name] = platform_config
        
        return config
    
    def _config_to_dict(self, config: TradingConfig) -> Dict[str, Any]:
        """Convert TradingConfig to dictionary"""
        config_dict = asdict(config)
        return config_dict

class EnvironmentConfigLoader:
    """Utility for loading configuration from environment variables"""
    
    @staticmethod
    def create_config_from_env() -> TradingConfig:
        """Create configuration entirely from environment variables"""
        config = TradingConfig()
        manager = ConfigManager()
        manager.config = config
        manager._load_from_environment()
        return manager.config
    
    @staticmethod
    def get_platform_config_from_env(platform_name: str) -> PlatformConfig:
        """Get platform configuration from environment variables"""
        config = PlatformConfig()
        prefix = f"TRADING_{platform_name.upper()}_"
        
        # Load all possible configuration values
        env_mappings = {
            'enabled': 'ENABLED',
            'api_key': 'API_KEY',
            'secret_key': 'SECRET_KEY',
            'private_key': 'PRIVATE_KEY',
            'passphrase': 'PASSPHRASE',
            'base_url': 'BASE_URL',
            'sandbox': 'SANDBOX',
            'max_order_size': 'MAX_ORDER_SIZE',
            'default_order_type': 'DEFAULT_ORDER_TYPE'
        }
        
        for config_key, env_suffix in env_mappings.items():
            env_var = f"{prefix}{env_suffix}"
            value = os.getenv(env_var)
            
            if value is not None:
                # Convert types
                if config_key in ['enabled', 'sandbox']:
                    value = value.lower() == 'true'
                elif config_key == 'max_order_size':
                    value = float(value)
                
                setattr(config, config_key, value)
        
        return config
