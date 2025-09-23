"""
Multi-Platform Trading Package
Provides unified interfaces for trading across multiple cryptocurrency platforms
"""

from .platform_interface import (
    TradingPlatformInterface,
    TradingPlatform,
    OrderType,
    OrderSide,
    OrderStatus,
    AssetBalance,
    TradingPair,
    OrderRequest,
    OrderResponse,
    TradeExecution,
    PlatformHealthStatus,
    TradingFee,
    PlatformConnectionError,
    PlatformAuthenticationError,
    InsufficientBalanceError,
    InvalidOrderError,
    RateLimitExceededError
)

from .platform_factory import PlatformFactory, PlatformManager, ConfigurationValidator
from .coinbase_platform import CoinbasePlatform
from .binance_us_platform import BinanceUSPlatform
from .kucoin_platform import KuCoinPlatform

__all__ = [
    # Core interfaces and enums
    'TradingPlatformInterface',
    'TradingPlatform',
    'OrderType',
    'OrderSide', 
    'OrderStatus',
    
    # Data classes
    'AssetBalance',
    'TradingPair',
    'OrderRequest',
    'OrderResponse',
    'TradeExecution',
    'PlatformHealthStatus',
    'TradingFee',
    
    # Exceptions
    'PlatformConnectionError',
    'PlatformAuthenticationError',
    'InsufficientBalanceError',
    'InvalidOrderError',
    'RateLimitExceededError',
    
    # Factory and management
    'PlatformFactory',
    'PlatformManager',
    'ConfigurationValidator',
    
    # Platform implementations
    'CoinbasePlatform',
    'BinanceUSPlatform',
    'KuCoinPlatform'
]

# Version information
__version__ = '1.0.0'
__author__ = 'AI Trading System'
__description__ = 'Multi-platform cryptocurrency trading abstraction layer'
