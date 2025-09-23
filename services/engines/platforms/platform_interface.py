#!/usr/bin/env python3
"""
Trading Platform Abstraction Layer
Provides a unified interface for multiple cryptocurrency trading platforms
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from decimal import Decimal
from datetime import datetime

class TradingPlatform(Enum):
    """Supported trading platforms"""
    COINBASE = "coinbase"
    BINANCE_US = "binance_us"
    KUCOIN = "kucoin"

class OrderType(Enum):
    """Order types supported across platforms"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LOSS_LIMIT = "stop_loss_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"

class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status across platforms"""
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class TradingFee:
    """Trading fee structure"""
    maker_fee: Decimal
    taker_fee: Decimal
    platform: TradingPlatform

@dataclass
class AssetBalance:
    """Asset balance information"""
    asset: str
    free: Decimal
    locked: Decimal
    total: Decimal
    usd_value: Optional[Decimal] = None

@dataclass
class TradingPair:
    """Trading pair information"""
    symbol: str
    base_asset: str
    quote_asset: str
    platform: TradingPlatform
    min_order_size: Decimal
    max_order_size: Decimal
    price_precision: int
    quantity_precision: int
    is_active: bool

@dataclass
class OrderRequest:
    """Order request structure"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Optional[Decimal] = None
    quote_quantity: Optional[Decimal] = None  # For market orders
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = "GTC"
    client_order_id: Optional[str] = None

@dataclass
class OrderResponse:
    """Order response structure"""
    platform_order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal]
    status: OrderStatus
    filled_quantity: Decimal
    remaining_quantity: Decimal
    average_price: Optional[Decimal]
    total_fee: Decimal
    fee_asset: str
    created_at: datetime
    updated_at: Optional[datetime]
    platform: TradingPlatform

@dataclass
class TradeExecution:
    """Individual trade execution"""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    fee: Decimal
    fee_asset: str
    executed_at: datetime
    platform: TradingPlatform

@dataclass
class PlatformHealthStatus:
    """Platform connectivity and health status"""
    platform: TradingPlatform
    is_connected: bool
    is_trading_enabled: bool
    last_ping: Optional[datetime]
    error_message: Optional[str]
    rate_limit_remaining: Optional[int]

class TradingPlatformInterface(ABC):
    """Abstract base class for trading platform implementations"""
    
    def __init__(self, platform: TradingPlatform, config: Dict[str, Any]):
        self.platform = platform
        self.config = config
        self._is_connected = False
        self._is_authenticated = False
    
    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with the platform"""
        pass
    
    @abstractmethod
    async def get_health_status(self) -> PlatformHealthStatus:
        """Get platform health and connectivity status"""
        pass
    
    @abstractmethod
    async def get_account_balances(self) -> List[AssetBalance]:
        """Get all account balances"""
        pass
    
    @abstractmethod
    async def get_asset_balance(self, asset: str) -> AssetBalance:
        """Get balance for a specific asset"""
        pass
    
    @abstractmethod
    async def get_trading_pairs(self) -> List[TradingPair]:
        """Get all available trading pairs"""
        pass
    
    @abstractmethod
    async def get_trading_pair(self, symbol: str) -> TradingPair:
        """Get specific trading pair information"""
        pass
    
    @abstractmethod
    async def get_current_price(self, symbol: str) -> Decimal:
        """Get current market price for a symbol"""
        pass
    
    @abstractmethod
    async def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """Place a new order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> OrderResponse:
        """Cancel an existing order"""
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str, symbol: str) -> OrderResponse:
        """Get order details"""
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResponse]:
        """Get all open orders"""
        pass
    
    @abstractmethod
    async def get_order_history(self, symbol: Optional[str] = None, 
                               limit: int = 100, 
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None) -> List[OrderResponse]:
        """Get order history"""
        pass
    
    @abstractmethod
    async def get_trade_history(self, symbol: Optional[str] = None,
                               limit: int = 100,
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None) -> List[TradeExecution]:
        """Get trade execution history"""
        pass
    
    @abstractmethod
    async def get_trading_fees(self, symbol: Optional[str] = None) -> TradingFee:
        """Get trading fees for the platform or specific symbol"""
        pass
    
    @abstractmethod
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to platform-specific format"""
        pass
    
    @abstractmethod
    def standardize_symbol(self, platform_symbol: str) -> str:
        """Convert platform symbol to standardized format"""
        pass
    
    # Platform-specific validation methods
    @abstractmethod
    def validate_order_request(self, order_request: OrderRequest) -> Tuple[bool, Optional[str]]:
        """Validate order request for platform-specific requirements"""
        pass
    
    @abstractmethod
    def calculate_order_size(self, symbol: str, side: OrderSide, 
                           usd_amount: Decimal, price: Optional[Decimal] = None) -> Decimal:
        """Calculate appropriate order size based on USD amount"""
        pass
    
    # Utility methods
    def is_connected(self) -> bool:
        """Check if platform is connected"""
        return self._is_connected
    
    def is_authenticated(self) -> bool:
        """Check if platform is authenticated"""
        return self._is_authenticated
    
    def get_platform_name(self) -> str:
        """Get platform name"""
        return self.platform.value

class RiskManager:
    """Risk management for multi-platform trading"""
    
    def __init__(self, config: Dict[str, Any]):
        self.max_position_size_usd = Decimal(config.get('max_position_size_usd', '1000'))
        self.max_daily_trades = int(config.get('max_daily_trades', '100'))
        self.max_daily_loss_usd = Decimal(config.get('max_daily_loss_usd', '500'))
        self.max_platform_allocation = Decimal(config.get('max_platform_allocation', '0.5'))  # 50% max per platform
        
    def validate_order(self, order_request: OrderRequest, 
                      current_balances: List[AssetBalance],
                      current_positions: Dict[str, Decimal]) -> Tuple[bool, Optional[str]]:
        """Validate order against risk parameters"""
        # Implementation for risk validation
        return True, None

class PlatformConnectionError(Exception):
    """Exception raised when platform connection fails"""
    pass

class PlatformAuthenticationError(Exception):
    """Exception raised when platform authentication fails"""
    pass

class InsufficientBalanceError(Exception):
    """Exception raised when insufficient balance for trade"""
    pass

class InvalidOrderError(Exception):
    """Exception raised when order parameters are invalid"""
    pass

class RateLimitExceededError(Exception):
    """Exception raised when rate limit is exceeded"""
    pass
