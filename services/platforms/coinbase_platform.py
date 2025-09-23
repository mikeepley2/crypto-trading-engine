#!/usr/bin/env python3
"""
Coinbase Advanced Trade API Platform Implementation
Implements the TradingPlatformInterface for Coinbase
"""

import os
import json
import time
import hmac
import hashlib
import requests
import secrets
import base64
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ec
import jwt

from .platform_interface import (
    TradingPlatformInterface, TradingPlatform, OrderType, OrderSide, OrderStatus,
    AssetBalance, TradingPair, OrderRequest, OrderResponse, TradeExecution,
    PlatformHealthStatus, TradingFee, PlatformConnectionError, 
    PlatformAuthenticationError, InsufficientBalanceError, 
    InvalidOrderError, RateLimitExceededError
)

logger = logging.getLogger(__name__)

class CoinbasePlatform(TradingPlatformInterface):
    """Coinbase Advanced Trade API implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(TradingPlatform.COINBASE, config)
        
        # API Configuration
        self.api_key = config.get('api_key')
        self.private_key = config.get('private_key')
        self.base_url = config.get('base_url', 'https://api.coinbase.com')
        
        # Parse private key
        self._load_private_key()
        
        # Session for requests
        self.session = requests.Session()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Symbol mappings
        self.symbol_map = {}  # Will be populated during initialization
        
    def _load_private_key(self):
        """Load and parse the private key"""
        try:
            if self.private_key.startswith('-----BEGIN'):
                # PEM format (legacy)
                self.private_key_obj = serialization.load_pem_private_key(
                    self.private_key.encode('utf-8'),
                    password=None
                )
                self.auth_method = 'EC'
            else:
                # Base64 format (new)
                self.private_key_obj = self.private_key
                self.auth_method = 'HMAC'
        except Exception as e:
            logger.error(f"Failed to load private key: {e}")
            raise PlatformAuthenticationError(f"Invalid private key: {e}")
    
    def _create_jwt_token(self, request_method: str, request_path: str, body: str = "") -> str:
        """Create JWT token for authentication"""
        timestamp = str(int(time.time()))
        
        if self.auth_method == 'HMAC':
            return self._create_hmac_jwt(request_method, request_path, body, timestamp)
        else:
            return self._create_ec_jwt(request_method, request_path, body, timestamp)
    
    def _create_hmac_jwt(self, request_method: str, request_path: str, body: str, timestamp: str) -> str:
        """Create HMAC JWT token"""
        hostname = "api.coinbase.com"
        uri = f"{request_method.upper()} {hostname}{request_path}"
        
        payload = {
            'iss': 'coinbase-cloud',
            'nbf': int(timestamp),
            'exp': int(timestamp) + 120,
            'sub': self.api_key,
            'uri': uri,
        }
        
        headers = {
            'kid': self.api_key,
            'nonce': secrets.token_hex(16),
        }
        
        secret_bytes = base64.b64decode(self.private_key_obj)
        return jwt.encode(payload, secret_bytes, algorithm='HS256', headers=headers)
    
    def _create_ec_jwt(self, request_method: str, request_path: str, body: str, timestamp: str) -> str:
        """Create EC JWT token"""
        hostname = "api.coinbase.com"
        uri = f"{request_method.upper()} {hostname}{request_path}"
        
        payload = {
            'iss': 'coinbase-cloud',
            'nbf': int(timestamp),
            'exp': int(timestamp) + 120,
            'sub': self.api_key,
            'uri': uri,
        }
        
        headers = {
            'kid': self.api_key,
            'nonce': timestamp,
        }
        
        return jwt.encode(payload, self.private_key_obj, algorithm='ES256', headers=headers)
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    async def _make_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict:
        """Make authenticated request to Coinbase API"""
        self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        body = json.dumps(data) if data else ""
        
        jwt_token = self._create_jwt_token(method, endpoint, body)
        
        headers = {
            'Authorization': f'Bearer {jwt_token}',
            'Content-Type': 'application/json',
        }
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, headers=headers, params=params)
            elif method.upper() == 'POST':
                response = self.session.post(url, headers=headers, json=data)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            if hasattr(e.response, 'status_code'):
                if e.response.status_code == 429:
                    raise RateLimitExceededError("Rate limit exceeded")
                elif e.response.status_code == 401:
                    raise PlatformAuthenticationError("Authentication failed")
                elif e.response.status_code >= 500:
                    raise PlatformConnectionError(f"Server error: {e}")
            
            logger.error(f"Coinbase API request failed: {e}")
            raise PlatformConnectionError(f"Request failed: {e}")
    
    async def authenticate(self) -> bool:
        """Authenticate with Coinbase"""
        try:
            # Test authentication by getting account info
            await self._make_request('GET', '/api/v3/brokerage/accounts')
            self._is_authenticated = True
            self._is_connected = True
            return True
        except Exception as e:
            logger.error(f"Coinbase authentication failed: {e}")
            self._is_authenticated = False
            self._is_connected = False
            return False
    
    async def get_health_status(self) -> PlatformHealthStatus:
        """Get platform health status"""
        try:
            start_time = time.time()
            await self._make_request('GET', '/api/v3/brokerage/accounts')
            response_time = int((time.time() - start_time) * 1000)
            
            return PlatformHealthStatus(
                platform=self.platform,
                is_connected=True,
                is_trading_enabled=True,
                last_ping=datetime.now(),
                error_message=None,
                rate_limit_remaining=None  # Coinbase doesn't provide this
            )
        except Exception as e:
            return PlatformHealthStatus(
                platform=self.platform,
                is_connected=False,
                is_trading_enabled=False,
                last_ping=datetime.now(),
                error_message=str(e),
                rate_limit_remaining=None
            )
    
    async def get_account_balances(self) -> List[AssetBalance]:
        """Get all account balances"""
        try:
            response = await self._make_request('GET', '/api/v3/brokerage/accounts')
            accounts = response.get('accounts', [])
            
            balances = []
            for account in accounts:
                currency = account.get('currency')
                available = Decimal(account.get('available_balance', {}).get('value', '0'))
                hold = Decimal(account.get('hold', {}).get('value', '0'))
                
                balance = AssetBalance(
                    asset=currency,
                    free=available,
                    locked=hold,
                    total=available + hold
                )
                balances.append(balance)
            
            return balances
            
        except Exception as e:
            logger.error(f"Failed to get account balances: {e}")
            raise
    
    async def get_asset_balance(self, asset: str) -> AssetBalance:
        """Get balance for specific asset"""
        balances = await self.get_account_balances()
        for balance in balances:
            if balance.asset == asset:
                return balance
        
        return AssetBalance(asset=asset, free=Decimal('0'), locked=Decimal('0'), total=Decimal('0'))
    
    async def get_trading_pairs(self) -> List[TradingPair]:
        """Get all trading pairs"""
        try:
            response = await self._make_request('GET', '/api/v3/brokerage/products')
            products = response.get('products', [])
            
            pairs = []
            for product in products:
                if product.get('status') == 'online':
                    pair = TradingPair(
                        symbol=product.get('product_id'),
                        base_asset=product.get('base_currency_id'),
                        quote_asset=product.get('quote_currency_id'),
                        platform=self.platform,
                        min_order_size=Decimal(product.get('base_min_size', '0')),
                        max_order_size=Decimal(product.get('base_max_size', '1000000')),
                        price_precision=int(product.get('quote_increment_scale', 2)),
                        quantity_precision=int(product.get('base_increment_scale', 8)),
                        is_active=True
                    )
                    pairs.append(pair)
            
            return pairs
            
        except Exception as e:
            logger.error(f"Failed to get trading pairs: {e}")
            raise
    
    async def get_trading_pair(self, symbol: str) -> TradingPair:
        """Get specific trading pair"""
        try:
            response = await self._make_request('GET', f'/api/v3/brokerage/products/{symbol}')
            
            return TradingPair(
                symbol=response.get('product_id'),
                base_asset=response.get('base_currency_id'),
                quote_asset=response.get('quote_currency_id'),
                platform=self.platform,
                min_order_size=Decimal(response.get('base_min_size', '0')),
                max_order_size=Decimal(response.get('base_max_size', '1000000')),
                price_precision=int(response.get('quote_increment_scale', 2)),
                quantity_precision=int(response.get('base_increment_scale', 8)),
                is_active=response.get('status') == 'online'
            )
            
        except Exception as e:
            logger.error(f"Failed to get trading pair {symbol}: {e}")
            raise
    
    async def get_current_price(self, symbol: str) -> Decimal:
        """Get current market price"""
        try:
            response = await self._make_request('GET', f'/api/v3/brokerage/products/{symbol}/ticker')
            return Decimal(response.get('price', '0'))
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            raise
    
    async def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """Place a new order"""
        # Validate order
        is_valid, error_msg = self.validate_order_request(order_request)
        if not is_valid:
            raise InvalidOrderError(error_msg)
        
        try:
            # Prepare order data
            order_data = {
                'product_id': order_request.symbol,
                'side': order_request.side.value.upper(),
                'order_configuration': {}
            }
            
            if order_request.order_type == OrderType.MARKET:
                if order_request.quote_quantity:
                    order_data['order_configuration']['market_market_ioc'] = {
                        'quote_size': str(order_request.quote_quantity)
                    }
                else:
                    order_data['order_configuration']['market_market_ioc'] = {
                        'base_size': str(order_request.quantity)
                    }
            elif order_request.order_type == OrderType.LIMIT:
                order_data['order_configuration']['limit_limit_gtc'] = {
                    'base_size': str(order_request.quantity),
                    'limit_price': str(order_request.price)
                }
            
            response = await self._make_request('POST', '/api/v3/brokerage/orders', data=order_data)
            
            # Parse response
            order_id = response.get('order_id')
            status_str = response.get('order_status', 'NEW')
            
            return OrderResponse(
                platform_order_id=order_id,
                client_order_id=order_request.client_order_id,
                symbol=order_request.symbol,
                side=order_request.side,
                order_type=order_request.order_type,
                quantity=order_request.quantity or Decimal('0'),
                price=order_request.price,
                status=self._parse_order_status(status_str),
                filled_quantity=Decimal('0'),
                remaining_quantity=order_request.quantity or Decimal('0'),
                average_price=None,
                total_fee=Decimal('0'),
                fee_asset='USD',
                created_at=datetime.now(),
                updated_at=None,
                platform=self.platform
            )
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str) -> OrderResponse:
        """Cancel an existing order"""
        try:
            response = await self._make_request('DELETE', f'/api/v3/brokerage/orders/{order_id}')
            
            # Get updated order details
            return await self.get_order(order_id, symbol)
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise
    
    async def get_order(self, order_id: str, symbol: str) -> OrderResponse:
        """Get order details"""
        try:
            response = await self._make_request('GET', f'/api/v3/brokerage/orders/historical/{order_id}')
            order = response.get('order', {})
            
            return self._parse_order_response(order)
            
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            raise
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResponse]:
        """Get all open orders"""
        try:
            params = {}
            if symbol:
                params['product_id'] = symbol
            
            response = await self._make_request('GET', '/api/v3/brokerage/orders/historical/batch', params=params)
            orders = response.get('orders', [])
            
            open_orders = []
            for order in orders:
                if order.get('status') in ['OPEN', 'PENDING']:
                    open_orders.append(self._parse_order_response(order))
            
            return open_orders
            
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            raise
    
    async def get_order_history(self, symbol: Optional[str] = None, 
                               limit: int = 100, 
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None) -> List[OrderResponse]:
        """Get order history"""
        try:
            params = {'limit': limit}
            if symbol:
                params['product_id'] = symbol
            if start_time:
                params['start_date'] = start_time.isoformat()
            if end_time:
                params['end_date'] = end_time.isoformat()
            
            response = await self._make_request('GET', '/api/v3/brokerage/orders/historical/batch', params=params)
            orders = response.get('orders', [])
            
            return [self._parse_order_response(order) for order in orders]
            
        except Exception as e:
            logger.error(f"Failed to get order history: {e}")
            raise
    
    async def get_trade_history(self, symbol: Optional[str] = None,
                               limit: int = 100,
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None) -> List[TradeExecution]:
        """Get trade execution history"""
        try:
            params = {'limit': limit}
            if symbol:
                params['product_id'] = symbol
            if start_time:
                params['start_date'] = start_time.isoformat()
            if end_time:
                params['end_date'] = end_time.isoformat()
            
            response = await self._make_request('GET', '/api/v3/brokerage/orders/historical/fills', params=params)
            fills = response.get('fills', [])
            
            trades = []
            for fill in fills:
                trade = TradeExecution(
                    trade_id=fill.get('trade_id'),
                    order_id=fill.get('order_id'),
                    symbol=fill.get('product_id'),
                    side=OrderSide.BUY if fill.get('side') == 'BUY' else OrderSide.SELL,
                    quantity=Decimal(fill.get('size', '0')),
                    price=Decimal(fill.get('price', '0')),
                    fee=Decimal(fill.get('commission', '0')),
                    fee_asset='USD',
                    executed_at=datetime.fromisoformat(fill.get('trade_time', '').replace('Z', '+00:00')),
                    platform=self.platform
                )
                trades.append(trade)
            
            return trades
            
        except Exception as e:
            logger.error(f"Failed to get trade history: {e}")
            raise
    
    async def get_trading_fees(self, symbol: Optional[str] = None) -> TradingFee:
        """Get trading fees"""
        # Coinbase Advanced Trade typically uses a flat fee structure
        # This would need to be updated based on actual fee schedule
        return TradingFee(
            maker_fee=Decimal('0.005'),  # 0.5%
            taker_fee=Decimal('0.005'),  # 0.5%
            platform=self.platform
        )
    
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to Coinbase format"""
        # Coinbase uses hyphenated format: BTC-USD
        if '-' not in symbol and len(symbol) > 3:
            # Convert BTCUSD to BTC-USD
            if symbol.endswith('USD'):
                base = symbol[:-3]
                return f"{base}-USD"
            elif symbol.endswith('USDT'):
                base = symbol[:-4]
                return f"{base}-USDT"
        return symbol
    
    def standardize_symbol(self, platform_symbol: str) -> str:
        """Convert platform symbol to standardized format"""
        # Our standard format is also BTC-USD
        return platform_symbol
    
    def validate_order_request(self, order_request: OrderRequest) -> Tuple[bool, Optional[str]]:
        """Validate order request"""
        if order_request.order_type == OrderType.MARKET:
            if not order_request.quantity and not order_request.quote_quantity:
                return False, "Market order requires either quantity or quote_quantity"
        elif order_request.order_type == OrderType.LIMIT:
            if not order_request.quantity or not order_request.price:
                return False, "Limit order requires both quantity and price"
        
        return True, None
    
    def calculate_order_size(self, symbol: str, side: OrderSide, 
                           usd_amount: Decimal, price: Optional[Decimal] = None) -> Decimal:
        """Calculate order size based on USD amount"""
        if not price:
            # Would need to fetch current price
            raise ValueError("Price is required for size calculation")
        
        if side == OrderSide.BUY:
            return usd_amount / price
        else:
            return usd_amount / price
    
    def _parse_order_status(self, status_str: str) -> OrderStatus:
        """Parse Coinbase order status to standard format"""
        status_map = {
            'PENDING': OrderStatus.NEW,
            'OPEN': OrderStatus.NEW,
            'FILLED': OrderStatus.FILLED,
            'CANCELLED': OrderStatus.CANCELED,
            'EXPIRED': OrderStatus.EXPIRED,
            'FAILED': OrderStatus.REJECTED,
            'UNKNOWN': OrderStatus.NEW
        }
        return status_map.get(status_str.upper(), OrderStatus.NEW)
    
    def _parse_order_response(self, order_data: Dict) -> OrderResponse:
        """Parse Coinbase order data to OrderResponse"""
        return OrderResponse(
            platform_order_id=order_data.get('order_id'),
            client_order_id=order_data.get('client_order_id'),
            symbol=order_data.get('product_id'),
            side=OrderSide.BUY if order_data.get('side') == 'BUY' else OrderSide.SELL,
            order_type=self._parse_order_type(order_data.get('order_type', 'UNKNOWN')),
            quantity=Decimal(order_data.get('size', '0')),
            price=Decimal(order_data.get('price', '0')) if order_data.get('price') else None,
            status=self._parse_order_status(order_data.get('status', 'UNKNOWN')),
            filled_quantity=Decimal(order_data.get('filled_size', '0')),
            remaining_quantity=Decimal(order_data.get('size', '0')) - Decimal(order_data.get('filled_size', '0')),
            average_price=Decimal(order_data.get('average_filled_price', '0')) if order_data.get('average_filled_price') else None,
            total_fee=Decimal(order_data.get('total_fees', '0')),
            fee_asset='USD',
            created_at=datetime.fromisoformat(order_data.get('created_time', '').replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(order_data.get('completion_time', '').replace('Z', '+00:00')) if order_data.get('completion_time') else None,
            platform=self.platform
        )
    
    def _parse_order_type(self, order_type_str: str) -> OrderType:
        """Parse Coinbase order type to standard format"""
        type_map = {
            'MARKET': OrderType.MARKET,
            'LIMIT': OrderType.LIMIT,
            'STOP': OrderType.STOP_LOSS,
            'STOP_LIMIT': OrderType.STOP_LOSS_LIMIT
        }
        return type_map.get(order_type_str.upper(), OrderType.MARKET)
