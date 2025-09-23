#!/usr/bin/env python3
"""
KuCoin Platform Implementation
Implements the TradingPlatformInterface for KuCoin
"""

import os
import json
import time
import hmac
import hashlib
import base64
import requests
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from urllib.parse import urlencode

from .platform_interface import (
    TradingPlatformInterface, TradingPlatform, OrderType, OrderSide, OrderStatus,
    AssetBalance, TradingPair, OrderRequest, OrderResponse, TradeExecution,
    PlatformHealthStatus, TradingFee, PlatformConnectionError, 
    PlatformAuthenticationError, InsufficientBalanceError, 
    InvalidOrderError, RateLimitExceededError
)

logger = logging.getLogger(__name__)

class KuCoinPlatform(TradingPlatformInterface):
    """KuCoin platform implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(TradingPlatform.KUCOIN, config)
        
        # API Configuration
        self.api_key = config.get('api_key')
        self.secret_key = config.get('secret_key')
        self.passphrase = config.get('passphrase')
        self.base_url = config.get('base_url', 'https://api.kucoin.com')
        self.sandbox = config.get('sandbox', False)
        
        if self.sandbox:
            self.base_url = 'https://openapi-sandbox.kucoin.com'
        
        # Session for requests
        self.session = requests.Session()
        
        # Rate limiting (KuCoin uses different limits for different endpoints)
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Symbol mappings
        self.symbol_map = {}  # Will be populated during initialization
        
    def _generate_signature(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Generate signature for KuCoin API"""
        # KC-API-SIGN = base64(hmac-sha256(KC-API-SECRET, str_to_sign))
        # str_to_sign = timestamp + method + requestPath + body
        str_to_sign = f"{timestamp}{method.upper()}{path}{body}"
        
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            str_to_sign.encode('utf-8'),
            hashlib.sha256
        ).digest()
        
        return base64.b64encode(signature).decode('utf-8')
    
    def _generate_passphrase_signature(self) -> str:
        """Generate passphrase signature for KuCoin API"""
        # KC-API-PASSPHRASE = base64(hmac-sha256(KC-API-SECRET, KC-API-PASSPHRASE))
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            self.passphrase.encode('utf-8'),
            hashlib.sha256
        ).digest()
        
        return base64.b64encode(signature).decode('utf-8')
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    async def _make_request(self, method: str, endpoint: str, params: Dict = None, 
                           data: Dict = None, signed: bool = False) -> Dict:
        """Make authenticated request to KuCoin API"""
        self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        
        headers = {
            'Content-Type': 'application/json',
            'KC-API-KEY-VERSION': '2'
        }
        
        # Prepare request body
        body = ""
        if data:
            body = json.dumps(data)
        
        if signed:
            if not self.api_key or not self.secret_key or not self.passphrase:
                raise PlatformAuthenticationError("API key, secret, and passphrase required for signed requests")
            
            # Generate timestamp
            timestamp = str(int(time.time() * 1000))
            
            # Prepare path for signature
            path = endpoint
            if params and method.upper() == 'GET':
                path += '?' + urlencode(params)
            
            # Generate signatures
            signature = self._generate_signature(timestamp, method, path, body)
            passphrase_signature = self._generate_passphrase_signature()
            
            headers.update({
                'KC-API-KEY': self.api_key,
                'KC-API-SIGN': signature,
                'KC-API-TIMESTAMP': timestamp,
                'KC-API-PASSPHRASE': passphrase_signature
            })
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, headers=headers, params=params)
            elif method.upper() == 'POST':
                response = self.session.post(url, headers=headers, data=body, params=params)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url, headers=headers, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            response_data = response.json()
            
            # KuCoin API returns {code, data, msg}
            if response_data.get('code') != '200000':
                error_msg = response_data.get('msg', 'Unknown error')
                error_code = response_data.get('code')
                
                if error_code == '400100':
                    raise PlatformAuthenticationError("Authentication failed")
                elif error_code in ['200004', '400760']:
                    raise InsufficientBalanceError("Insufficient balance")
                elif error_code in ['400001', '400003', '400100']:
                    raise InvalidOrderError(error_msg)
                elif 'rate limit' in error_msg.lower():
                    raise RateLimitExceededError("Rate limit exceeded")
                
                raise PlatformConnectionError(f"API error: {error_msg}")
            
            return response_data.get('data', {})
            
        except requests.exceptions.RequestException as e:
            if hasattr(e.response, 'status_code'):
                if e.response.status_code == 429:
                    raise RateLimitExceededError("Rate limit exceeded")
                elif e.response.status_code == 401:
                    raise PlatformAuthenticationError("Authentication failed")
                elif e.response.status_code >= 500:
                    raise PlatformConnectionError(f"Server error: {e}")
            
            logger.error(f"KuCoin API request failed: {e}")
            raise PlatformConnectionError(f"Request failed: {e}")
    
    async def authenticate(self) -> bool:
        """Authenticate with KuCoin"""
        try:
            # Test authentication by getting account info
            await self._make_request('GET', '/api/v1/accounts', signed=True)
            self._is_authenticated = True
            self._is_connected = True
            return True
        except Exception as e:
            logger.error(f"KuCoin authentication failed: {e}")
            self._is_authenticated = False
            self._is_connected = False
            return False
    
    async def get_health_status(self) -> PlatformHealthStatus:
        """Get platform health status"""
        try:
            start_time = time.time()
            await self._make_request('GET', '/api/v1/status')
            response_time = int((time.time() - start_time) * 1000)
            
            return PlatformHealthStatus(
                platform=self.platform,
                is_connected=True,
                is_trading_enabled=True,
                last_ping=datetime.now(),
                error_message=None,
                rate_limit_remaining=None  # KuCoin doesn't provide this in headers
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
            response = await self._make_request('GET', '/api/v1/accounts', signed=True)
            accounts = response if isinstance(response, list) else []
            
            # Group by currency
            balance_map = {}
            for account in accounts:
                currency = account.get('currency')
                account_type = account.get('type')  # main, trade, margin
                
                if currency not in balance_map:
                    balance_map[currency] = {'available': Decimal('0'), 'holds': Decimal('0')}
                
                balance_map[currency]['available'] += Decimal(account.get('available', '0'))
                balance_map[currency]['holds'] += Decimal(account.get('holds', '0'))
            
            balances = []
            for currency, balance_data in balance_map.items():
                available = balance_data['available']
                holds = balance_data['holds']
                
                if available > 0 or holds > 0:  # Only include non-zero balances
                    balance = AssetBalance(
                        asset=currency,
                        free=available,
                        locked=holds,
                        total=available + holds
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
            response = await self._make_request('GET', '/api/v1/symbols')
            symbols = response if isinstance(response, list) else []
            
            pairs = []
            for symbol_data in symbols:
                if symbol_data.get('enableTrading'):
                    pair = TradingPair(
                        symbol=symbol_data.get('symbol'),
                        base_asset=symbol_data.get('baseCurrency'),
                        quote_asset=symbol_data.get('quoteCurrency'),
                        platform=self.platform,
                        min_order_size=Decimal(symbol_data.get('baseMinSize', '0')),
                        max_order_size=Decimal(symbol_data.get('baseMaxSize', '1000000')),
                        price_precision=int(symbol_data.get('priceIncrement', 8)),
                        quantity_precision=int(symbol_data.get('baseIncrement', 8)),
                        is_active=True
                    )
                    pairs.append(pair)
            
            return pairs
            
        except Exception as e:
            logger.error(f"Failed to get trading pairs: {e}")
            raise
    
    async def get_trading_pair(self, symbol: str) -> TradingPair:
        """Get specific trading pair"""
        all_pairs = await self.get_trading_pairs()
        for pair in all_pairs:
            if pair.symbol == symbol:
                return pair
        
        raise ValueError(f"Trading pair {symbol} not found")
    
    async def get_current_price(self, symbol: str) -> Decimal:
        """Get current market price"""
        try:
            params = {'symbol': symbol}
            response = await self._make_request('GET', '/api/v1/market/orderbook/level1', params=params)
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
                'symbol': order_request.symbol,
                'side': order_request.side.value.lower(),  # KuCoin uses lowercase
                'type': self._convert_order_type(order_request.order_type)
            }
            
            if order_request.client_order_id:
                order_data['clientOid'] = order_request.client_order_id
            else:
                # Generate a client order ID
                order_data['clientOid'] = f"kc_{int(time.time() * 1000)}"
            
            if order_request.order_type == OrderType.MARKET:
                if order_request.quote_quantity:
                    order_data['funds'] = str(order_request.quote_quantity)
                else:
                    order_data['size'] = str(order_request.quantity)
            elif order_request.order_type == OrderType.LIMIT:
                order_data['size'] = str(order_request.quantity)
                order_data['price'] = str(order_request.price)
            
            response = await self._make_request('POST', '/api/v1/orders', data=order_data, signed=True)
            
            # KuCoin returns orderId in response
            order_id = response.get('orderId')
            
            return OrderResponse(
                platform_order_id=order_id,
                client_order_id=order_data['clientOid'],
                symbol=order_request.symbol,
                side=order_request.side,
                order_type=order_request.order_type,
                quantity=order_request.quantity or Decimal('0'),
                price=order_request.price,
                status=OrderStatus.NEW,
                filled_quantity=Decimal('0'),
                remaining_quantity=order_request.quantity or Decimal('0'),
                average_price=None,
                total_fee=Decimal('0'),
                fee_asset='KCS',
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
            response = await self._make_request('DELETE', f'/api/v1/orders/{order_id}', signed=True)
            
            # Get updated order details
            return await self.get_order(order_id, symbol)
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise
    
    async def get_order(self, order_id: str, symbol: str) -> OrderResponse:
        """Get order details"""
        try:
            response = await self._make_request('GET', f'/api/v1/orders/{order_id}', signed=True)
            
            return self._parse_order_response(response)
            
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            raise
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResponse]:
        """Get all open orders"""
        try:
            params = {'status': 'active'}
            if symbol:
                params['symbol'] = symbol
            
            response = await self._make_request('GET', '/api/v1/orders', params=params, signed=True)
            
            items = response.get('items', [])
            return [self._parse_order_response(order) for order in items]
            
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            raise
    
    async def get_order_history(self, symbol: Optional[str] = None, 
                               limit: int = 500, 
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None) -> List[OrderResponse]:
        """Get order history"""
        try:
            params = {'status': 'done', 'pageSize': min(limit, 500)}
            
            if symbol:
                params['symbol'] = symbol
            if start_time:
                params['startAt'] = int(start_time.timestamp() * 1000)
            if end_time:
                params['endAt'] = int(end_time.timestamp() * 1000)
            
            response = await self._make_request('GET', '/api/v1/orders', params=params, signed=True)
            
            items = response.get('items', [])
            return [self._parse_order_response(order) for order in items]
            
        except Exception as e:
            logger.error(f"Failed to get order history: {e}")
            raise
    
    async def get_trade_history(self, symbol: Optional[str] = None,
                               limit: int = 500,
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None) -> List[TradeExecution]:
        """Get trade execution history"""
        try:
            params = {'pageSize': min(limit, 500)}
            
            if symbol:
                params['symbol'] = symbol
            if start_time:
                params['startAt'] = int(start_time.timestamp() * 1000)
            if end_time:
                params['endAt'] = int(end_time.timestamp() * 1000)
            
            response = await self._make_request('GET', '/api/v1/fills', params=params, signed=True)
            
            items = response.get('items', [])
            trades = []
            
            for fill in items:
                trade = TradeExecution(
                    trade_id=fill.get('tradeId'),
                    order_id=fill.get('orderId'),
                    symbol=fill.get('symbol'),
                    side=OrderSide.BUY if fill.get('side') == 'buy' else OrderSide.SELL,
                    quantity=Decimal(fill.get('size', '0')),
                    price=Decimal(fill.get('price', '0')),
                    fee=Decimal(fill.get('fee', '0')),
                    fee_asset=fill.get('feeCurrency', 'KCS'),
                    executed_at=datetime.fromtimestamp(int(fill.get('createdAt', 0)) / 1000),
                    platform=self.platform
                )
                trades.append(trade)
            
            return trades
            
        except Exception as e:
            logger.error(f"Failed to get trade history: {e}")
            raise
    
    async def get_trading_fees(self, symbol: Optional[str] = None) -> TradingFee:
        """Get trading fees"""
        try:
            response = await self._make_request('GET', '/api/v1/base-fee', signed=True)
            
            # KuCoin returns maker and taker rates
            maker_fee_rate = Decimal(response.get('makerFeeRate', '0.001'))
            taker_fee_rate = Decimal(response.get('takerFeeRate', '0.001'))
            
            return TradingFee(
                maker_fee=maker_fee_rate,
                taker_fee=taker_fee_rate,
                platform=self.platform
            )
            
        except Exception as e:
            logger.error(f"Failed to get trading fees: {e}")
            # Return default fees if unable to fetch
            return TradingFee(
                maker_fee=Decimal('0.001'),  # 0.1%
                taker_fee=Decimal('0.001'),  # 0.1%
                platform=self.platform
            )
    
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to KuCoin format"""
        # KuCoin uses hyphenated format: BTC-USDT
        if '-' not in symbol and len(symbol) > 3:
            # Convert BTCUSDT to BTC-USDT
            if symbol.endswith('USDT'):
                base = symbol[:-4]
                return f"{base}-USDT"
            elif symbol.endswith('USD'):
                base = symbol[:-3]
                return f"{base}-USD"
        return symbol.upper()
    
    def standardize_symbol(self, platform_symbol: str) -> str:
        """Convert platform symbol to standardized format"""
        # KuCoin format is already our standard: BTC-USDT
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
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert standard order type to KuCoin format"""
        type_map = {
            OrderType.MARKET: 'market',
            OrderType.LIMIT: 'limit',
            OrderType.STOP_LOSS: 'stop',
            OrderType.STOP_LOSS_LIMIT: 'stop_limit'
        }
        return type_map.get(order_type, 'market')
    
    def _parse_order_status(self, status_str: str) -> OrderStatus:
        """Parse KuCoin order status to standard format"""
        status_map = {
            'active': OrderStatus.NEW,
            'done': OrderStatus.FILLED,
            'cancelled': OrderStatus.CANCELED
        }
        return status_map.get(status_str.lower(), OrderStatus.NEW)
    
    def _parse_order_type(self, order_type_str: str) -> OrderType:
        """Parse KuCoin order type to standard format"""
        type_map = {
            'market': OrderType.MARKET,
            'limit': OrderType.LIMIT,
            'stop': OrderType.STOP_LOSS,
            'stop_limit': OrderType.STOP_LOSS_LIMIT
        }
        return type_map.get(order_type_str.lower(), OrderType.MARKET)
    
    def _parse_order_response(self, order_data: Dict) -> OrderResponse:
        """Parse KuCoin order data to OrderResponse"""
        # Handle different response formats
        if 'dealSize' in order_data:
            filled_quantity = Decimal(order_data.get('dealSize', '0'))
        else:
            filled_quantity = Decimal(order_data.get('filledSize', '0'))
        
        quantity = Decimal(order_data.get('size', '0'))
        
        return OrderResponse(
            platform_order_id=order_data.get('id'),
            client_order_id=order_data.get('clientOid'),
            symbol=order_data.get('symbol'),
            side=OrderSide.BUY if order_data.get('side') == 'buy' else OrderSide.SELL,
            order_type=self._parse_order_type(order_data.get('type', 'market')),
            quantity=quantity,
            price=Decimal(order_data.get('price', '0')) if order_data.get('price') else None,
            status=self._parse_order_status(order_data.get('isActive', True) and 'active' or 'done'),
            filled_quantity=filled_quantity,
            remaining_quantity=quantity - filled_quantity,
            average_price=Decimal(order_data.get('dealFunds', '0')) / filled_quantity if filled_quantity > 0 else None,
            total_fee=Decimal(order_data.get('fee', '0')),
            fee_asset=order_data.get('feeCurrency', 'KCS'),
            created_at=datetime.fromtimestamp(int(order_data.get('createdAt', 0)) / 1000),
            updated_at=None,
            platform=self.platform
        )
