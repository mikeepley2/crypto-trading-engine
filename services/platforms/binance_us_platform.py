#!/usr/bin/env python3
"""
Binance.US Platform Implementation
Implements the TradingPlatformInterface for Binance.US
"""

import os
import json
import time
import hmac
import hashlib
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

class BinanceUSPlatform(TradingPlatformInterface):
    """Binance.US platform implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(TradingPlatform.BINANCE_US, config)
        
        # API Configuration
        self.api_key = config.get('api_key')
        self.secret_key = config.get('secret_key')
        self.base_url = config.get('base_url', 'https://api.binance.us')
        
        # Session for requests
        self.session = requests.Session()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.05  # 50ms between requests (20 req/sec)
        self.weight_counter = 0
        self.weight_reset_time = 0
        
        # Symbol mappings
        self.symbol_map = {}  # Will be populated during initialization
        
    def _generate_signature(self, query_string: str) -> str:
        """Generate HMAC SHA256 signature for Binance.US"""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _rate_limit(self, weight: int = 1):
        """Enforce rate limiting with request weight"""
        current_time = time.time()
        
        # Check request interval
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        # Reset weight counter if needed
        if current_time > self.weight_reset_time:
            self.weight_counter = 0
            self.weight_reset_time = current_time + 60  # Reset every minute
        
        # Check weight limit (1200 per minute)
        if self.weight_counter + weight > 1200:
            sleep_time = self.weight_reset_time - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)
                self.weight_counter = 0
                self.weight_reset_time = time.time() + 60
        
        self.weight_counter += weight
        self.last_request_time = time.time()
    
    async def _make_request(self, method: str, endpoint: str, params: Dict = None, 
                           signed: bool = False, weight: int = 1) -> Dict:
        """Make authenticated request to Binance.US API"""
        self._rate_limit(weight)
        
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        
        headers = {}
        
        if signed:
            if not self.api_key or not self.secret_key:
                raise PlatformAuthenticationError("API key and secret required for signed requests")
            
            # Add timestamp
            params['timestamp'] = int(time.time() * 1000)
            
            # Create query string
            query_string = urlencode(params)
            
            # Generate signature
            signature = self._generate_signature(query_string)
            params['signature'] = signature
            
            headers['X-MBX-APIKEY'] = self.api_key
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, headers=headers, params=params)
            elif method.upper() == 'POST':
                response = self.session.post(url, headers=headers, params=params)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url, headers=headers, params=params)
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
                
                # Try to parse error response
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get('msg', 'Unknown error')
                    error_code = error_data.get('code')
                    
                    if error_code == -2010:
                        raise InsufficientBalanceError("Insufficient balance")
                    elif error_code in [-1013, -1021, -1022]:
                        raise InvalidOrderError(error_msg)
                    
                except:
                    pass
            
            logger.error(f"Binance.US API request failed: {e}")
            raise PlatformConnectionError(f"Request failed: {e}")
    
    async def authenticate(self) -> bool:
        """Authenticate with Binance.US"""
        try:
            # Test authentication by getting account info
            await self._make_request('GET', '/api/v3/account', signed=True, weight=10)
            self._is_authenticated = True
            self._is_connected = True
            return True
        except Exception as e:
            logger.error(f"Binance.US authentication failed: {e}")
            self._is_authenticated = False
            self._is_connected = False
            return False
    
    async def get_health_status(self) -> PlatformHealthStatus:
        """Get platform health status"""
        try:
            start_time = time.time()
            await self._make_request('GET', '/api/v3/ping', weight=1)
            response_time = int((time.time() - start_time) * 1000)
            
            return PlatformHealthStatus(
                platform=self.platform,
                is_connected=True,
                is_trading_enabled=True,
                last_ping=datetime.now(),
                error_message=None,
                rate_limit_remaining=1200 - self.weight_counter
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
            response = await self._make_request('GET', '/api/v3/account', signed=True, weight=10)
            balances_data = response.get('balances', [])
            
            balances = []
            for balance_data in balances_data:
                asset = balance_data.get('asset')
                free = Decimal(balance_data.get('free', '0'))
                locked = Decimal(balance_data.get('locked', '0'))
                
                if free > 0 or locked > 0:  # Only include non-zero balances
                    balance = AssetBalance(
                        asset=asset,
                        free=free,
                        locked=locked,
                        total=free + locked
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
            response = await self._make_request('GET', '/api/v3/exchangeInfo', weight=10)
            symbols = response.get('symbols', [])
            
            pairs = []
            for symbol_data in symbols:
                if symbol_data.get('status') == 'TRADING':
                    # Extract filters for min/max order sizes
                    min_qty = Decimal('0')
                    max_qty = Decimal('1000000')
                    price_precision = 8
                    qty_precision = 8
                    
                    for filter_data in symbol_data.get('filters', []):
                        if filter_data.get('filterType') == 'LOT_SIZE':
                            min_qty = Decimal(filter_data.get('minQty', '0'))
                            max_qty = Decimal(filter_data.get('maxQty', '1000000'))
                        elif filter_data.get('filterType') == 'PRICE_FILTER':
                            # Count decimal places for precision
                            tick_size = filter_data.get('tickSize', '0.00000001')
                            price_precision = len(tick_size.split('.')[-1].rstrip('0'))
                    
                    qty_precision = symbol_data.get('baseAssetPrecision', 8)
                    
                    pair = TradingPair(
                        symbol=symbol_data.get('symbol'),
                        base_asset=symbol_data.get('baseAsset'),
                        quote_asset=symbol_data.get('quoteAsset'),
                        platform=self.platform,
                        min_order_size=min_qty,
                        max_order_size=max_qty,
                        price_precision=price_precision,
                        quantity_precision=qty_precision,
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
            response = await self._make_request('GET', '/api/v3/ticker/price', params=params, weight=1)
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
            # Prepare order parameters
            params = {
                'symbol': order_request.symbol,
                'side': order_request.side.value.upper(),
                'type': self._convert_order_type(order_request.order_type)
            }
            
            if order_request.client_order_id:
                params['newClientOrderId'] = order_request.client_order_id
            
            if order_request.order_type == OrderType.MARKET:
                if order_request.quote_quantity:
                    params['quoteOrderQty'] = str(order_request.quote_quantity)
                else:
                    params['quantity'] = str(order_request.quantity)
            elif order_request.order_type == OrderType.LIMIT:
                params['quantity'] = str(order_request.quantity)
                params['price'] = str(order_request.price)
                params['timeInForce'] = 'GTC'  # Good Till Canceled
            
            response = await self._make_request('POST', '/api/v3/order', params=params, signed=True, weight=1)
            
            return self._parse_order_response(response)
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str) -> OrderResponse:
        """Cancel an existing order"""
        try:
            params = {
                'symbol': symbol,
                'orderId': order_id
            }
            
            response = await self._make_request('DELETE', '/api/v3/order', params=params, signed=True, weight=1)
            
            return self._parse_order_response(response)
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise
    
    async def get_order(self, order_id: str, symbol: str) -> OrderResponse:
        """Get order details"""
        try:
            params = {
                'symbol': symbol,
                'orderId': order_id
            }
            
            response = await self._make_request('GET', '/api/v3/order', params=params, signed=True, weight=2)
            
            return self._parse_order_response(response)
            
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            raise
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResponse]:
        """Get all open orders"""
        try:
            params = {}
            if symbol:
                params['symbol'] = symbol
            
            response = await self._make_request('GET', '/api/v3/openOrders', params=params, signed=True, weight=3)
            
            return [self._parse_order_response(order) for order in response]
            
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            raise
    
    async def get_order_history(self, symbol: Optional[str] = None, 
                               limit: int = 500, 
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None) -> List[OrderResponse]:
        """Get order history"""
        try:
            params = {'limit': min(limit, 1000)}  # Binance max is 1000
            
            if symbol:
                params['symbol'] = symbol
            if start_time:
                params['startTime'] = int(start_time.timestamp() * 1000)
            if end_time:
                params['endTime'] = int(end_time.timestamp() * 1000)
            
            endpoint = '/api/v3/allOrders' if symbol else '/api/v3/myTrades'
            weight = 10 if symbol else 10
            
            response = await self._make_request('GET', endpoint, params=params, signed=True, weight=weight)
            
            return [self._parse_order_response(order) for order in response]
            
        except Exception as e:
            logger.error(f"Failed to get order history: {e}")
            raise
    
    async def get_trade_history(self, symbol: Optional[str] = None,
                               limit: int = 500,
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None) -> List[TradeExecution]:
        """Get trade execution history"""
        try:
            params = {'limit': min(limit, 1000)}
            
            if symbol:
                params['symbol'] = symbol
            if start_time:
                params['startTime'] = int(start_time.timestamp() * 1000)
            if end_time:
                params['endTime'] = int(end_time.timestamp() * 1000)
            
            response = await self._make_request('GET', '/api/v3/myTrades', params=params, signed=True, weight=10)
            
            trades = []
            for trade_data in response:
                trade = TradeExecution(
                    trade_id=str(trade_data.get('id')),
                    order_id=str(trade_data.get('orderId')),
                    symbol=trade_data.get('symbol'),
                    side=OrderSide.BUY if trade_data.get('isBuyer') else OrderSide.SELL,
                    quantity=Decimal(trade_data.get('qty', '0')),
                    price=Decimal(trade_data.get('price', '0')),
                    fee=Decimal(trade_data.get('commission', '0')),
                    fee_asset=trade_data.get('commissionAsset', 'BNB'),
                    executed_at=datetime.fromtimestamp(trade_data.get('time', 0) / 1000),
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
            response = await self._make_request('GET', '/api/v3/account', signed=True, weight=10)
            
            maker_commission = response.get('makerCommission', 10)
            taker_commission = response.get('takerCommission', 10)
            
            # Commission is in basis points (e.g., 10 = 0.1%)
            maker_fee = Decimal(maker_commission) / Decimal('10000')
            taker_fee = Decimal(taker_commission) / Decimal('10000')
            
            return TradingFee(
                maker_fee=maker_fee,
                taker_fee=taker_fee,
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
        """Normalize symbol to Binance.US format"""
        # Binance.US uses concatenated format: BTCUSD
        if '-' in symbol:
            # Convert BTC-USD to BTCUSD
            return symbol.replace('-', '')
        return symbol.upper()
    
    def standardize_symbol(self, platform_symbol: str) -> str:
        """Convert platform symbol to standardized format"""
        # Convert BTCUSD to BTC-USD for our standard format
        if platform_symbol.endswith('USD'):
            base = platform_symbol[:-3]
            return f"{base}-USD"
        elif platform_symbol.endswith('USDT'):
            base = platform_symbol[:-4]
            return f"{base}-USDT"
        elif platform_symbol.endswith('BTC'):
            base = platform_symbol[:-3]
            return f"{base}-BTC"
        elif platform_symbol.endswith('ETH'):
            base = platform_symbol[:-3]
            return f"{base}-ETH"
        
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
        """Convert standard order type to Binance format"""
        type_map = {
            OrderType.MARKET: 'MARKET',
            OrderType.LIMIT: 'LIMIT',
            OrderType.STOP_LOSS: 'STOP_LOSS',
            OrderType.STOP_LOSS_LIMIT: 'STOP_LOSS_LIMIT',
            OrderType.TAKE_PROFIT: 'TAKE_PROFIT',
            OrderType.TAKE_PROFIT_LIMIT: 'TAKE_PROFIT_LIMIT'
        }
        return type_map.get(order_type, 'MARKET')
    
    def _parse_order_status(self, status_str: str) -> OrderStatus:
        """Parse Binance order status to standard format"""
        status_map = {
            'NEW': OrderStatus.NEW,
            'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
            'FILLED': OrderStatus.FILLED,
            'CANCELED': OrderStatus.CANCELED,
            'PENDING_CANCEL': OrderStatus.PENDING_CANCEL,
            'REJECTED': OrderStatus.REJECTED,
            'EXPIRED': OrderStatus.EXPIRED
        }
        return status_map.get(status_str, OrderStatus.NEW)
    
    def _parse_order_type(self, order_type_str: str) -> OrderType:
        """Parse Binance order type to standard format"""
        type_map = {
            'MARKET': OrderType.MARKET,
            'LIMIT': OrderType.LIMIT,
            'STOP_LOSS': OrderType.STOP_LOSS,
            'STOP_LOSS_LIMIT': OrderType.STOP_LOSS_LIMIT,
            'TAKE_PROFIT': OrderType.TAKE_PROFIT,
            'TAKE_PROFIT_LIMIT': OrderType.TAKE_PROFIT_LIMIT
        }
        return type_map.get(order_type_str, OrderType.MARKET)
    
    def _parse_order_response(self, order_data: Dict) -> OrderResponse:
        """Parse Binance order data to OrderResponse"""
        return OrderResponse(
            platform_order_id=str(order_data.get('orderId')),
            client_order_id=order_data.get('clientOrderId'),
            symbol=order_data.get('symbol'),
            side=OrderSide.BUY if order_data.get('side') == 'BUY' else OrderSide.SELL,
            order_type=self._parse_order_type(order_data.get('type', 'MARKET')),
            quantity=Decimal(order_data.get('origQty', '0')),
            price=Decimal(order_data.get('price', '0')) if order_data.get('price') else None,
            status=self._parse_order_status(order_data.get('status', 'NEW')),
            filled_quantity=Decimal(order_data.get('executedQty', '0')),
            remaining_quantity=Decimal(order_data.get('origQty', '0')) - Decimal(order_data.get('executedQty', '0')),
            average_price=Decimal(order_data.get('cummulativeQuoteQty', '0')) / Decimal(order_data.get('executedQty', '1')) if Decimal(order_data.get('executedQty', '0')) > 0 else None,
            total_fee=Decimal('0'),  # Would need to calculate from fills
            fee_asset='BNB',
            created_at=datetime.fromtimestamp(order_data.get('time', 0) / 1000),
            updated_at=datetime.fromtimestamp(order_data.get('updateTime', 0) / 1000) if order_data.get('updateTime') else None,
            platform=self.platform
        )
