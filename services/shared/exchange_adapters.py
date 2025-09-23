#!/usr/bin/env python3
"""
Exchange API Adapters - Abstract interface and concrete implementations
Provides a unified interface for multiple cryptocurrency exchanges
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
import asyncio
import aiohttp
import hashlib
import hmac
import base64
import time
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class OrderResult:
    """Standardized order result across all exchanges"""
    success: bool
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    status: str = "unknown"  # pending, filled, partially_filled, cancelled, rejected
    filled_quantity: Decimal = Decimal('0')
    remaining_quantity: Decimal = Decimal('0')
    avg_fill_price: Optional[Decimal] = None
    total_fees: Decimal = Decimal('0')
    error_message: Optional[str] = None
    raw_response: Optional[Dict] = None

@dataclass
class Balance:
    """Account balance for a specific asset"""
    asset: str
    available: Decimal
    locked: Decimal
    total: Decimal

@dataclass
class Ticker:
    """Price ticker information"""
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume_24h: Decimal
    timestamp: datetime

class ExchangeAdapter(ABC):
    """Abstract base class for all exchange adapters"""
    
    def __init__(self, api_key: str, api_secret: str, passphrase: str = None, sandbox: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.sandbox = sandbox
        self.session: Optional[aiohttp.ClientSession] = None
        
    @property
    @abstractmethod
    def exchange_name(self) -> str:
        """Return the exchange name"""
        pass
    
    @property
    @abstractmethod
    def base_url(self) -> str:
        """Return the base URL for the exchange API"""
        pass
    
    @abstractmethod
    async def authenticate_request(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Generate authentication headers for API requests"""
        pass
    
    @abstractmethod
    async def place_market_order(self, symbol: str, side: str, quantity: Decimal) -> OrderResult:
        """Place a market order"""
        pass
    
    @abstractmethod
    async def place_limit_order(self, symbol: str, side: str, quantity: Decimal, price: Decimal) -> OrderResult:
        """Place a limit order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """Cancel an existing order"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str, symbol: str = None) -> OrderResult:
        """Get the status of an order"""
        pass
    
    @abstractmethod
    async def get_balances(self) -> List[Balance]:
        """Get account balances"""
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get ticker information for a symbol"""
        pass
    
    @abstractmethod
    async def get_trading_fees(self, symbol: str = None) -> Dict[str, Decimal]:
        """Get trading fees (maker/taker)"""
        pass
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def test_connection(self) -> Tuple[bool, str]:
        """Test the API connection and credentials"""
        try:
            balances = await self.get_balances()
            return True, f"Connected successfully. Found {len(balances)} assets."
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

class CoinbaseProAdapter(ExchangeAdapter):
    """Coinbase Pro (now Coinbase Advanced Trade) adapter"""
    
    @property
    def exchange_name(self) -> str:
        return "coinbase_pro"
    
    @property
    def base_url(self) -> str:
        if self.sandbox:
            return "https://api-public.sandbox.pro.coinbase.com"
        return "https://api.pro.coinbase.com"
    
    async def authenticate_request(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Generate Coinbase Pro authentication headers"""
        timestamp = str(int(time.time()))
        message = timestamp + method.upper() + path + body
        
        signature = base64.b64encode(
            hmac.new(
                base64.b64decode(self.api_secret),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        
        return {
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-SIGN': signature,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
    
    async def place_market_order(self, symbol: str, side: str, quantity: Decimal) -> OrderResult:
        """Place a market order on Coinbase Pro"""
        try:
            path = "/orders"
            body = json.dumps({
                "type": "market",
                "side": side.lower(),
                "product_id": symbol,
                "size": str(quantity)
            })
            
            headers = await self.authenticate_request("POST", path, body)
            
            async with self.session.post(
                f"{self.base_url}{path}",
                headers=headers,
                data=body
            ) as response:
                result = await response.json()
                
                if response.status == 200:
                    return OrderResult(
                        success=True,
                        order_id=result.get('id'),
                        status='pending',
                        raw_response=result
                    )
                else:
                    return OrderResult(
                        success=False,
                        error_message=result.get('message', 'Unknown error'),
                        raw_response=result
                    )
                    
        except Exception as e:
            logger.error(f"Coinbase Pro market order error: {e}")
            return OrderResult(success=False, error_message=str(e))
    
    async def place_limit_order(self, symbol: str, side: str, quantity: Decimal, price: Decimal) -> OrderResult:
        """Place a limit order on Coinbase Pro"""
        try:
            path = "/orders"
            body = json.dumps({
                "type": "limit",
                "side": side.lower(),
                "product_id": symbol,
                "size": str(quantity),
                "price": str(price)
            })
            
            headers = await self.authenticate_request("POST", path, body)
            
            async with self.session.post(
                f"{self.base_url}{path}",
                headers=headers,
                data=body
            ) as response:
                result = await response.json()
                
                if response.status == 200:
                    return OrderResult(
                        success=True,
                        order_id=result.get('id'),
                        status='pending',
                        raw_response=result
                    )
                else:
                    return OrderResult(
                        success=False,
                        error_message=result.get('message', 'Unknown error'),
                        raw_response=result
                    )
                    
        except Exception as e:
            logger.error(f"Coinbase Pro limit order error: {e}")
            return OrderResult(success=False, error_message=str(e))
    
    async def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """Cancel an order on Coinbase Pro"""
        try:
            path = f"/orders/{order_id}"
            headers = await self.authenticate_request("DELETE", path)
            
            async with self.session.delete(
                f"{self.base_url}{path}",
                headers=headers
            ) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Coinbase Pro cancel order error: {e}")
            return False
    
    async def get_order_status(self, order_id: str, symbol: str = None) -> OrderResult:
        """Get order status from Coinbase Pro"""
        try:
            path = f"/orders/{order_id}"
            headers = await self.authenticate_request("GET", path)
            
            async with self.session.get(
                f"{self.base_url}{path}",
                headers=headers
            ) as response:
                result = await response.json()
                
                if response.status == 200:
                    return OrderResult(
                        success=True,
                        order_id=result.get('id'),
                        status=result.get('status', 'unknown'),
                        filled_quantity=Decimal(result.get('filled_size', '0')),
                        remaining_quantity=Decimal(result.get('size', '0')) - Decimal(result.get('filled_size', '0')),
                        avg_fill_price=Decimal(result.get('executed_value', '0')) / Decimal(result.get('filled_size', '1')) if result.get('filled_size', '0') != '0' else None,
                        total_fees=Decimal(result.get('fill_fees', '0')),
                        raw_response=result
                    )
                else:
                    return OrderResult(
                        success=False,
                        error_message=result.get('message', 'Unknown error'),
                        raw_response=result
                    )
                    
        except Exception as e:
            logger.error(f"Coinbase Pro get order status error: {e}")
            return OrderResult(success=False, error_message=str(e))
    
    async def get_balances(self) -> List[Balance]:
        """Get account balances from Coinbase Pro"""
        try:
            path = "/accounts"
            headers = await self.authenticate_request("GET", path)
            
            async with self.session.get(
                f"{self.base_url}{path}",
                headers=headers
            ) as response:
                accounts = await response.json()
                
                if response.status == 200:
                    balances = []
                    for account in accounts:
                        balance = Balance(
                            asset=account['currency'],
                            available=Decimal(account['available']),
                            locked=Decimal(account['hold']),
                            total=Decimal(account['balance'])
                        )
                        balances.append(balance)
                    return balances
                else:
                    logger.error(f"Failed to get balances: {accounts}")
                    return []
                    
        except Exception as e:
            logger.error(f"Coinbase Pro get balances error: {e}")
            return []
    
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get ticker information from Coinbase Pro"""
        try:
            path = f"/products/{symbol}/ticker"
            
            async with self.session.get(f"{self.base_url}{path}") as response:
                result = await response.json()
                
                if response.status == 200:
                    return Ticker(
                        symbol=symbol,
                        bid=Decimal(result['bid']),
                        ask=Decimal(result['ask']),
                        last=Decimal(result['price']),
                        volume_24h=Decimal(result['volume']),
                        timestamp=datetime.now()
                    )
                else:
                    raise Exception(f"Failed to get ticker: {result}")
                    
        except Exception as e:
            logger.error(f"Coinbase Pro get ticker error: {e}")
            raise
    
    async def get_trading_fees(self, symbol: str = None) -> Dict[str, Decimal]:
        """Get trading fees from Coinbase Pro"""
        try:
            path = "/fees"
            headers = await self.authenticate_request("GET", path)
            
            async with self.session.get(
                f"{self.base_url}{path}",
                headers=headers
            ) as response:
                result = await response.json()
                
                if response.status == 200:
                    return {
                        'maker': Decimal(result['maker_fee_rate']),
                        'taker': Decimal(result['taker_fee_rate'])
                    }
                else:
                    return {'maker': Decimal('0.005'), 'taker': Decimal('0.005')}  # Default fees
                    
        except Exception as e:
            logger.error(f"Coinbase Pro get fees error: {e}")
            return {'maker': Decimal('0.005'), 'taker': Decimal('0.005')}

class BinanceAdapter(ExchangeAdapter):
    """Binance exchange adapter"""
    
    @property
    def exchange_name(self) -> str:
        return "binance"
    
    @property
    def base_url(self) -> str:
        if self.sandbox:
            return "https://testnet.binance.vision"
        return "https://api.binance.com"
    
    async def authenticate_request(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Generate Binance authentication headers"""
        timestamp = int(time.time() * 1000)
        query_string = f"timestamp={timestamp}"
        if body:
            query_string = body + f"&timestamp={timestamp}"
        
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return {
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/x-www-form-urlencoded'
        }
    
    # Implement other methods similar to CoinbaseProAdapter...
    async def place_market_order(self, symbol: str, side: str, quantity: Decimal) -> OrderResult:
        # Implementation for Binance market orders
        return OrderResult(success=False, error_message="Binance adapter not fully implemented yet")
    
    async def place_limit_order(self, symbol: str, side: str, quantity: Decimal, price: Decimal) -> OrderResult:
        return OrderResult(success=False, error_message="Binance adapter not fully implemented yet")
    
    async def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        return False
    
    async def get_order_status(self, order_id: str, symbol: str = None) -> OrderResult:
        return OrderResult(success=False, error_message="Binance adapter not fully implemented yet")
    
    async def get_balances(self) -> List[Balance]:
        return []
    
    async def get_ticker(self, symbol: str) -> Ticker:
        raise NotImplementedError("Binance adapter not fully implemented yet")
    
    async def get_trading_fees(self, symbol: str = None) -> Dict[str, Decimal]:
        return {'maker': Decimal('0.001'), 'taker': Decimal('0.001')}

class KrakenAdapter(ExchangeAdapter):
    """Kraken exchange adapter"""
    
    @property
    def exchange_name(self) -> str:
        return "kraken"
    
    @property
    def base_url(self) -> str:
        return "https://api.kraken.com"
    
    async def authenticate_request(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Generate Kraken authentication headers"""
        # Kraken uses a different authentication method
        return {}
    
    # Placeholder implementations
    async def place_market_order(self, symbol: str, side: str, quantity: Decimal) -> OrderResult:
        return OrderResult(success=False, error_message="Kraken adapter not fully implemented yet")
    
    async def place_limit_order(self, symbol: str, side: str, quantity: Decimal, price: Decimal) -> OrderResult:
        return OrderResult(success=False, error_message="Kraken adapter not fully implemented yet")
    
    async def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        return False
    
    async def get_order_status(self, order_id: str, symbol: str = None) -> OrderResult:
        return OrderResult(success=False, error_message="Kraken adapter not fully implemented yet")
    
    async def get_balances(self) -> List[Balance]:
        return []
    
    async def get_ticker(self, symbol: str) -> Ticker:
        raise NotImplementedError("Kraken adapter not fully implemented yet")
    
    async def get_trading_fees(self, symbol: str = None) -> Dict[str, Decimal]:
        return {'maker': Decimal('0.0016'), 'taker': Decimal('0.0026')}

class ExchangeFactory:
    """Factory class for creating exchange adapters"""
    
    @staticmethod
    def create_adapter(exchange: str, api_key: str, api_secret: str, passphrase: str = None, sandbox: bool = True) -> ExchangeAdapter:
        """Create an exchange adapter instance"""
        exchange = exchange.lower()
        
        if exchange == 'coinbase_pro':
            return CoinbaseProAdapter(api_key, api_secret, passphrase, sandbox)
        elif exchange == 'binance':
            return BinanceAdapter(api_key, api_secret, sandbox=sandbox)
        elif exchange == 'kraken':
            return KrakenAdapter(api_key, api_secret, sandbox=sandbox)
        else:
            raise ValueError(f"Unsupported exchange: {exchange}")
    
    @staticmethod
    def get_supported_exchanges() -> List[str]:
        """Get list of supported exchanges"""
        return ['coinbase_pro', 'binance', 'kraken']

# Example usage and testing
async def test_exchange_adapter():
    """Test function for exchange adapters"""
    # This would use real API credentials in a real environment
    test_credentials = {
        'coinbase_pro': {
            'api_key': 'test_key',
            'api_secret': 'test_secret',
            'passphrase': 'test_passphrase'
        }
    }
    
    logger.info("Testing exchange adapters...")
    
    for exchange_name in ExchangeFactory.get_supported_exchanges():
        if exchange_name in test_credentials:
            try:
                creds = test_credentials[exchange_name]
                adapter = ExchangeFactory.create_adapter(
                    exchange_name,
                    creds['api_key'],
                    creds['api_secret'],
                    creds.get('passphrase'),
                    sandbox=True
                )
                
                async with adapter:
                    connected, message = await adapter.test_connection()
                    logger.info(f"{exchange_name}: {message}")
                    
            except Exception as e:
                logger.error(f"Error testing {exchange_name}: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_exchange_adapter())
