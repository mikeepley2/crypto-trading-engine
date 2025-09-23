#!/usr/bin/env python3
"""
Coinbase Advanced Trade API Integration
Secure wrapper for live cryptocurrency trading
"""

import os
import json
import time
import hmac
import hashlib
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ec
import jwt
import logging

logger = logging.getLogger(__name__)

class CoinbaseAdvancedTradeAPI:
    """Coinbase Advanced Trade API client with authentication and safety features"""
    
    def __init__(self, api_key: str, private_key: str, base_url: str = "https://api.coinbase.com"):
        self.api_key = api_key
        self.private_key = private_key
        self.base_url = base_url
        self.session = requests.Session()
        
        # Parse the private key
        self._load_private_key()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
    def _load_private_key(self):
        """Load and parse the EC private key"""
        try:
            # Load the private key directly (already properly formatted in .env)
            self.private_key_obj = serialization.load_pem_private_key(
                self.private_key.encode('utf-8'),
                password=None
            )
            logger.info("[+] Coinbase private key loaded successfully")
        except Exception as e:
            logger.error(f"[!] Failed to load private key: {e}")
            raise
    
    def _create_jwt_token(self, request_method: str, request_path: str, body: str = "") -> str:
        """Create JWT token for Coinbase Advanced Trade API authentication"""
        try:
            # Create timestamp
            timestamp = str(int(time.time()))
            
            # Format URI like the official SDK: METHOD hostname/path
            hostname = "api.coinbase.com"
            jwt_uri = f"{request_method.upper()} {hostname}{request_path}"
            if body:
                jwt_uri += body
                
            # Create JWT payload
            payload = {
                'iss': 'coinbase-cloud',
                'nbf': int(timestamp),
                'exp': int(timestamp) + 120,  # 2 minute expiry
                'sub': self.api_key,
                'uri': jwt_uri,
            }
            
            # Create headers
            headers = {
                'kid': self.api_key,
                'nonce': timestamp,
            }
            
            # Create JWT token using the private key object directly
            token = jwt.encode(
                payload,
                self.private_key_obj,
                algorithm='ES256',
                headers=headers
            )
            
            return token
            
        except Exception as e:
            logger.error(f"[!] Failed to create JWT token: {e}")
            raise
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict:
        """Make authenticated request to Coinbase API"""
        self._rate_limit()
        
        # Prepare request
        url = f"{self.base_url}{endpoint}"
        body = json.dumps(data) if data else ""
        
        # Create JWT token
        jwt_token = self._create_jwt_token(method, endpoint, body)
        
        # Set headers
        headers = {
            'Authorization': f'Bearer {jwt_token}',
            'Content-Type': 'application/json',
        }
        
        try:
            # Make request
            if method.upper() == 'GET':
                response = self.session.get(url, headers=headers, params=params)
            elif method.upper() == 'POST':
                response = self.session.post(url, headers=headers, json=data)
            elif method.upper() == 'PUT':
                response = self.session.put(url, headers=headers, json=data)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Check response
            response.raise_for_status()
            
            # Parse JSON response
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[ERROR] Coinbase API request failed: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response: {e.response.text}")
            raise
    
    def get_accounts(self) -> List[Dict]:
        """Get all accounts"""
        try:
            response = self._make_request('GET', '/api/v3/brokerage/accounts')
            return response.get('accounts', [])
        except Exception as e:
            logger.error(f"[!] Failed to get accounts: {e}")
            raise
    
    def get_account_balance(self, currency: str = 'USD') -> float:
        """Get account balance for specific currency"""
        try:
            accounts = self.get_accounts()
            
            for account in accounts:
                if account.get('currency') == currency:
                    return float(account.get('available_balance', {}).get('value', 0))
            
            return 0.0
        except Exception as e:
            logger.error(f"[ERROR] Failed to get account balance: {e}")
            return 0.0
    
    def get_product_info(self, product_id: str) -> Dict:
        """Get product information"""
        try:
            response = self._make_request('GET', f'/api/v3/brokerage/products/{product_id}')
            return response
        except Exception as e:
            logger.error(f"[ERROR] Failed to get product info for {product_id}: {e}")
            raise
    
    def get_current_price(self, product_id: str) -> float:
        """Get current price for a product"""
        try:
            response = self._make_request('GET', f'/api/v3/brokerage/products/{product_id}/ticker')
            return float(response.get('price', 0))
        except Exception as e:
            logger.error(f"[ERROR] Failed to get current price for {product_id}: {e}")
            return 0.0
    
    def place_market_order(self, product_id: str, side: str, size: str) -> Dict:
        """Place a market order"""
        try:
            data = {
                'client_order_id': f"{product_id}_{side}_{int(time.time())}",
                'product_id': product_id,
                'side': side.upper(),
                'order_configuration': {
                    'market_market_ioc': {
                        'base_size': size
                    }
                }
            }
            
            response = self._make_request('POST', '/api/v3/brokerage/orders', data=data)
            logger.info(f"[SUCCESS] Market order placed: {side} {size} {product_id}")
            return response
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to place market order: {e}")
            raise
    
    def place_limit_order(self, product_id: str, side: str, size: str, price: str) -> Dict:
        """Place a limit order"""
        try:
            data = {
                'client_order_id': f"{product_id}_{side}_{int(time.time())}",
                'product_id': product_id,
                'side': side.upper(),
                'order_configuration': {
                    'limit_limit_gtc': {
                        'base_size': size,
                        'limit_price': price
                    }
                }
            }
            
            response = self._make_request('POST', '/api/v3/brokerage/orders', data=data)
            logger.info(f"[SUCCESS] Limit order placed: {side} {size} {product_id} @ {price}")
            return response
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to place limit order: {e}")
            raise
    
    def get_order_status(self, order_id: str) -> Dict:
        """Get order status"""
        try:
            response = self._make_request('GET', f'/api/v3/brokerage/orders/historical/{order_id}')
            return response.get('order', {})
        except Exception as e:
            logger.error(f"[ERROR] Failed to get order status: {e}")
            raise
    
    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an order"""
        try:
            data = {'order_ids': [order_id]}
            response = self._make_request('POST', '/api/v3/brokerage/orders/batch_cancel', data=data)
            logger.info(f"[SUCCESS] Order cancelled: {order_id}")
            return response
        except Exception as e:
            logger.error(f"[ERROR] Failed to cancel order: {e}")
            raise
    
    def get_positions(self) -> List[Dict]:
        """Get all positions"""
        try:
            accounts = self.get_accounts()
            positions = []
            
            for account in accounts:
                if account.get('currency') != 'USD':
                    balance = float(account.get('available_balance', {}).get('value', 0))
                    if balance > 0:
                        positions.append({
                            'currency': account.get('currency'),
                            'balance': balance,
                            'value_usd': balance  # This would need price conversion
                        })
            
            return positions
        except Exception as e:
            logger.error(f"[ERROR] Failed to get positions: {e}")
            return []
    
    def validate_connection(self) -> bool:
        """Validate API connection and credentials"""
        try:
            accounts = self.get_accounts()
            logger.info(f"[+] Coinbase API connection validated - {len(accounts)} accounts found")
            return True
        except Exception as e:
            logger.error(f"[!] Coinbase API connection failed: {e}")
            return False

class CoinbaseRiskManager:
    """Risk management for Coinbase trading"""
    
    def __init__(self, coinbase_api: CoinbaseAdvancedTradeAPI, max_position_size_usd: float = 100.0):
        self.api = coinbase_api
        self.max_position_size_usd = max_position_size_usd
        self.daily_trades = 0
        self.daily_loss = 0.0
        self.max_daily_trades = 10
        self.max_daily_loss = 50.0
        
    def can_place_trade(self, trade_size_usd: float) -> bool:
        """Check if trade can be placed within risk limits"""
        # Check position size
        if trade_size_usd > self.max_position_size_usd:
            logger.warning(f"⚠️ Trade size ${trade_size_usd} exceeds max position size ${self.max_position_size_usd}")
            return False
        
        # Check daily trade count
        if self.daily_trades >= self.max_daily_trades:
            logger.warning(f"⚠️ Daily trade limit reached: {self.daily_trades}/{self.max_daily_trades}")
            return False
        
        # Check daily loss
        if self.daily_loss >= self.max_daily_loss:
            logger.warning(f"⚠️ Daily loss limit reached: ${self.daily_loss}/${self.max_daily_loss}")
            return False
        
        # Check account balance
        balance = self.api.get_account_balance('USD')
        if trade_size_usd > balance * 0.5:  # Don't use more than 50% of balance
            logger.warning(f"⚠️ Trade size ${trade_size_usd} too large for balance ${balance}")
            return False
        
        return True
    
    def record_trade(self, trade_result: Dict):
        """Record trade for risk tracking"""
        self.daily_trades += 1
        
        # Track losses (simplified)
        if 'loss' in trade_result:
            self.daily_loss += abs(trade_result['loss'])
        
        logger.info(f"📊 Risk stats - Trades: {self.daily_trades}, Loss: ${self.daily_loss}")

if __name__ == "__main__":
    # Test the API connection
    import os
    from dotenv import load_dotenv
    
    load_dotenv('.env.live')
    
    api_key = os.getenv('COINBASE_API_KEY')
    private_key = os.getenv('COINBASE_PRIVATE_KEY')
    
    if api_key and private_key:
        try:
            api = CoinbaseAdvancedTradeAPI(api_key, private_key)
            if api.validate_connection():
                print("✅ Coinbase API integration successful!")
                
                # Get account info
                balance = api.get_account_balance('USD')
                print(f"USD Balance: ${balance}")
                
                # Get BTC price
                btc_price = api.get_current_price('BTC-USD')
                print(f"BTC Price: ${btc_price}")
                
            else:
                print("❌ Coinbase API connection failed")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("❌ Missing API credentials")
