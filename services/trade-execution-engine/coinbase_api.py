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
import secrets
import base64
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
        """Load and parse the private key - supports both PEM and base64 formats"""
        try:
            # Check if it's a PEM-formatted key (legacy format)
            if self.private_key.startswith('-----BEGIN'):
                self.private_key_obj = serialization.load_pem_private_key(
                    self.private_key.encode('utf-8'),
                    password=None
                )
                self.auth_method = 'EC'
                logger.info("[+] Coinbase EC private key loaded successfully (legacy format)")
            else:
                # New base64 secret format
                self.private_key_obj = self.private_key
                self.auth_method = 'HMAC'
                logger.info("[+] Coinbase HMAC secret loaded successfully (new format)")
        except Exception as e:
            logger.error(f"[!] Failed to load private key: {e}")
            raise
    
    def _create_jwt_token(self, request_method: str, request_path: str, body: str = "") -> str:
        """Create JWT token for Coinbase Advanced Trade API authentication"""
        try:
            # Create timestamp
            timestamp = str(int(time.time()))
            
            if self.auth_method == 'HMAC':
                # New HMAC authentication method
                return self._create_hmac_jwt(request_method, request_path, body, timestamp)
            else:
                # Legacy EC authentication method
                return self._create_ec_jwt(request_method, request_path, body, timestamp)
                
        except Exception as e:
            logger.error(f"[!] Failed to create JWT token: {e}")
            raise
    
    def _create_hmac_jwt(self, request_method: str, request_path: str, body: str, timestamp: str) -> str:
        """Create JWT token using HMAC-SHA256 (new format)"""
        # Format URI exactly like the official Coinbase SDK
        # Format: METHOD hostname/path (body should NOT be included for JWT signature)
        hostname = "api.coinbase.com"
        uri = f"{request_method.upper()} {hostname}{request_path}"
        # Note: For Coinbase Advanced Trade API, the body should NOT be included in the JWT URI
            
        # Create JWT payload
        payload = {
            'iss': 'coinbase-cloud',
            'nbf': int(timestamp),
            'exp': int(timestamp) + 120,  # 2 minute expiry
            'sub': self.api_key,
            'uri': uri,
        }
        
        # Create headers
        headers = {
            'kid': self.api_key,
            'nonce': secrets.token_hex(16),  # Use random nonce
        }
        
        # Import base64 for decoding
        import base64
        
        # Decode the base64 secret
        try:
            # For new format, secret is in private_key_obj (which is the raw string)
            secret_bytes = base64.b64decode(self.private_key_obj)
            logger.debug(f"[+] Decoded secret: {len(secret_bytes)} bytes")
        except Exception as e:
            logger.error(f"[!] Failed to decode base64 secret: {e}")
            raise
        
        # Create JWT token using HMAC-SHA256
        try:
            token = jwt.encode(
                payload,
                secret_bytes,
                algorithm='HS256',
                headers=headers
            )
            logger.debug(f"[+] Created HMAC JWT token for URI: {uri}")
            return token
        except Exception as e:
            logger.error(f"[!] Failed to create HMAC JWT: {e}")
            raise
    
    def _create_ec_jwt(self, request_method: str, request_path: str, body: str, timestamp: str) -> str:
        """Create JWT token using EC private key (legacy format)"""
        # Format URI exactly like the official Coinbase SDK
        # Format: METHOD hostname/path (body should NOT be included for JWT signature)
        hostname = "api.coinbase.com"
        uri = f"{request_method.upper()} {hostname}{request_path}"
        # Note: For Coinbase Advanced Trade API, the body should NOT be included in the JWT URI
            
        # Create JWT payload
        payload = {
            'iss': 'coinbase-cloud',
            'nbf': int(timestamp),
            'exp': int(timestamp) + 120,  # 2 minute expiry
            'sub': self.api_key,
            'uri': uri,
        }
        
        # Create headers
        headers = {
            'kid': self.api_key,
            'nonce': timestamp,
        }
        
        # Create JWT token using the private key object directly
        try:
            token = jwt.encode(
                payload,
                self.private_key_obj,
                algorithm='ES256',
                headers=headers
            )
            logger.debug(f"[+] Created EC JWT token for URI: {uri}")
            return token
        except Exception as e:
            logger.error(f"[!] Failed to create EC JWT: {e}")
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
        
        # Debug logging
        logger.debug(f"[+] Making {method.upper()} request to: {url}")
        logger.debug(f"[+] Request body: {body}")
        logger.debug(f"[+] Using auth method: {self.auth_method}")
        
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
        """Get account balance using portfolio breakdown endpoint"""
        try:
            # Get portfolio breakdown
            portfolios_response = self._make_request('GET', '/api/v3/brokerage/portfolios')
            portfolios = portfolios_response.get('portfolios', [])
            
            if not portfolios:
                return 0.0
                
            portfolio_id = portfolios[0]['uuid']
            breakdown_response = self._make_request('GET', f'/api/v3/brokerage/portfolios/{portfolio_id}')
            breakdown = breakdown_response.get('breakdown', {})
            
            # Extract USD cash from spot positions
            spot_positions = breakdown.get('spot_positions', [])
            for position in spot_positions:
                if position.get('is_cash') and position.get('asset') == currency:
                    return float(position.get('available_to_trade_fiat', 0))
                    
            return 0.0
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to get account balance: {e}")
            return 0.0
    
    def get_position_value(self, currency: str) -> float:
        """Get the USD value of a specific cryptocurrency position"""
        try:
            # Get portfolio breakdown
            portfolios_response = self._make_request('GET', '/api/v3/brokerage/portfolios')
            portfolios = portfolios_response.get('portfolios', [])
            
            if not portfolios:
                return 0.0
                
            portfolio_id = portfolios[0]['uuid']
            breakdown_response = self._make_request('GET', f'/api/v3/brokerage/portfolios/{portfolio_id}')
            breakdown = breakdown_response.get('breakdown', {})
            
            # Look for the currency in spot positions
            spot_positions = breakdown.get('spot_positions', [])
            for position in spot_positions:
                if position.get('asset') == currency and not position.get('is_cash'):
                    # Calculate USD value: quantity * current_price
                    quantity = float(position.get('total_balance_fiat', 0))
                    if quantity > 0:
                        logger.debug(f"[POSITION] {currency}: ${quantity:.2f}")
                        return quantity
                    
            logger.debug(f"[POSITION] {currency}: $0.00 (no position found)")
            return 0.0
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to get position value for {currency}: {e}")
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
            
            # Coinbase Advanced Trade ticker response structure:
            # - No top-level 'price' field
            # - Has 'best_bid' and 'best_ask' 
            # - Has 'trades' array with recent trade prices
            
            # Priority 1: Use midpoint of bid/ask spread
            best_bid = response.get('best_bid')
            best_ask = response.get('best_ask')
            
            if best_bid and best_ask:
                try:
                    bid_price = float(best_bid)
                    ask_price = float(best_ask)
                    midpoint = (bid_price + ask_price) / 2.0
                    logger.debug(f"[PRICE] {product_id}: Bid={bid_price}, Ask={ask_price}, Midpoint={midpoint}")
                    return midpoint
                except (ValueError, TypeError):
                    pass
            
            # Priority 2: Use most recent trade price
            trades = response.get('trades', [])
            if trades and len(trades) > 0:
                try:
                    recent_trade_price = float(trades[0].get('price', 0))
                    if recent_trade_price > 0:
                        logger.debug(f"[PRICE] {product_id}: Using recent trade price={recent_trade_price}")
                        return recent_trade_price
                except (ValueError, TypeError, IndexError):
                    pass
            
            # Priority 3: Use best_bid as fallback
            if best_bid:
                try:
                    bid_price = float(best_bid)
                    if bid_price > 0:
                        logger.warning(f"[PRICE] {product_id}: Using bid price as fallback={bid_price}")
                        return bid_price
                except (ValueError, TypeError):
                    pass
            
            logger.error(f"[ERROR] No valid price data found for {product_id}")
            return 0.0
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to get current price for {product_id}: {e}")
            return 0.0
    
    def _check_trading_status(self, product_id: str) -> bool:
        """Check if trading is enabled for a specific product"""
        try:
            # Known working trading pairs (whitelist)
            known_working_pairs = {
                'BTC-USD', 'ETH-USD', 'LTC-USD', 'BCH-USD', 'ADA-USD', 
                'DOT-USD', 'LINK-USD', 'XLM-USD', 'ALGO-USD', 'ATOM-USD',
                'SOL-USD', 'MATIC-USD', 'AVAX-USD', 'UNI-USD', 'AAVE-USD'
            }
            
            # If it's a known working pair, allow it
            if product_id in known_working_pairs:
                logger.debug(f"[TRADING_STATUS] {product_id} is whitelisted")
                return True
            
            # For other pairs, check with Coinbase API
            try:
                product = self.get_product_info(product_id)
                
                if not product:
                    logger.warning(f"[TRADING_STATUS] {product_id} not found")
                    return False
                
                # Check if the product is online and trading is enabled
                status = product.get('status', '').lower()
                trading_disabled = product.get('trading_disabled', False)
                cancel_only = product.get('cancel_only', False)
                
                if status != 'online':
                    logger.warning(f"[TRADING_STATUS] {product_id} is not online (status: {status})")
                    return False
                
                if trading_disabled:
                    logger.warning(f"[TRADING_STATUS] Trading is disabled for {product_id}")
                    return False
                
                if cancel_only:
                    logger.warning(f"[TRADING_STATUS] {product_id} is in cancel-only mode")
                    return False
                
                logger.debug(f"[TRADING_STATUS] {product_id} is available for trading")
                return True
                
            except Exception as e:
                # If we can't check status, default to allowing known working pairs
                if product_id in known_working_pairs:
                    logger.warning(f"[TRADING_STATUS] {product_id} API check failed but whitelisted: {e}")
                    return True
                logger.error(f"[TRADING_STATUS] Unable to verify trading status for {product_id}: {e}")
                return False
                
        except Exception as e:
            logger.error(f"[TRADING_STATUS] Error checking trading status for {product_id}: {e}")
            return False

    def place_market_order(self, product_id: str, side: str, size: str) -> Dict:
        """Place a market order"""
        # Check trading status first
        if not self._check_trading_status(product_id):
            error_msg = f"Trading not enabled for {product_id}. This asset may not be available for trading on Coinbase Advanced Trade API."
            logger.error(f"[TRADING_BLOCKED] {error_msg}")
            raise Exception(error_msg)
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
            
            logger.info(f"[ORDER_REQUEST] Placing order: {side} {size} {product_id}")
            logger.info(f"[ORDER_DATA] Request data: {data}")
            
            response = self._make_request('POST', '/api/v3/brokerage/orders', data=data)
            
            logger.info(f"[ORDER_RESPONSE] Raw response: {response}")
            logger.info(f"[ORDER_TYPE] Response type: {type(response)}")
            
            # Extract order_id from response structure
            # Coinbase Advanced Trade API can return different response structures
            order_id = None
            if response and isinstance(response, dict):
                
                # Check if the order actually failed
                if 'error_response' in response:
                    error_msg = response.get('error_response', {}).get('message', 'Unknown error')
                    error_code = response.get('error_response', {}).get('error', 'UNKNOWN_ERROR')
                    logger.error(f"[ORDER_ID] Order failed - {error_code}: {error_msg}")
                    raise Exception(f"Order placement failed: {error_code} - {error_msg}")
                
                # Try to extract order_id from successful response
                if 'order_id' in response:
                    order_id = response['order_id']
                    logger.info(f"[ORDER_ID] Found order_id directly: {order_id}")
                elif 'success_response' in response and isinstance(response['success_response'], dict):
                    success_resp = response['success_response']
                    if 'order_id' in success_resp:
                        order_id = success_resp['order_id']
                        logger.info(f"[ORDER_ID] Found order_id in success_response: {order_id}")
                    else:
                        logger.warning(f"[ORDER_ID] No order_id in success_response keys: {list(success_resp.keys())}")
                elif 'order' in response and 'order_id' in response['order']:
                    order_id = response['order']['order_id']
                    logger.info(f"[ORDER_ID] Found order_id in order: {order_id}")
                else:
                    logger.warning(f"[ORDER_ID] No order_id found in response keys: {list(response.keys())}")
                    # Log full response for debugging
                    logger.warning(f"[ORDER_ID] Full response for debugging: {response}")
            else:
                logger.error(f"[ORDER_ID] Invalid response type or None: {response}")
            
            # Only log success if we got a valid response
            if response:
                logger.info(f"[SUCCESS] Market order placed: {side} {size} {product_id} (order_id: {order_id})")
            else:
                logger.error(f"[ERROR] Market order failed - no response: {side} {size} {product_id}")
            
            # Return standardized response with order_id
            return {
                'order_id': order_id,
                'original_response': response
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to place market order: {e}")
            logger.error(f"[ERROR] Exception type: {type(e)}")
            import traceback
            logger.error(f"[ERROR] Traceback: {traceback.format_exc()}")
            raise
    
    def place_limit_order(self, product_id: str, side: str, size: str, price: str) -> Dict:
        """Place a limit order"""
        # Check trading status first
        if not self._check_trading_status(product_id):
            error_msg = f"Trading not enabled for {product_id}. This asset may not be available for trading on Coinbase Advanced Trade API."
            logger.error(f"[TRADING_BLOCKED] {error_msg}")
            raise Exception(error_msg)
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
        """Get crypto positions using portfolio breakdown endpoint"""
        try:
            # Get portfolio breakdown
            portfolios_response = self._make_request('GET', '/api/v3/brokerage/portfolios')
            portfolios = portfolios_response.get('portfolios', [])
            
            if not portfolios:
                return []
                
            portfolio_id = portfolios[0]['uuid']
            breakdown_response = self._make_request('GET', f'/api/v3/brokerage/portfolios/{portfolio_id}')
            breakdown = breakdown_response.get('breakdown', {})
            
            positions = []
            spot_positions = breakdown.get('spot_positions', [])
            
            for position in spot_positions:
                if not position.get('is_cash'):
                    asset = position.get('asset')
                    balance_crypto = float(position.get('total_balance_crypto', 0))
                    balance_fiat = float(position.get('total_balance_fiat', 0))
                    
                    if balance_crypto > 0:
                        current_price = balance_fiat / balance_crypto if balance_crypto > 0 else 0.0
                        
                        positions.append({
                            'currency': asset,
                            'available_balance': balance_crypto,
                            'hold': 0.0,
                            'value': balance_fiat,
                            'current_price': current_price
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
        # Read from environment variables with fallback defaults
        self.max_daily_trades = int(os.getenv('MAX_DAILY_TRADES', '200'))
        self.max_daily_loss = float(os.getenv('MAX_DAILY_LOSS_USD', '500.0'))
        self.last_reset_date = None  # Track when we last reset counters
        
    def can_place_trade(self, trade_size_usd: float, action: str = 'buy') -> bool:
        """Check if trade can be placed within risk limits"""
        logger.info(f"🔍 Risk check for {action} trade size: ${trade_size_usd}")
        
        # Reset daily counters if new day
        self.reset_daily_counters()
        
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
        
        # For sell orders, allow rebalancing without balance restriction
        if action.lower() == 'sell':
            logger.info(f"✅ Sell order approved for rebalancing: ${trade_size_usd}")
            return True
        
        # Check account balance for buy orders only
        try:
            logger.info("🔍 Checking account balance for buy order...")
            balance = self.api.get_account_balance('USD')
            logger.info(f"💰 Current USD balance: ${balance}")
            
            # Allow trades up to 90% of balance for buy orders (reduced from 95% for safety)
            # But also allow trades up to $50 even if balance is low (for small automated trades)
            max_allowed = max(balance * 0.90, 50.0)
            logger.info(f"🔍 Max allowed buy trade size (90% of balance or $50 min): ${max_allowed}")
            
            if trade_size_usd > max_allowed:
                logger.warning(f"⚠️ Buy trade size ${trade_size_usd} too large for balance ${balance} (max: ${max_allowed})")
                return False
            
            # More lenient minimum balance check - allow trades if balance > trade_size * 1.1
            min_required_balance = trade_size_usd * 1.1  # 10% buffer
            if balance < min_required_balance:
                logger.warning(f"⚠️ Account balance ${balance} too low for ${trade_size_usd} trade (need ${min_required_balance})")
                return False
                
            logger.info(f"✅ Balance check passed: ${trade_size_usd} <= ${max_allowed}, balance ${balance} > required ${min_required_balance}")
            
        except Exception as e:
            logger.error(f"⚠️ Failed to check balance: {e}")
            # For automated trading, allow trade if balance check fails (might be API issue)
            logger.info("⚠️ Allowing trade despite balance check failure")
        
        logger.info(f"✅ Risk check passed for ${trade_size_usd} {action} trade")
        return True
    
    def reset_daily_counters(self):
        """Reset daily counters for new day"""
        from datetime import datetime
        current_date = datetime.now().date()
        if self.last_reset_date != current_date:
            logger.info(f"🔄 Resetting daily counters - Previous: {self.daily_trades} trades, ${self.daily_loss} loss")
            self.daily_trades = 0
            self.daily_loss = 0.0
            self.last_reset_date = current_date
    
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
    
    # Try multiple paths for the environment file
    load_dotenv('.env.live')
    load_dotenv('/e/git/aitest/.env.live')
    load_dotenv('e:/git/aitest/.env.live')
    
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
