# Coinbase Advanced Trade API Authentication Guide

## 🚨 **CRITICAL DISCOVERY: Manual vs SDK Authentication**

### **Problem We Encountered:**
- Manual HMAC-SHA256 authentication implementation was returning `401 Unauthorized`
- API credentials were valid but our manual signature generation was incorrect
- Multiple attempts with different key formats failed

### **Root Cause:**
The manual authentication implementation in our trade executor was not correctly implementing the Coinbase Advanced Trade API authentication protocol, even though the endpoint and headers were correct.

### **Solution: Use Official Coinbase SDK**

**❌ DON'T USE:** Manual HMAC authentication
```python
# This approach failed with 401 Unauthorized
def create_coinbase_signature(method, path, timestamp, body=""):
    message = f"{timestamp}{method}{path}{body}"
    signature = base64.b64encode(
        hmac.new(
            base64.b64decode(api_secret),
            message.encode(), 
            hashlib.sha256
        ).digest()
    )
    return signature.decode()
```

**✅ USE:** Official Coinbase Advanced Trade Python SDK
```python
from coinbase.rest import RESTClient

def get_coinbase_client():
    api_key = os.getenv('COINBASE_API_KEY')
    api_secret = os.getenv('COINBASE_PRIVATE_KEY', '')
    return RESTClient(api_key=api_key, api_secret=api_secret)

# Usage
client = get_coinbase_client()
accounts = client.get_accounts()  # Works perfectly!
```

## 📋 **Implementation Instructions**

### **1. Install the Official SDK**
```bash
pip install coinbase-advanced-py
```

### **2. Use SDK for All API Calls**
```python
from coinbase.rest import RESTClient

# Initialize client
client = RESTClient(api_key=api_key, api_secret=api_secret)

# Get accounts
accounts = client.get_accounts()

# Place orders
order_result = client.create_order(
    client_order_id=f"trade_{symbol}_{side}_{timestamp}",
    product_id=symbol,
    side=side.upper(),
    order_configuration=order_config
)
```

### **3. Environment Variables**
```bash
COINBASE_API_KEY=organizations/5f04b9a1-3467-4f94-bb5c-2769d89fe5d6/apiKeys/7dd53cef-f159-45af-947f-a861eeb79204
COINBASE_PRIVATE_KEY=-----BEGIN EC PRIVATE KEY-----
MHcCAQEEINkJvTXlxbpshkEEh/ok/b6YLDl3M/xmZN44j/aH5zVvoAoGCCqGSM49
AwEHoUQDQgAEBjCRSYMYkIUOXmVIWraEgaNaoCyUjMp9T0KZeh3TUaQqt5enFzSi
HjLyUjFFNCsK8ljnQw87E0Q5YHutrG9eGA==
-----END EC PRIVATE KEY-----
```

## 🔧 **Working Implementation**

### **Trade Executor with SDK (`coinbase_trade_executor_sdk.py`)**
```python
from coinbase.rest import RESTClient

def get_coinbase_client():
    """Initialize Coinbase REST client"""
    api_key = os.getenv('COINBASE_API_KEY')
    api_secret = os.getenv('COINBASE_PRIVATE_KEY', '')
    
    if not api_key or not api_secret:
        raise Exception("Missing Coinbase API credentials")
    
    return RESTClient(api_key=api_key, api_secret=api_secret)

def get_coinbase_accounts():
    """Get Coinbase account balances using SDK"""
    try:
        client = get_coinbase_client()
        accounts = client.get_accounts()
        return accounts
    except Exception as e:
        raise Exception(f"Failed to get accounts: {e}")

def place_coinbase_order(symbol, side, size_usd):
    """Place a real order using Coinbase SDK"""
    try:
        client = get_coinbase_client()
        
        # Ensure symbol is in correct format (e.g., BTC-USD)
        if '-USD' not in symbol:
            symbol = f"{symbol}-USD"
        
        # Create order configuration for market order
        order_config = {
            "market_market_ioc": {
                "quote_size": str(size_usd)
            }
        }
        
        # Place the order
        order_result = client.create_order(
            client_order_id=f"trade_{symbol}_{side}_{int(time.time())}",
            product_id=symbol,
            side=side.upper(),
            order_configuration=order_config
        )
        
        return order_result
        
    except Exception as e:
        raise Exception(f"Failed to place order: {e}")
```

## ✅ **Verification Steps**

### **1. Test API Connection**
```python
# Test with SDK
from coinbase.rest import RESTClient
client = RESTClient(api_key=api_key, api_secret=api_secret)
accounts = client.get_accounts()
print(f"SUCCESS: Connected to {len(accounts.accounts)} accounts")
```

### **2. Test Trade Execution**
```python
# Test trade execution
trade_request = {
    "symbol": "BTC",
    "action": "BUY", 
    "size_usd": 25.0,
    "order_type": "MARKET"
}
response = requests.post('http://localhost:8024/execute_trade', json=trade_request)
print(f"Trade Result: {response.json()}")
```

## 🎯 **Key Takeaways**

1. **Always use the official Coinbase SDK** for authentication
2. **Manual HMAC implementation is error-prone** and not recommended
3. **API credentials were valid** - the issue was authentication method
4. **SDK handles all authentication complexity** automatically
5. **Test with small amounts first** to verify functionality

## 🚀 **Deployment Notes**

- Update Kubernetes deployments to use SDK-based trade executor
- Ensure `coinbase-advanced-py` is installed in containers
- Use environment variables for API credentials
- Test authentication before deploying to production

## 📚 **References**

- [Coinbase Advanced Trade Python SDK](https://github.com/coinbase/coinbase-advanced-py)
- [Coinbase Advanced Trade API Documentation](https://docs.cdp.coinbase.com/advanced-trade/docs/rest-api-overview/)
- [Working Implementation: `coinbase_trade_executor_sdk.py`](./coinbase_trade_executor_sdk.py)

---

**⚠️ IMPORTANT:** This issue has occurred multiple times. Always use the official SDK instead of manual authentication implementation.
