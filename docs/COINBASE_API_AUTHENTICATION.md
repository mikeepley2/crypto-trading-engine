# Coinbase Advanced Trade API Authentication Guide

## Overview

This document provides comprehensive guidance for Coinbase Advanced Trade API authentication to prevent authentication issues in the future.

## Critical Authentication Requirements

### 1. API Endpoint
**MUST USE**: `https://api.coinbase.com/api/v3/brokerage`
- ❌ **WRONG**: `https://api.coinbase.com` (old Coinbase Pro API)
- ✅ **CORRECT**: `https://api.coinbase.com/api/v3/brokerage` (Coinbase Advanced Trade API)

### 2. Authentication Method
**MUST USE**: JWT Authentication with EC Private Key
- ❌ **WRONG**: HMAC authentication (old method)
- ✅ **CORRECT**: JWT authentication with ES256 algorithm

### 3. Private Key Format
**MUST USE**: Proper PEM format with actual newlines
- ❌ **WRONG**: Escaped newlines (`\\n`) in Kubernetes secrets
- ✅ **CORRECT**: Actual newlines (`\n`) in PEM format

## Authentication Implementation

### JWT Token Creation
```python
import jwt
import secrets
import time
from cryptography.hazmat.primitives import serialization

def create_jwt_token(api_key, private_key, method, path, body=""):
    """Create JWT token for Coinbase Advanced Trade API"""
    try:
        # Parse the private key
        private_key_obj = serialization.load_pem_private_key(
            private_key.encode('utf-8'),
            password=None
        )
        
        # Create timestamp
        timestamp = str(int(time.time()))
        
        # Format URI for JWT
        hostname = "api.coinbase.com"
        uri = f"{method.upper()} {hostname}{path}"
        
        # Create JWT payload
        payload = {
            'iss': 'coinbase-cloud',
            'nbf': int(timestamp),
            'exp': int(timestamp) + 120,  # 2 minute expiry
            'sub': api_key,
            'uri': uri,
        }
        
        # Create headers
        headers = {
            'kid': api_key,
            'nonce': secrets.token_hex(16),
        }
        
        # Create JWT token
        token = jwt.encode(
            payload,
            private_key_obj,
            algorithm='ES256',
            headers=headers
        )
        
        return token
        
    except Exception as e:
        raise Exception(f"Failed to create JWT token: {e}")
```

### API Request Headers
```python
headers = {
    'Authorization': f'Bearer {jwt_token}',
    'Content-Type': 'application/json'
}
```

## Kubernetes Configuration

### ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: trade-exec-coinbase-config
  namespace: crypto-trading
data:
  COINBASE_BASE_URL: "https://api.coinbase.com/api/v3/brokerage"
  COINBASE_API_MODE: "live"
  # ... other config
```

### Secret (CRITICAL: Proper PEM Format)
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: trade-exec-coinbase-secrets
  namespace: crypto-trading
type: Opaque
data:
  COINBASE_API_KEY: <base64-encoded-api-key>
  COINBASE_PRIVATE_KEY: <base64-encoded-pem-with-actual-newlines>
  DB_PASSWORD: <base64-encoded-password>
```

### Creating Secret with Correct Format
```bash
# Create secret with proper PEM format (actual newlines, not escaped)
kubectl create secret generic trade-exec-coinbase-secrets \
  --from-literal=COINBASE_API_KEY="organizations/xxx/apiKeys/xxx" \
  --from-literal=COINBASE_PRIVATE_KEY="-----BEGIN EC PRIVATE KEY-----
MHcCAQEEINkJvTXlxbpshkEEh/ok/b6YLDl3M/xmZN44j/aH5zVvoAoGCCqGSM49
AwEHoUQDQgAEBjCRSYMYkIUOXmVIWraEgaNaoCyUjMp9T0KZeh3TUaQqt5enFzSi
HjLyUjFFNCsK8ljnQw87E0Q5YHutrG9eGA==
-----END EC PRIVATE KEY-----
" \
  --from-literal=DB_PASSWORD="password"
```

## Common Issues and Solutions

### Issue 1: 401 Unauthorized
**Cause**: Wrong authentication method (HMAC instead of JWT)
**Solution**: Use JWT authentication with ES256 algorithm

### Issue 2: Invalid PEM File Error
**Cause**: Escaped newlines in Kubernetes secret (`\\n` instead of `\n`)
**Solution**: Create secret with actual newlines in PEM format

### Issue 3: Wrong API Endpoint
**Cause**: Using old Coinbase Pro API endpoint
**Solution**: Use `https://api.coinbase.com/api/v3/brokerage`

### Issue 4: API Key Expired/Disabled
**Cause**: API key is no longer valid on Coinbase side
**Solution**: Generate new API key in Coinbase Advanced Trade

## Testing Authentication

### Test Script
```python
#!/usr/bin/env python3
import requests
import json
import time
import jwt
import secrets
from cryptography.hazmat.primitives import serialization

def test_coinbase_auth():
    # Load credentials
    api_key = "organizations/xxx/apiKeys/xxx"
    private_key = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEINkJvTXlxbpshkEEh/ok/b6YLDl3M/xmZN44j/aH5zVvoAoGCCqGSM49
AwEHoUQDQgAEBjCRSYMYkIUOXmVIWraEgaNaoCyUjMp9T0KZeh3TUaQqt5enFzSi
HjLyUjFFNCsK8ljnQw87E0Q5YHutrG9eGA==
-----END EC PRIVATE KEY-----"""
    
    # Create JWT token
    private_key_obj = serialization.load_pem_private_key(
        private_key.encode('utf-8'), password=None
    )
    
    timestamp = str(int(time.time()))
    uri = "GET api.coinbase.com/api/v3/brokerage/accounts"
    
    payload = {
        'iss': 'coinbase-cloud',
        'nbf': int(timestamp),
        'exp': int(timestamp) + 120,
        'sub': api_key,
        'uri': uri,
    }
    
    headers = {
        'kid': api_key,
        'nonce': secrets.token_hex(16),
    }
    
    jwt_token = jwt.encode(
        payload, private_key_obj, algorithm='ES256', headers=headers
    )
    
    # Test API call
    url = "https://api.coinbase.com/api/v3/brokerage/accounts"
    request_headers = {
        'Authorization': f'Bearer {jwt_token}',
        'Content-Type': 'application/json'
    }
    
    response = requests.get(url, headers=request_headers, timeout=10)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:200]}")
    
    return response.status_code == 200

if __name__ == "__main__":
    test_coinbase_auth()
```

## API Endpoints

### Accounts
- **Endpoint**: `GET /api/v3/brokerage/accounts`
- **Full URL**: `https://api.coinbase.com/api/v3/brokerage/accounts`

### Portfolios
- **Endpoint**: `GET /api/v3/brokerage/portfolios`
- **Full URL**: `https://api.coinbase.com/api/v3/brokerage/portfolios`

### Products
- **Endpoint**: `GET /api/v3/brokerage/products/{product_id}`
- **Full URL**: `https://api.coinbase.com/api/v3/brokerage/products/BTC-USD`

### Orders
- **Endpoint**: `POST /api/v3/brokerage/orders`
- **Full URL**: `https://api.coinbase.com/api/v3/brokerage/orders`

## Troubleshooting Checklist

1. ✅ **API Endpoint**: Using `https://api.coinbase.com/api/v3/brokerage`
2. ✅ **Authentication**: Using JWT with ES256 algorithm
3. ✅ **Private Key**: Proper PEM format with actual newlines
4. ✅ **API Key**: Valid and active in Coinbase Advanced Trade
5. ✅ **Headers**: `Authorization: Bearer {jwt_token}`
6. ✅ **Content-Type**: `application/json`

## Key Takeaways

1. **Always use JWT authentication** - HMAC is deprecated
2. **Always use the correct endpoint** - `/api/v3/brokerage` not `/api/v3`
3. **Always use proper PEM format** - actual newlines, not escaped
4. **Always test authentication** before deploying to production
5. **Always check API key status** in Coinbase Advanced Trade dashboard

## Last Updated
- **Date**: January 6, 2025
- **Issue Resolved**: 401 Unauthorized errors due to wrong authentication method
- **Status**: ✅ RESOLVED - JWT authentication working perfectly
