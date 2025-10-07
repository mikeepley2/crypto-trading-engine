# Coinbase API Authentication - Quick Reference

## 🚨 CRITICAL REQUIREMENTS

### ✅ CORRECT Configuration
```yaml
# ConfigMap
COINBASE_BASE_URL: "https://api.coinbase.com/api/v3/brokerage"

# Secret (with actual newlines)
COINBASE_PRIVATE_KEY: |
  -----BEGIN EC PRIVATE KEY-----
  MHcCAQEEINkJvTXlxbpshkEEh/ok/b6YLDl3M/xmZN44j/aH5zVvoAoGCCqGSM49
  AwEHoUQDQgAEBjCRSYMYkIUOXmVIWraEgaNaoCyUjMp9T0KZeh3TUaQqt5enFzSi
  HjLyUjFFNCsK8ljnQw87E0Q5YHutrG9eGA==
  -----END EC PRIVATE KEY-----
```

### ❌ WRONG Configuration
```yaml
# WRONG: Old endpoint
COINBASE_BASE_URL: "https://api.coinbase.com"

# WRONG: Escaped newlines
COINBASE_PRIVATE_KEY: "-----BEGIN EC PRIVATE KEY-----\\nMHcCAQEE..."

# WRONG: HMAC authentication
# Using HMAC instead of JWT
```

## 🔧 Authentication Method

**MUST USE**: JWT with ES256 algorithm
```python
# Correct JWT implementation
token = jwt.encode(
    payload,
    private_key_obj,
    algorithm='ES256',
    headers={'kid': api_key, 'nonce': secrets.token_hex(16)}
)
```

## 🧪 Quick Test

```bash
# Test authentication
kubectl exec -n crypto-trading <pod-name> -- python -c "
import requests, jwt, secrets, time
from cryptography.hazmat.primitives import serialization

# Load credentials
api_key = 'organizations/xxx/apiKeys/xxx'
private_key = '''-----BEGIN EC PRIVATE KEY-----
MHcCAQEEINkJvTXlxbpshkEEh/ok/b6YLDl3M/xmZN44j/aH5zVvoAoGCCqGSM49
AwEHoUQDQgAEBjCRSYMYkIUOXmVIWraEgaNaoCyUjMp9T0KZeh3TUaQqt5enFzSi
HjLyUjFFNCsK8ljnQw87E0Q5YHutrG9eGA==
-----END EC PRIVATE KEY-----'''

# Create JWT
private_key_obj = serialization.load_pem_private_key(private_key.encode(), password=None)
timestamp = str(int(time.time()))
uri = 'GET api.coinbase.com/api/v3/brokerage/accounts'
payload = {'iss': 'coinbase-cloud', 'nbf': int(timestamp), 'exp': int(timestamp) + 120, 'sub': api_key, 'uri': uri}
headers = {'kid': api_key, 'nonce': secrets.token_hex(16)}
jwt_token = jwt.encode(payload, private_key_obj, algorithm='ES256', headers=headers)

# Test API
response = requests.get('https://api.coinbase.com/api/v3/brokerage/accounts', 
                       headers={'Authorization': f'Bearer {jwt_token}', 'Content-Type': 'application/json'})
print(f'Status: {response.status_code}')
print('SUCCESS!' if response.status_code == 200 else 'FAILED!')
"
```

## 🚨 Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `401 Unauthorized` | Wrong auth method (HMAC) | Use JWT authentication |
| `Invalid PEM file` | Escaped newlines (`\\n`) | Use actual newlines (`\n`) |
| `Connection refused` | Wrong endpoint | Use `/api/v3/brokerage` |
| `API key expired` | Invalid/disabled key | Generate new key in Coinbase |

## 📋 Checklist

- [ ] Using correct endpoint: `https://api.coinbase.com/api/v3/brokerage`
- [ ] Using JWT authentication (not HMAC)
- [ ] Private key has actual newlines (not escaped)
- [ ] API key is valid and active
- [ ] Headers include `Authorization: Bearer {jwt_token}`
- [ ] Test authentication before deployment

## 📚 Full Documentation

See [COINBASE_API_AUTHENTICATION.md](COINBASE_API_AUTHENTICATION.md) for complete details.

---
**Last Updated**: January 6, 2025  
**Status**: ✅ RESOLVED - JWT authentication working perfectly
