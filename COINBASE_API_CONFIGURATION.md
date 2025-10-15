# Coinbase API Configuration Guide

## Overview
This document provides comprehensive instructions for configuring Coinbase API credentials in the crypto trading system.

## API Key Location
The valid Coinbase API key is stored in:
- **File**: `coinbase_api_key.json` (root directory)
- **Kubernetes Secret**: `coinbase-api-secrets` (crypto-trading namespace)
- **Kubernetes ConfigMap**: `coinbase-api-config` (crypto-trading namespace)

## Valid API Key Details
```json
{
   "name": "organizations/5f04b9a1-3467-4f94-bb5c-2769d89fe5d6/apiKeys/7dd53cef-f159-45af-947f-a861eeb79204",
   "privateKey": "-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEINkJvTXlxbpshkEEh/ok/b6YLDl3M/xmZN44j/aH5zVvoAoGCCqGSM49\nAwEHoUQDQgAEBjCRSYMYkIUOXmVIWraEgaNaoCyUjMp9T0KZeh3TUaQqt5enFzSi\nHjLyUjFFNCsK8ljnQw87E0Q5YHutrG9eGA==\n-----END EC PRIVATE KEY-----\n"
}
```

## Kubernetes Configuration

### Secret: `coinbase-api-secrets`
Contains the base64-encoded API credentials:
- `COINBASE_API_KEY`: The API key name
- `COINBASE_PRIVATE_KEY`: The EC private key in PEM format
- `COINBASE_BASE_URL`: The Coinbase API base URL

### ConfigMap: `coinbase-api-config`
Contains the API configuration:
- `COINBASE_API_MODE`: "live"
- `COINBASE_BASE_URL`: "https://api.coinbase.com"
- `COINBASE_API_VERSION`: "v3"
- `EXECUTION_MODE`: "live"
- `TRADE_EXECUTION_ENABLED`: "true"
- `ORDER_TYPE`: "market"
- Risk management settings

## Services Using Coinbase API

### Trade Executor Real
- **Deployment**: `trade-executor-real`
- **Namespace**: `crypto-trading`
- **Environment Variables**:
  - `COINBASE_API_KEY` (from secret)
  - `COINBASE_PRIVATE_KEY` (from secret)
  - `COINBASE_BASE_URL` (from config)
  - All other config from `coinbase-api-config`

### Risk Management Service
- **Deployment**: `risk-management-service`
- **Namespace**: `crypto-trading`
- **Environment Variables**: Same as trade executor

### Trade Orchestrator LLM
- **Deployment**: `trade-orchestrator-llm`
- **Namespace**: `crypto-trading`
- **Environment Variables**: Same as trade executor

## Deployment Commands

### Apply Configuration
```bash
kubectl apply -f k8s/coinbase-api-secrets.yaml
kubectl apply -f k8s/corrected-architecture-deployments.yaml
```

### Verify Configuration
```bash
# Check secret exists
kubectl get secret coinbase-api-secrets -n crypto-trading

# Check configmap exists
kubectl get configmap coinbase-api-config -n crypto-trading

# Verify environment variables in pod
kubectl exec -it <pod-name> -n crypto-trading -- env | grep COINBASE
```

### Restart Services
```bash
kubectl rollout restart deployment/trade-executor-real -n crypto-trading
kubectl rollout restart deployment/risk-management-service -n crypto-trading
kubectl rollout restart deployment/trade-orchestrator-llm -n crypto-trading
```

## Troubleshooting

### Common Issues

1. **"Unable to load PEM file" Error**
   - **Cause**: Private key format issue
   - **Solution**: Ensure private key is properly base64 encoded in secret

2. **"Failed to create JWT token" Error**
   - **Cause**: Missing or invalid API credentials
   - **Solution**: Verify secret contains correct base64-encoded values

3. **"MalformedFraming" Error**
   - **Cause**: Private key not properly formatted
   - **Solution**: Ensure private key includes proper PEM headers/footers

### Verification Steps

1. **Check Secret Content**:
   ```bash
   kubectl get secret coinbase-api-secrets -n crypto-trading -o yaml
   ```

2. **Test API Connection**:
   ```bash
   kubectl exec -it <trade-executor-pod> -n crypto-trading -- python -c "
   import os
   print('API Key:', os.getenv('COINBASE_API_KEY'))
   print('Base URL:', os.getenv('COINBASE_BASE_URL'))
   "
   ```

3. **Test Trade Execution**:
   ```bash
   curl -X POST http://localhost:8024/execute-trade \
     -H "Content-Type: application/json" \
     -d '{"symbol": "BTC", "side": "BUY", "amount_usd": 10.0}'
   ```

## Security Notes

- API credentials are stored as Kubernetes secrets (base64 encoded)
- Secrets are only accessible within the crypto-trading namespace
- Private key is in EC format as required by Coinbase Advanced Trade API
- All API calls use JWT authentication with ES256 algorithm

## File Locations

- **API Key File**: `coinbase_api_key.json`
- **Kubernetes Secret**: `k8s/coinbase-api-secrets.yaml`
- **Deployment Config**: `k8s/corrected-architecture-deployments.yaml`
- **Documentation**: `COINBASE_API_CONFIGURATION.md`

## Last Updated
October 14, 2025 - API key configuration standardized and documented
