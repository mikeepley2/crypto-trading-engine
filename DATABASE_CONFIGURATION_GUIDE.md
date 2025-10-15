# Centralized Database Configuration Guide

## Overview

The crypto trading engine now uses a centralized database configuration system that allows you to change the database location by updating a single file. This eliminates the need to update multiple deployment files when moving the database to a different server.

## Single Point of Configuration

**File:** `k8s/database-config.yaml`

This file contains all database configuration settings and is the **ONLY** place you need to update when changing the database location.

### Key Configuration Variables

```yaml
# SINGLE POINT OF CHANGE FOR DATABASE LOCATION
DB_HOST: "192.168.230.163"  # <-- Change this IP to move database
DB_PORT: "3306"
DB_USER: "news_collector"
DB_NAME_PRICES: "crypto_prices"
DB_NAME_TRANSACTIONS: "crypto_transactions"
DB_NAME_NEWS: "crypto_news"
```

## How to Change Database Location

### Step 1: Update the Configuration

Edit `k8s/database-config.yaml` and change the `DB_HOST` value:

```yaml
# Before (old database server)
DB_HOST: "192.168.230.163"

# After (new database server)
DB_HOST: "192.168.1.100"  # Your new database server IP
```

### Step 2: Apply the Configuration

```bash
kubectl apply -f k8s/database-config.yaml
```

### Step 3: Restart Services (if needed)

The services will automatically pick up the new configuration. If you want to force a restart:

```bash
kubectl rollout restart deployment/signal-generator-working -n crypto-trading
kubectl rollout restart deployment/trade-executor-real -n crypto-trading
kubectl rollout restart deployment/risk-management-service -n crypto-trading
kubectl rollout restart deployment/ollama-llm-validation -n crypto-trading
kubectl rollout restart deployment/trade-orchestrator-llm -n crypto-trading
```

## Services Using Centralized Configuration

The following services now use the centralized database configuration:

1. **Signal Generator** (`signal-generator-working`)
2. **Trade Executor** (`trade-executor-real`)
3. **Risk Management Service** (`risk-management-service`)
4. **LLM Validation Service** (`ollama-llm-validation`)
5. **Trade Orchestrator** (`trade-orchestrator-llm`)

## Environment Variables

All services now use these standardized environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `DB_HOST` | Database server IP/hostname | `192.168.230.163` |
| `DB_PORT` | Database port | `3306` |
| `DB_USER` | Database username | `news_collector` |
| `DB_PASSWORD` | Database password | `99Rules!` |
| `DB_NAME_PRICES` | Prices database name | `crypto_prices` |
| `DB_NAME_TRANSACTIONS` | Transactions database name | `crypto_transactions` |
| `DB_NAME_NEWS` | News database name | `crypto_news` |

## Legacy Compatibility

The configuration also includes legacy variable names for backward compatibility:

- `MYSQL_HOST` (same as `DB_HOST`)
- `MYSQL_USER` (same as `DB_USER`)
- `MYSQL_PASSWORD` (same as `DB_PASSWORD`)
- `DATABASE_HOST` (same as `DB_HOST`)

## Security

Database passwords are stored in Kubernetes Secrets (`database-secrets`) and are base64 encoded:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: database-secrets
  namespace: crypto-trading
data:
  DB_PASSWORD: "OTlSdWxlcyE="  # "99Rules!" in base64
```

## Configuration Files Updated

The following files have been updated to use the centralized configuration:

1. `k8s/database-config.yaml` - **NEW** - Centralized configuration
2. `k8s/corrected-architecture-deployments.yaml` - Updated all services
3. `k8s/trade-executor-real.yaml` - Updated database connection
4. `k8s/risk-management-service.yaml` - Updated database connection
5. `k8s/crypto-trading-config.yaml` - Updated to reference centralized config

## Verification

To verify that services are using the correct database configuration:

```bash
# Check if the ConfigMap is applied
kubectl get configmap database-config -n crypto-trading

# Check environment variables in a pod
kubectl exec -it deployment/signal-generator-working -n crypto-trading -- env | grep DB_

# Check database connectivity
kubectl exec -it deployment/signal-generator-working -n crypto-trading -- python -c "
import os
print(f'DB_HOST: {os.getenv(\"DB_HOST\")}')
print(f'DB_USER: {os.getenv(\"DB_USER\")}')
print(f'DB_NAME_PRICES: {os.getenv(\"DB_NAME_PRICES\")}')
"
```

## Troubleshooting

### Service Not Connecting to Database

1. Check if the ConfigMap is applied:
   ```bash
   kubectl get configmap database-config -n crypto-trading
   ```

2. Check if the Secret is applied:
   ```bash
   kubectl get secret database-secrets -n crypto-trading
   ```

3. Check service logs:
   ```bash
   kubectl logs deployment/signal-generator-working -n crypto-trading
   ```

4. Verify environment variables:
   ```bash
   kubectl exec -it deployment/signal-generator-working -n crypto-trading -- env | grep DB_
   ```

### Database Connection Errors

1. Verify database server is accessible from Kubernetes cluster
2. Check firewall rules
3. Verify database credentials
4. Check database server logs

## Migration from Old Configuration

If you have existing services with hardcoded database configurations, they will continue to work but should be updated to use the centralized configuration for consistency.

## Best Practices

1. **Always use the centralized configuration** - Don't hardcode database settings in individual services
2. **Test configuration changes** - Verify connectivity after changing database location
3. **Use secrets for passwords** - Never put passwords in ConfigMaps
4. **Document changes** - Keep track of database location changes
5. **Monitor connectivity** - Set up monitoring for database connectivity

## Future Enhancements

- Database connection pooling configuration
- Multiple database support (read replicas)
- Database failover configuration
- Connection health checks
- Automatic retry logic
