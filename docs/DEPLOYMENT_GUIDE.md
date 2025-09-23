# Trading Engine Deployment Guide

## Overview
This guide covers the complete deployment of the crypto trading engine in a Kubernetes environment with live trading capabilities.

## Prerequisites

### System Requirements
- **Kubernetes Cluster**: Kind, Minikube, or cloud-based cluster
- **Database**: MySQL (Windows-based recommended for development)
- **Resources**: Minimum 4 CPU cores, 8GB RAM for production trading
- **Storage**: 50GB for logs and model data

### Database Setup
```bash
# Verify MySQL connectivity from containers
mysql -h host.docker.internal -u news_collector -p99Rules! -e "SHOW DATABASES;"

# Required databases
crypto_transactions  # Trading data, portfolio positions, trades
crypto_prices       # ML features and historical data
```

### API Credentials
- **Coinbase Advanced Trade API**: API key and private key in EC format
- **Optional**: Binance.US, KuCoin API credentials for multi-platform trading

## Deployment Steps

### 1. Prepare Kubernetes Environment
```bash
# Create namespace
kubectl create namespace crypto-trading

# Create secrets for API credentials
kubectl create secret generic coinbase-api \
  --from-literal=api-key=<your-api-key> \
  --from-literal=private-key=<your-private-key> \
  -n crypto-trading
```

### 2. Deploy Configuration
```bash
# Apply configuration maps
kubectl apply -f k8s/config/ -n crypto-trading

# Verify configuration
kubectl get configmaps -n crypto-trading
```

### 3. Deploy Core Services
```bash
# Deploy signal generation services
kubectl apply -f k8s/enhanced-signal-generator.yaml

# Deploy trading execution engine
kubectl apply -f k8s/trade-execution-engine.yaml

# Deploy signal bridge
kubectl apply -f k8s/microservices-signal-bridge.yaml

# Deploy portfolio manager
kubectl apply -f k8s/portfolio-rebalancer.yaml
```

### 4. Verify Deployment
```bash
# Check pod status
kubectl get pods -n crypto-trading

# Check service endpoints
kubectl get services -n crypto-trading

# Test health endpoints
curl http://localhost:8025/health  # Signal generator
curl http://localhost:8024/health  # Trade execution
curl http://localhost:8022/health  # Signal bridge
```

## Configuration Management

### Trading Parameters
Edit the trading configuration:
```bash
kubectl edit configmap trading-config -n crypto-trading
```

Key parameters:
- `TRADING_MODE`: "live" or "mock"
- `MAX_POSITION_SIZE`: Maximum position as percentage of portfolio
- `SIGNAL_THRESHOLD`: Minimum ML confidence for trade execution
- `RISK_LIMIT`: Maximum portfolio drawdown before emergency stop

### Database Configuration
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: database-config
data:
  MYSQL_HOST: "host.docker.internal"
  MYSQL_USER: "news_collector"
  MYSQL_DATABASE_TRADING: "crypto_transactions"
  MYSQL_DATABASE_ML: "crypto_prices"
```

## Monitoring and Health Checks

### Service Health
```bash
# Individual service health
kubectl exec -n crypto-trading deployment/signal-generator -- curl localhost:8025/health
kubectl exec -n crypto-trading deployment/trade-execution -- curl localhost:8024/health

# Comprehensive health check
./scripts/monitoring/comprehensive_health_check.sh
```

### Trading Performance
```bash
# Portfolio status
curl http://localhost:8024/portfolio | jq

# Recent trades
curl http://localhost:8024/trades/recent | jq

# Signal performance metrics
curl http://localhost:8025/signals/performance | jq
```

### Log Monitoring
```bash
# Stream logs from all trading services
kubectl logs -n crypto-trading -l app=trading-engine -f

# Specific service logs
kubectl logs -n crypto-trading deployment/signal-generator -f
kubectl logs -n crypto-trading deployment/trade-execution -f
```

## Scaling and Performance

### Horizontal Scaling
```bash
# Scale signal generators for high throughput
kubectl scale deployment/signal-generator --replicas=3 -n crypto-trading

# Scale portfolio analyzers
kubectl scale deployment/portfolio-rebalancer --replicas=2 -n crypto-trading
```

### Resource Limits
```yaml
resources:
  requests:
    cpu: "500m"
    memory: "1Gi"
  limits:
    cpu: "2000m"
    memory: "4Gi"
```

## Backup and Recovery

### Database Backup
```bash
# Backup trading data
mysqldump -h host.docker.internal -u news_collector -p99Rules! crypto_transactions > trading_backup.sql

# Backup ML features
mysqldump -h host.docker.internal -u news_collector -p99Rules! crypto_prices ml_features_materialized > ml_backup.sql
```

### Configuration Backup
```bash
# Export all configurations
kubectl get configmaps -n crypto-trading -o yaml > configs_backup.yaml
kubectl get secrets -n crypto-trading -o yaml > secrets_backup.yaml
```

### Disaster Recovery
```bash
# Restore from backup
mysql -h host.docker.internal -u news_collector -p99Rules! crypto_transactions < trading_backup.sql

# Redeploy services
kubectl apply -f k8s/ -n crypto-trading
```

## Security Best Practices

### API Key Management
- Store all API keys in Kubernetes secrets
- Rotate API keys monthly
- Use separate keys for different environments (dev/prod)
- Monitor API key usage and rate limits

### Network Security
- Use TLS for all external connections
- Implement network policies for pod-to-pod communication
- Restrict database access to specific service accounts
- Enable audit logging for all trading operations

### Access Control
```bash
# Create service account for trading services
kubectl create serviceaccount trading-service -n crypto-trading

# Apply RBAC permissions
kubectl apply -f k8s/rbac.yaml -n crypto-trading
```

## Troubleshooting

### Common Issues

#### Signal Generation Not Working
```bash
# Check ML model availability
kubectl exec -n crypto-trading deployment/signal-generator -- ls -la /models/

# Verify database connectivity
kubectl exec -n crypto-trading deployment/signal-generator -- \
  mysql -h host.docker.internal -u news_collector -p99Rules! -e "SELECT COUNT(*) FROM crypto_prices.ml_features_materialized;"
```

#### Trade Execution Failures
```bash
# Check API connectivity
kubectl logs -n crypto-trading deployment/trade-execution | grep -i "api\|error"

# Verify API key configuration
kubectl get secret coinbase-api -n crypto-trading -o yaml
```

#### Portfolio Sync Issues
```bash
# Check portfolio table
mysql -h host.docker.internal -u news_collector -p99Rules! crypto_transactions \
  -e "SELECT * FROM portfolio_positions ORDER BY last_updated DESC LIMIT 10;"

# Restart portfolio services
kubectl rollout restart deployment/portfolio-rebalancer -n crypto-trading
```

### Performance Issues
```bash
# Check resource usage
kubectl top pods -n crypto-trading

# Monitor database connections
mysql -h host.docker.internal -u news_collector -p99Rules! \
  -e "SHOW PROCESSLIST;"

# Check service response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8025/health
```

## Updates and Maintenance

### Rolling Updates
```bash
# Update signal generator
kubectl set image deployment/signal-generator signal-generator=new-image:tag -n crypto-trading

# Monitor rollout
kubectl rollout status deployment/signal-generator -n crypto-trading

# Rollback if needed
kubectl rollout undo deployment/signal-generator -n crypto-trading
```

### Maintenance Windows
```bash
# Scale down for maintenance
kubectl scale deployment --all --replicas=0 -n crypto-trading

# Perform maintenance tasks
./scripts/maintenance/update_ml_models.sh
./scripts/maintenance/optimize_database.sh

# Scale back up
kubectl scale deployment --all --replicas=1 -n crypto-trading
```

### Model Updates
```bash
# Update ML models
kubectl create configmap ml-models --from-file=models/ -n crypto-trading

# Restart services to load new models
kubectl rollout restart deployment/signal-generator -n crypto-trading
```

## Emergency Procedures

### Emergency Trading Stop
```bash
# Immediate stop all trading
kubectl scale deployment/trade-execution --replicas=0 -n crypto-trading

# Optional: Liquidate all positions
curl -X POST http://localhost:8024/emergency/liquidate-all \
  -H "Content-Type: application/json" \
  -d '{"confirm": true}'
```

### System Recovery
```bash
# Check system status
./scripts/monitoring/system_status.sh

# Restart all services
kubectl rollout restart deployment -n crypto-trading

# Verify recovery
./scripts/monitoring/comprehensive_health_check.sh
```

### Contact Information
- **System Administrator**: [admin contact]
- **Emergency Hotline**: [emergency contact]
- **Escalation Procedure**: [escalation steps]