# Crypto Trading System - Final Consolidated Documentation

## 🚨 CRITICAL INSTRUCTIONS
**NEVER CREATE NEW SERVICES UNLESS EXPLICITLY REQUESTED**
- Always work with existing services only
- Only modify existing services when needed
- Do not create duplicate or "simple" versions of services
- Consolidate functionality into existing services instead of creating new ones

**NEVER USE FALLBACK MODES FOR LLM SERVICES**
- All LLM validation must be fully functional with real AI models
- If LLM services fail, the system should fail rather than use fallback rules
- No fallback operations are acceptable for AI/ML services
- LLM validation must use actual Ollama models, not rule-based alternatives

## System Overview

The crypto trading system is a complete end-to-end pipeline that generates ML-backed trading signals and executes trades automatically. The system consists of 4 core services that work together seamlessly.

## Architecture

```
Market Data → Signal Generator → Signal Bridge → Trade Orchestrator → Trade Executor → Coinbase API
```

## Core Services

### 1. Signal Generator (`signal-generator`)
- **Port**: 8025
- **Purpose**: ML-backed signal generation using XGBoost model
- **Features**:
  - Loads balanced realistic ML model (51 features)
  - Generates BUY/SELL/HOLD signals based on confidence thresholds
  - NO FALLBACKS policy - service fails if model cannot load
  - Processes top 20 symbols every 5 minutes
  - Filters unsupported assets
  - Saves signals to `trading_signals` table

**Key Configuration**:
- Model: `balanced_realistic_model_20251005_155755.joblib`
- Features: 51 technical indicators
- Confidence thresholds: BUY > 0.5, SELL > 0.6
- Database: `crypto_prices.trading_signals`

### 2. Signal Bridge (`signal-bridge`)
- **Port**: 8022
- **Purpose**: Converts trading signals to trade recommendations
- **Features**:
  - Processes signals from last hour with confidence > 0.6
  - Creates recommendations in `trade_recommendations` table
  - Runs every 30 seconds
  - Prevents duplicate recommendations

**Key Configuration**:
- Database: `crypto_prices.trade_recommendations`
- Processing interval: 30 seconds
- Confidence threshold: 0.6
- Default amount: $100 USD

### 3. Trade Executor (`trade-executor`)
- **Port**: 8024
- **Purpose**: Executes trade recommendations via Coinbase API
- **Features**:
  - JWT authentication with Coinbase Advanced Trade API
  - Processes recommendations via `/process_recommendation/{id}` endpoint
  - Updates recommendation status to EXECUTED
  - Health checks include API connectivity validation

**Key Configuration**:
- API: Coinbase Advanced Trade (`https://api.coinbase.com/api/v3/brokerage`)
- Authentication: JWT with ES256 algorithm
- Database: `crypto_prices.trade_recommendations`

### 4. Trade Orchestrator (`trade-orchestrator`)
- **Port**: 8023
- **Purpose**: Orchestrates the complete trading flow
- **Features**:
  - Monitors pending recommendations
  - Calls trade executor to process recommendations
  - Limits to 3 trades per cycle
  - Prioritizes high-confidence recommendations
  - Runs every 30 seconds

**Key Configuration**:
- Processing interval: 30 seconds
- Max trades per cycle: 3
- Time window: 2 hours
- Service URLs: Uses Kubernetes DNS

## Database Schema

### trading_signals Table
```sql
CREATE TABLE trading_signals (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    symbol VARCHAR(50) NOT NULL,
    price DECIMAL(15,8) DEFAULT 0.0,
    signal_type ENUM('BUY', 'SELL', 'HOLD') NOT NULL,
    model VARCHAR(100),
    confidence DECIMAL(6,4) NOT NULL,
    threshold DECIMAL(6,4) DEFAULT 0.8,
    regime VARCHAR(50) DEFAULT 'bull',
    model_version VARCHAR(100),
    features_used INT DEFAULT 79,
    xgboost_confidence DECIMAL(6,4),
    data_source VARCHAR(100) DEFAULT 'database',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_mock TINYINT(1) DEFAULT 0,
    processed TINYINT(1) DEFAULT 0,
    prediction DECIMAL(6,4)
);
```

### trade_recommendations Table
```sql
CREATE TABLE trade_recommendations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    signal_id INT,
    symbol VARCHAR(50) NOT NULL,
    signal_type ENUM('BUY', 'SELL', 'HOLD', 'STRONG_BUY', 'STRONG_SELL') NOT NULL,
    amount_usd DECIMAL(15,8) NOT NULL,
    confidence DECIMAL(6,4) NOT NULL,
    reasoning TEXT,
    execution_status ENUM('PENDING', 'EXECUTED', 'FAILED', 'CANCELLED') DEFAULT 'PENDING',
    entry_price DECIMAL(15,8),
    stop_loss DECIMAL(15,8),
    take_profit DECIMAL(15,8),
    position_size_percent DECIMAL(5,2),
    amount_crypto DECIMAL(20,8),
    is_mock TINYINT(1) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    executed_at TIMESTAMP NULL,
    INDEX idx_signal_id (signal_id),
    INDEX idx_symbol (symbol),
    INDEX idx_status (execution_status),
    INDEX idx_created_at (created_at)
);
```

## Kubernetes Deployment

### Services and Ports
- `signal-generator`: 8025 (ClusterIP)
- `signal-bridge`: 8022 (ClusterIP)
- `trade-executor`: 8024 (ClusterIP)
- `trade-orchestrator`: 8023 (ClusterIP)

### DNS Resolution
All services use Kubernetes DNS for inter-service communication:
- `signal-bridge:8022`
- `trade-executor:8024`
- `trade-orchestrator:8023`

### Configuration
- **ConfigMaps**: `trade-exec-coinbase-config`, `trade-proc-orchestrator-config`
- **Secrets**: `trade-exec-coinbase-secrets`
- **Database**: MySQL at `172.22.32.1:3306`

## Deployment Commands

### Deploy Complete System
```bash
kubectl apply -f k8s/crypto-trading-system-final.yaml
```

### Check System Status
```bash
kubectl get pods -n crypto-trading
kubectl get services -n crypto-trading
```

### View Logs
```bash
kubectl logs -f deployment/signal-generator -n crypto-trading
kubectl logs -f deployment/signal-bridge -n crypto-trading
kubectl logs -f deployment/trade-executor -n crypto-trading
kubectl logs -f deployment/trade-orchestrator -n crypto-trading
```

### Health Checks
```bash
kubectl exec -it deployment/signal-generator -n crypto-trading -- curl http://localhost:8025/health
kubectl exec -it deployment/signal-bridge -n crypto-trading -- curl http://localhost:8022/health
kubectl exec -it deployment/trade-executor -n crypto-trading -- curl http://localhost:8024/health
kubectl exec -it deployment/trade-orchestrator -n crypto-trading -- curl http://localhost:8023/health
```

## Testing the Complete Flow

### 1. Check Signal Generation
```bash
kubectl logs deployment/signal-generator -n crypto-trading | grep "Generated.*signal"
```

### 2. Check Signal Processing
```bash
kubectl logs deployment/signal-bridge -n crypto-trading | grep "Created recommendation"
```

### 3. Check Trade Execution
```bash
kubectl logs deployment/trade-orchestrator -n crypto-trading | grep "Successfully executed"
```

### 4. Verify Database
```sql
-- Check recent signals
SELECT * FROM trading_signals ORDER BY timestamp DESC LIMIT 10;

-- Check recent recommendations
SELECT * FROM trade_recommendations ORDER BY created_at DESC LIMIT 10;

-- Check executed trades
SELECT COUNT(*) as executed_trades FROM trade_recommendations WHERE execution_status = 'EXECUTED';
```

## Troubleshooting

### Common Issues
1. **Model Loading Failures**: Ensure model file is copied to `/app/models/`
2. **Database Connection Issues**: Check ConfigMap and Secret configurations
3. **API Authentication Failures**: Verify JWT credentials in secrets
4. **DNS Resolution Issues**: Use Kubernetes service names, not IPs

### Service Dependencies
- Signal Generator → Database (MySQL)
- Signal Bridge → Database (MySQL)
- Trade Executor → Database (MySQL) + Coinbase API
- Trade Orchestrator → Signal Bridge + Trade Executor

## Performance Metrics

### Expected Performance
- **Signal Generation**: Every 5 minutes
- **Signal Processing**: Every 30 seconds
- **Trade Execution**: Every 30 seconds
- **Max Trades per Cycle**: 3
- **Confidence Threshold**: 0.6 for recommendations

### Monitoring
- Health endpoints on all services
- Database logging for all operations
- API connectivity validation
- Model loading validation

## Security Considerations

- JWT authentication for Coinbase API
- Database credentials in Kubernetes secrets
- No hardcoded credentials in code
- Service-to-service communication via ClusterIP

## Maintenance

### Model Updates
1. Copy new model file to `/app/models/`
2. Restart signal generator deployment
3. Verify model loading in logs

### Configuration Updates
1. Update ConfigMaps/Secrets
2. Restart affected deployments
3. Verify service health

### Database Maintenance
- Monitor `trading_signals` table size
- Archive old signals periodically
- Monitor `trade_recommendations` execution status

---

**Last Updated**: October 7, 2025
**System Version**: Final Consolidated v1.0
**Status**: Production Ready
