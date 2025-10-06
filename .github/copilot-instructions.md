# Crypto Trading Engine - AI Agent Instructions

This is a **production-ready** AI-powered cryptocurrency trading engine that's part of a larger ecosystem. Understanding the architecture and service interactions is critical for effective development.

## System Architecture Overview

### Core Components
- **Signal Generation**: ML-powered XGBoost models (`enhanced_signal_generator.py`) on port 8025
- **Trade Execution**: Live trading engine (`automated_live_trader.py`) with Coinbase/Binance integration on port 8024
- **Recommendation Engine**: Signal-to-trade conversion service on port 8022
- **Portfolio Management**: Position tracking and rebalancing on port 8026
- **Risk Management**: Exposure limits and position sizing on port 8025

### Ecosystem Integration
This project is a **specialized trading execution node** within the larger distributed AI crypto ecosystem:

#### **Multi-Project Architecture**
- **Primary Hub**: `aitest` project (`e:\git\aitest`) - Main orchestration and shared resources
- **Data Collection**: `crypto-data-collection` - Market data ingestion and processing
- **Monitoring**: `crypto-monitoring` - System health, alerts, and performance metrics
- **Trading Engine**: `crypto-trading-engine` (this project) - Live trade execution and ML signals

#### **Shared Infrastructure**
- **Database**: Shared MySQL instance across all nodes (`192.168.230.163`)
- **Redis Cache**: Cross-node data sharing and pub/sub messaging
- **ML Models**: Centralized model storage and versioning in aitest project
- **Environment Configuration**: Production credentials stored in `e:\git\aitest\.env.live`
- **Kubernetes Deployment**: All nodes deployed as separate services in shared K8s cluster

#### **Inter-Node Communication**
- **REST APIs**: Service-to-service communication via FastAPI endpoints
- **Shared Database**: Common data layer for signals, trades, and monitoring
- **Message Queues**: Redis-based pub/sub for real-time data streams
- **Health Monitoring**: Cross-node health checks and failure detection

## Multi-Project Development Environment

### Project Locations
- **aitest** (Main Hub): `e:\git\aitest`
  - Production environment: `e:\git\aitest\.env.live`
  - ML models: `e:\git\aitest\ml_models_production\`
  - Shared services: `e:\git\aitest\backend\services\`

- **crypto-data-collection**: `e:\git\crypto-data-collection`
  - Market data ingestion and processing
  - Real-time price feeds and sentiment analysis

- **crypto-monitoring**: `e:\git\crypto-monitoring` 
  - Grafana dashboards and Prometheus metrics
  - System health monitoring and alerting

- **crypto-trading-engine** (Current): `e:\git\crypto-trading-engine`
  - Live trade execution and ML signal generation
  - Portfolio management and risk controls

### Development Workflow
When working across the ecosystem:

1. **Environment Setup**: Always source `e:\git\aitest\.env.live` for production credentials
2. **Database Connectivity**: Use shared MySQL at `192.168.230.163` (K8s) or `host.docker.internal` (Docker)
3. **Service Dependencies**: Verify cross-node service health before debugging issues
4. **Model Updates**: ML models are centrally managed in the aitest project
5. **Configuration Changes**: Update shared configs in aitest, then sync to other nodes

### Cross-Node Dependencies
- **aitest → crypto-trading-engine**: ML models, environment variables, LLM services
- **crypto-data-collection → crypto-trading-engine**: Market data, price feeds, technical indicators  
- **crypto-monitoring → all nodes**: Health checks, performance metrics, alerting
- **crypto-trading-engine → aitest**: Trade execution results, portfolio updates

## Database Architecture

### Primary Databases
- **MySQL Host**: `host.docker.internal` (Windows) or `192.168.230.163` (K8s)
- **Credentials**: `news_collector` / `99Rules!`
- **Trading DB**: `crypto_transactions` (trade records, recommendations)
- **ML DB**: `crypto_prices` (historical data, signals)

### Key Tables
- `trade_recommendations`: Pending/executed trade signals
- `portfolio_positions`: Current holdings and balances
- `ml_signals`: XGBoost model predictions
- `crypto_prices`: Historical OHLCV data

## Service Port Allocation

### Active Services
- **8022**: Trade Recommendation Engine (signal processing)
- **8024**: Trade Execution Engine (live trading)
- **8025**: Enhanced Signal Generator (ML models)
- **8026**: Portfolio Manager (position tracking)
- **8028**: Signals Service (signal aggregation)

### Development Patterns
- All services use FastAPI with `/health` endpoints
- Prometheus metrics on `/metrics` endpoints
- Unified API key auth: `X-TRADING-API-KEY`
- Docker containers with K8s deployment manifests

## Critical File Locations

### Core Services
- `enhanced_signal_generator.py`: Main ML signal generation
- `automated_live_trader.py`: Live trading controller
- `services/engines/`: Trade execution engines
- `services/recommendations/`: Signal-to-trade conversion
- `services/portfolio/`: Portfolio management services

### Configuration & Deployment
- `k8s/`: Kubernetes deployment manifests
- `docker-compose.trading-complete.yml`: Local development stack
- `services/shared/`: Common schemas and utilities
- `requirements.txt`: Python dependencies

## Development Workflows

### Local Development
```bash
# Start services locally
docker-compose -f docker-compose.trading-complete.yml up

# Check service health
curl http://localhost:8025/health  # Signal generator
curl http://localhost:8024/health  # Trade execution
curl http://localhost:8022/health  # Recommendations
```

### Kubernetes Deployment
```bash
# Deploy to K8s
kubectl apply -f k8s/
kubectl get pods -n crypto-trading

# Check service logs
kubectl logs -f deployment/enhanced-signal-generator -n crypto-trading
```

## Trading Platform Integration

### Multi-Platform Support
- **Coinbase Advanced Trade**: JWT with EC/HMAC signing (`coinbase_api.py`)
- **Binance.US**: HMAC SHA256 authentication
- **KuCoin**: Signature-based auth with passphrase

### Configuration-Driven Selection
Platform switching via environment variables and configuration files.

## Machine Learning Pipeline

### Signal Generation Flow
1. **Data Collection**: Historical price data from multiple sources
2. **Feature Engineering**: 83/86 ML features (96.5% coverage)
3. **XGBoost Models**: Individual models per cryptocurrency
4. **Signal Processing**: Convert ML predictions to trade recommendations
5. **Execution**: Kelly criterion position sizing with confidence weighting

### Model Files
- `full_dataset_gpu_xgboost_model_20250827_130225.joblib`: Main XGBoost model (latest GPU-trained model)
- ML models loaded at service startup, critical for system operation

## Risk Management Architecture

### Position Limits
- Max 5% position size per asset
- Max 80% portfolio exposure (20% cash reserve)
- Daily trade count limits
- Symbol exposure monitoring

### Risk Service Integration
Risk evaluation happens before trade execution via dedicated risk service.

## Common Development Patterns

### Service Communication
- REST API between services
- MySQL for persistent state
- Redis for caching and pub/sub
- Health checks and metrics for monitoring

### Error Handling
- Critical alerts logged to `/tmp/` for external monitoring
- Database connection pooling with retry logic
- Graceful degradation when external services unavailable

### Testing Patterns
- Mock trading mode for development: `TRADING_MODE=mock`
- Integration tests in `services/tests/`
- Health check endpoints for service validation

## Deployment Considerations

### Environment Variables
- `DB_HOST`, `DB_USER`, `DB_PASSWORD`: Database connection
- `TRADING_MODE`: `mock` or `live` for trade execution
- `REDIS_HOST`, `REDIS_PORT`: Redis configuration

### Resource Requirements
- Signal generator: 1-2GB RAM, 0.5-1 CPU
- Trade execution: 512MB RAM, 0.5 CPU
- Database: Persistent storage for trade history

When working with this codebase, always verify service health before making changes and understand the dependencies between services. The system is **actively trading with real money**, so changes must be thoroughly tested in mock mode first.

## Multi-Project Reference Guide

### **Complete Project Structure**
```
e:\git\
├── aitest\                          # 🧠 ORCHESTRATION HUB
│   ├── .env.live                    # 🔑 PRODUCTION CREDENTIALS
│   ├── backend\services\            # Shared services and LLM integration
│   ├── ml_models_production\        # 🤖 Centralized ML models
│   └── k8s\                        # Kubernetes manifests
│
├── crypto-data-collection\          # 📊 DATA INGESTION NODE
│   ├── collectors\                  # Price, news, sentiment collectors
│   ├── processors\                  # Data normalization and enrichment
│   └── streams\                     # Real-time data streaming
│
├── crypto-monitoring\               # 📈 MONITORING NODE
│   ├── grafana\                     # Dashboard configurations
│   ├── prometheus\                  # Metrics collection
│   └── alerts\                      # Alerting rules and notifications
│
└── crypto-trading-engine\           # ⚡ EXECUTION NODE (current)
    ├── services\                    # Trading services
    ├── .env.live                    # 🔗 Links to aitest/.env.live
    └── k8s\                         # Trading service manifests
```

### **Inter-Project Communication Patterns**
1. **Configuration**: All projects reference `aitest/.env.live` for credentials
2. **Database**: Shared MySQL instance with cross-project schemas
3. **Models**: ML models stored in `aitest/ml_models_production/`
4. **Monitoring**: All projects report metrics to crypto-monitoring
5. **Data Flow**: crypto-data-collection → aitest → crypto-trading-engine

### **Development Best Practices**
- **Environment**: Always work with shared environment from aitest project
- **Testing**: Use mock mode before deploying changes to live trading
- **Dependencies**: Check cross-node health before debugging individual services
- **Updates**: Coordinate ML model updates across all projects
- **Monitoring**: Use crypto-monitoring dashboards for system health visibility