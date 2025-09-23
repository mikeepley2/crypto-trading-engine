# Trading Engine API Documentation

## Overview
The crypto trading engine exposes REST APIs for monitoring, configuration, and control of the live trading system.

## Service Endpoints

### Signal Generator Service (Port 8025)

#### Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-09-23T12:00:00Z",
  "ml_models_loaded": 15,
  "features_available": 83,
  "last_prediction": "2025-09-23T11:58:00Z"
}
```

#### Get Latest Signals
```
GET /signals/latest
```

**Response:**
```json
{
  "signals": [
    {
      "symbol": "BTC",
      "signal": "BUY",
      "confidence": 0.745,
      "prediction_strength": 0.68,
      "timestamp": "2025-09-23T11:58:00Z",
      "features": {
        "technical": {"rsi": 34.5, "macd": 0.08},
        "sentiment": {"score": 0.72, "trend": "positive"},
        "ml_features": 83
      }
    }
  ]
}
```

#### Generate Signal for Specific Asset
```
POST /signals/generate
Content-Type: application/json

{
  "symbol": "ETH",
  "features": {
    "current_price": 2650.30,
    "volume_24h": 15000000,
    "rsi": 42.1
  }
}
```

#### Signal Performance Metrics
```
GET /signals/performance
```

**Response:**
```json
{
  "accuracy_7d": 0.673,
  "accuracy_30d": 0.665,
  "total_signals": 1247,
  "correct_predictions": 830,
  "avg_confidence": 0.728,
  "performance_by_asset": {
    "BTC": {"accuracy": 0.71, "signals": 342},
    "ETH": {"accuracy": 0.68, "signals": 298}
  }
}
```

### Trade Execution Service (Port 8024)

#### Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "trading_mode": "live",
  "connected_exchange": "coinbase",
  "api_status": "connected",
  "last_trade": "2025-09-23T11:45:00Z"
}
```

#### Portfolio Status
```
GET /portfolio
```

**Response:**
```json
{
  "total_value": 2571.86,
  "cash_balance": 514.32,
  "invested_value": 2057.54,
  "total_pnl": 2505.74,
  "total_pnl_percent": 3784.2,
  "positions": [
    {
      "symbol": "BTC",
      "quantity": 0.0234,
      "avg_price": 45230.50,
      "current_price": 67340.20,
      "market_value": 1575.85,
      "pnl": 517.23,
      "pnl_percent": 48.9
    }
  ]
}
```

#### Recent Trades
```
GET /trades/recent?limit=10
```

**Response:**
```json
{
  "trades": [
    {
      "id": "trade_12345",
      "symbol": "SOL",
      "side": "buy",
      "quantity": 12.5,
      "price": 142.30,
      "total": 1778.75,
      "timestamp": "2025-09-23T11:45:00Z",
      "signal_confidence": 0.78,
      "status": "filled"
    }
  ]
}
```

#### Execute Trade
```
POST /trades/execute
Content-Type: application/json

{
  "symbol": "ADA",
  "side": "buy",
  "amount_usd": 500.00,
  "signal_confidence": 0.72,
  "max_slippage": 0.005
}
```

**Response:**
```json
{
  "trade_id": "trade_12346",
  "status": "submitted",
  "order_id": "coinbase_order_789",
  "estimated_quantity": 1250.00,
  "estimated_price": 0.40
}
```

#### Emergency Liquidation
```
POST /emergency/liquidate-all
Content-Type: application/json

{
  "confirm": true,
  "reason": "emergency_stop"
}
```

### Signal Bridge Service (Port 8022)

#### Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "signals_processed": 1247,
  "trades_executed": 89,
  "success_rate": 0.071,
  "last_signal_processed": "2025-09-23T11:58:00Z"
}
```

#### Signal Processing Status
```
GET /bridge/status
```

**Response:**
```json
{
  "queue_depth": 3,
  "processing_rate": 12.5,
  "avg_processing_time": 1.2,
  "failed_signals": 0,
  "pending_trades": 1
}
```

#### Configure Signal Thresholds
```
POST /bridge/configure
Content-Type: application/json

{
  "signal_threshold": 0.65,
  "max_position_size": 0.05,
  "risk_limit": 0.15
}
```

### Portfolio Rebalancer Service (Port 8047)

#### Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "last_rebalance": "2025-09-23T11:30:00Z",
  "rebalance_frequency": "15min",
  "positions_managed": 7
}
```

#### Risk Metrics
```
GET /risk/metrics
```

**Response:**
```json
{
  "portfolio_risk": {
    "var_1d": -0.08,
    "var_7d": -0.15,
    "sharpe_ratio": 2.14,
    "max_drawdown": 0.082,
    "correlation_risk": 0.34
  },
  "position_risks": [
    {
      "symbol": "BTC",
      "position_risk": 0.045,
      "correlation_factor": 0.67,
      "volatility": 0.65
    }
  ]
}
```

#### Trigger Rebalance
```
POST /rebalance/trigger
Content-Type: application/json

{
  "reason": "manual_trigger",
  "force": false
}
```

## Authentication

### API Key Authentication
Most endpoints require API key authentication via header:
```
Authorization: Bearer <api-key>
```

### Internal Service Communication
Services communicate internally using Kubernetes service discovery and mutual TLS.

## Rate Limiting

### External API Limits
- **Coinbase**: 10 requests/second per API key
- **Internal Services**: 100 requests/second per service

### Rate Limit Headers
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1695456000
```

## Error Handling

### Standard Error Response
```json
{
  "error": {
    "code": "INSUFFICIENT_FUNDS",
    "message": "Insufficient balance for trade execution",
    "details": {
      "required": 1000.00,
      "available": 850.32
    },
    "timestamp": "2025-09-23T12:00:00Z"
  }
}
```

### Error Codes
- `INSUFFICIENT_FUNDS`: Not enough balance for trade
- `INVALID_SIGNAL`: Signal confidence below threshold
- `API_ERROR`: External API communication error
- `RISK_LIMIT_EXCEEDED`: Trade would exceed risk limits
- `MARKET_CLOSED`: Trading not available
- `SYSTEM_MAINTENANCE`: System in maintenance mode

## WebSocket Streaming

### Real-time Signal Stream
```
ws://localhost:8025/ws/signals
```

**Message Format:**
```json
{
  "type": "signal",
  "data": {
    "symbol": "BTC",
    "signal": "BUY",
    "confidence": 0.78,
    "timestamp": "2025-09-23T12:00:00Z"
  }
}
```

### Trade Execution Stream
```
ws://localhost:8024/ws/trades
```

**Message Format:**
```json
{
  "type": "trade_update",
  "data": {
    "trade_id": "trade_12345",
    "status": "filled",
    "fill_price": 67340.20,
    "timestamp": "2025-09-23T12:00:00Z"
  }
}
```

## Monitoring Integration

### Prometheus Metrics
```
GET /metrics
```

**Available Metrics:**
- `trading_signals_total`: Total signals generated
- `trading_accuracy_ratio`: Signal accuracy ratio
- `portfolio_value_usd`: Current portfolio value
- `trade_execution_duration_seconds`: Trade execution time
- `api_requests_total`: Total API requests
- `api_errors_total`: Total API errors

### Health Check Integration
All services expose standardized health endpoints for monitoring systems:
- **Kubernetes**: Readiness and liveness probes
- **Prometheus**: Metrics scraping
- **Grafana**: Dashboard visualization
- **AlertManager**: Error alerting

## Configuration API

### Get Current Configuration
```
GET /config
```

### Update Configuration
```
PUT /config
Content-Type: application/json

{
  "trading": {
    "mode": "live",
    "max_position_size": 0.05,
    "signal_threshold": 0.65
  },
  "risk": {
    "max_drawdown": 0.15,
    "stop_loss": 0.05
  }
}
```

### Configuration Validation
```
POST /config/validate
Content-Type: application/json

{
  "trading": {
    "max_position_size": 0.10
  }
}
```

**Response:**
```json
{
  "valid": false,
  "errors": [
    {
      "field": "trading.max_position_size",
      "error": "Value exceeds maximum allowed (0.05)"
    }
  ]
}
```

## Testing Endpoints

### Mock Trading Mode
```
POST /trading/mode
Content-Type: application/json

{
  "mode": "mock",
  "initial_balance": 10000.00
}
```

### Simulation Controls
```
POST /simulation/reset
POST /simulation/fast-forward
POST /simulation/add-scenario
```

### Test Signal Generation
```
POST /test/generate-signal
Content-Type: application/json

{
  "symbol": "BTC",
  "scenario": "bull_market",
  "confidence": 0.75
}
```