# Signal Generation Microservices Architecture

## Overview

The signal generation system has been decomposed from a monolithic 2,521-line service into 7 specialized microservices, providing better scalability, maintainability, and reliability.

## Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client API    │───▶│   Orchestrator  │───▶│    Database     │
│                 │    │    (Port 8025)  │    │  trading_signals│
└─────────────────┘    └─────────┬───────┘    └─────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌───────────┐ ┌───────────┐ ┌───────────┐
              │ML Engine  │ │Feature    │ │Market     │
              │Port 8051  │ │Engine     │ │Context    │
              │           │ │Port 8052  │ │Port 8053  │
              └───────────┘ └───────────┘ └───────────┘
                    ▼            ▼            ▼
              ┌───────────┐ ┌───────────┐ ┌───────────┐
              │Portfolio  │ │Risk Mgmt  │ │Analytics  │
              │Port 8054  │ │Port 8055  │ │Port 8056  │
              └───────────┘ └───────────┘ └───────────┘
```

## Services Detail

### 1. Signal Generation Orchestrator (Port 8025)
**File**: `backend/services/trading/signal-generation/orchestrator/signal_gen_orchestrator.py`

**Purpose**: Central coordinator that maintains API compatibility and orchestrates all microservices.

**Key Endpoints**:
- `POST /generate_signal` - Main signal generation endpoint
- `GET /health` - Health check
- `GET /status` - Service status and metrics

**Dependencies**: All other microservices, MySQL database

**Responsibilities**:
- Coordinate service calls in correct order
- Aggregate responses into final signal
- Save comprehensive signal data to database
- Maintain backward compatibility

### 2. ML Engine (Port 8051)
**File**: `backend/services/trading/signal-generation/ml-engine/signal_gen_ml_engine.py`

**Purpose**: Core machine learning predictions using XGBoost models.

**Key Endpoints**:
- `POST /predict` - Generate ML prediction
- `GET /health` - Health check
- `GET /model_info` - Model metadata

**Dependencies**: XGBoost model files, feature data

**Responsibilities**:
- Load and manage XGBoost models
- Generate probability predictions
- Handle model versioning
- Feature compatibility validation

### 3. Feature Engine (Port 8052)
**File**: `backend/services/trading/signal-generation/feature-engine/signal_gen_feature_engine.py`

**Purpose**: Technical indicator calculation and feature engineering.

**Key Endpoints**:
- `POST /engineer_features` - Calculate features
- `GET /health` - Health check
- `GET /available_features` - List available features

**Dependencies**: Price data, technical analysis libraries

**Responsibilities**:
- Calculate technical indicators (RSI, MACD, etc.)
- Feature normalization and scaling
- Missing data handling
- Feature validation

### 4. Market Context (Port 8053)
**File**: `backend/services/trading/signal-generation/market-context/signal_gen_market_context.py`

**Purpose**: Market sentiment and regime analysis.

**Key Endpoints**:
- `POST /analyze` - Analyze market context
- `GET /health` - Health check
- `GET /market_status` - Current market regime

**Dependencies**: Sentiment data sources, market data

**Responsibilities**:
- Sentiment analysis and scoring
- Momentum detection
- Market regime classification
- Volatility assessment

### 5. Portfolio Management (Port 8054)
**File**: `backend/services/trading/signal-generation/portfolio/signal_gen_portfolio.py`

**Purpose**: Position sizing and portfolio allocation logic.

**Key Endpoints**:
- `POST /kelly_sizing` - Calculate Kelly position size
- `GET /health` - Health check
- `GET /portfolio_metrics` - Portfolio statistics

**Dependencies**: Portfolio data, risk parameters

**Responsibilities**:
- Kelly criterion calculations
- Position size optimization
- Portfolio allocation logic
- Risk-adjusted sizing

### 6. Risk Management (Port 8055)
**File**: `backend/services/trading/signal-generation/risk-mgmt/signal_gen_risk_mgmt.py`

**Purpose**: Risk assessment and signal validation.

**Key Endpoints**:
- `POST /assess` - Assess signal risk
- `GET /health` - Health check
- `GET /risk_metrics` - Current risk levels

**Dependencies**: Market data, risk parameters

**Responsibilities**:
- Signal risk scoring
- Market condition evaluation
- Risk limit enforcement
- Selloff/recovery detection

### 7. Analytics (Port 8056)
**File**: `backend/services/trading/signal-generation/analytics/signal_gen_analytics.py`

**Purpose**: Performance tracking and metrics collection.

**Key Endpoints**:
- `POST /track_signal` - Track signal performance
- `GET /health` - Health check
- `GET /metrics` - Performance metrics

**Dependencies**: Database, historical data

**Responsibilities**:
- Signal performance tracking
- Accuracy metrics calculation
- Historical analysis
- Performance reporting

## Data Flow

1. **Request**: Client sends signal generation request to Orchestrator
2. **Feature Engineering**: Orchestrator requests features from Feature Engine
3. **ML Prediction**: Features sent to ML Engine for prediction
4. **Market Analysis**: Market Context analyzes current conditions
5. **Portfolio Sizing**: Portfolio service calculates position size
6. **Risk Assessment**: Risk Management validates signal safety
7. **Analytics**: Analytics service tracks signal generation
8. **Response**: Orchestrator aggregates all data and returns comprehensive signal
9. **Database**: Complete signal saved to trading_signals table
10. **Bridge**: Signal Bridge picks up saved signals for execution

## Database Integration

All signals are saved to the `trading_signals` table with comprehensive schema:

```sql
- id (auto-increment)
- symbol, signal_type, confidence, price, created_at
- model, model_version, features_used, xgboost_confidence
- sentiment_score, risk_score, position_size
- llm_analysis (JSON), features (JSON)
- Various strategy flags and metadata
```

## Health Monitoring

Each service provides:
- `/health` endpoint for basic health checks
- Service-specific status information
- Error logging and monitoring
- Graceful failure handling

## Configuration

Services are configured via:
- Environment variables
- Docker networks (`crypto-trading-network`)
- Shared configuration files
- Database connection pooling

## Deployment

Services are deployed as Docker containers with:
- Individual Dockerfiles for each service
- Shared requirements and dependencies
- Network isolation and communication
- Health check configurations

## Testing

- Individual service testing via direct API calls
- Integration testing through orchestrator
- End-to-end pipeline validation
- Performance and load testing

## Performance

- **Latency**: ~1-2 seconds for complete signal generation
- **Throughput**: Handles multiple concurrent requests
- **Scalability**: Services can be scaled independently
- **Reliability**: Isolated failure domains prevent cascade failures

## Migration Benefits

✅ **Maintainability**: Focused, single-purpose services
✅ **Scalability**: Independent scaling and resource allocation  
✅ **Reliability**: Fault isolation and graceful degradation
✅ **Development**: Parallel development and testing
✅ **Deployment**: Independent service updates and rollbacks
✅ **Monitoring**: Granular metrics and observability