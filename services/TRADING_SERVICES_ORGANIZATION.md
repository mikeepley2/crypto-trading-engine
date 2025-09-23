# Trading Engine Services Organization

## New Directory Structure

### `/backend/services/trading/` - Unified Trading Services

This directory consolidates all trading-related services into a logical, scalable structure:

#### **Core Service Categories**

##### 1. **`/engines/`** - Trade Execution Engines
- **Mock Trading Engine** (`fixed_mock_trading_engine.py`) - Port 8021
- **Live Trading Engine** (`live_trading_engine.py`) - Port 8023 (future)
- **Paper Trading Engine** - Port 8024 (future)
- **Backtesting Engine** - Port 8027 (future)

##### 2. **`/recommendations/`** - Trade Recommendation Services  
- **ML Recommendation Engine** (`trade_recommendation_service.py`) - Port 8022
- **Signal Aggregation Service** - Port 8028 (future)
- **Strategy Recommendation Service** - Port 8025 (future)

##### 3. **`/portfolio/`** - Portfolio Management Services
- **Portfolio Manager Service** (`portfolio_service.py`) - Port 8026 ✅ Active (core endpoints: /health, /portfolio, /positions, /performance, /holdings/update)
- **Position Tracker** - Real-time position monitoring (basic implementation via /positions)
- **Performance Calculator** - P&L and metrics calculation (initial metrics via /performance)
- **Allocation Manager** - Portfolio rebalancing logic (🔮 Future)

##### 4. **`/risk/`** - Risk Management Services
- **Risk Management Service** (`risk_service.py`) - Port 8025 ✅ Enhanced (size limits, allocation & symbol exposure, daily trade cap)
- **Risk Manager** - VaR calculations and limits (🔮 Future)
- **Stop Loss Manager** - Automated stop-loss execution (🔮 Future)
- **Position Sizer** - Dynamic position sizing logic (🔮 Future)
- **Exposure Monitor** - Real-time risk monitoring (🔮 Future)

##### 5. **`/analytics/`** - Trading Analytics Services
- **Performance Analytics** - Port 8029 (future)
- **Trade Analytics** - Trade performance analysis
- **Market Analytics** - Market condition analysis
- **Reporting Service** - Generate trading reports

##### 6. **`/signals/`** - Signal Processing Services
- **Signals Service** (`signals_service.py`) - Port 8028 🟡 Partial (recent signals endpoint, Prometheus metrics)
- **ML Signal Processor** - Advanced aggregation (🔮 Future)
- **Technical Signal Generator** - TA-based signals (🔮 Future)
- **Sentiment Signal Processor** - News/social signals (🔮 Future)
- **Fundamental Signal Generator** - Economic signals (🔮 Future)

##### 7. **`/shared/`** - Shared Components
- **Database Schemas** (`enhanced_mock_trading_schema.py`)
- **Common Models** - Shared data models
- **Utilities** - Helper functions and tools
- **Configuration** - Service configuration files

## Port Allocation by Service Category

### **Engines (8021, 8023-8024, 8027)**
- **8021**: Mock Trading Engine (deprecated; use unified engine on 8024 in mock mode)
- **8023**: Live Trading Engine 🔮 Future
- **8024**: Paper Trading Engine 🔮 Future  
- **8027**: Backtesting Engine 🔮 Future

### **Recommendations & Signals (8022, 8025, 8028)**
- **8022**: Trade Recommendation Engine ✅ Active (auth + metrics + execution_status unified)
- **8025**: Risk Management Service ✅ Enhanced (exposure checks)
- **8028**: Signals Service 🟡 Partial (recent signals API + metrics)

### **Portfolio & Analytics (8026, 8029)**
- **8026**: Portfolio Manager Service ✅ Active
- **8029**: Trading Analytics Service 🔮 Future

### **Reserved/Conflicts**
- **8020**: Available (realtime-materialized-updater moved to K8s)

## Service Dependencies

### **Current Active Services**
```mermaid
graph TD
    A[ML Models] --> B[Trade Recommendation Engine :8022]
   B --> C[Trade Execution Engine (mock mode) :8024]
    C --> D[Portfolio Database]
    C --> E[Price APIs]
    B --> F[Signal Database]
```

### **Future Service Architecture**
```mermaid
graph TD
    A[ML Signal Processor :8028] --> B[Trade Recommendation Engine :8022]
    C[Risk Manager] --> B
   B --> D[Trade Execution Engine (mock/live) :8024]
   D --> F[Portfolio Manager :8026]
    F --> G[Analytics Service :8029]
    H[Backtesting Engine :8027] --> G
```

## Migration Plan

### **Phase 1: Reorganization (Current)**
- [x] Create new directory structure
- [x] Move existing services to appropriate directories (mock engine, recommendations, portfolio)
- [x] Update import paths and references (initial pass)
- [x] Update Docker Compose configurations (added portfolio service)

### **Phase 2: Service Extraction (Week 1-2)**
- [x] Extract portfolio management from trading engines (fallback removed)
- [x] Create dedicated risk management service (enhanced exposure checks)
- [x] Implement signal processing service (basic recent signals endpoint)
- [ ] Add analytics service framework

### **Phase 3: Advanced Services (Week 3-4)**
- [ ] Live trading engine implementation
- [ ] Advanced risk management features
- [ ] Comprehensive analytics dashboard
- [ ] Performance optimization

## Benefits of New Organization

### **1. Separation of Concerns**
- Each service has a single, well-defined responsibility
- Easier to maintain and debug individual components
- Independent scaling of different service types

### **2. Scalability**
- Services can be deployed independently
- Resource allocation based on service requirements
- Horizontal scaling of high-demand services

### **3. Development Efficiency**
- Clear ownership and boundaries for each service
- Easier onboarding for new developers
- Reduced code duplication

### **4. Operational Excellence**
- Independent monitoring and alerting per service
- Granular health checks and metrics
- Easier troubleshooting and debugging

### **5. Future Expansion**
- Clear patterns for adding new trading services
- Standardized interfaces between services
- Plugin architecture for strategy extensions

## Implementation Steps

### **Immediate Actions (Completed)**
1. **Move Existing Services**
   ```bash
   # Move mock trading engine
   mv backend/services/trading_engine/* backend/services/trading/engines/
   
   # Move recommendation service  
   mv backend/services/trading_recommendation_engine/* backend/services/trading/recommendations/
   ```

2. **Update Docker Configurations**
   - Update build contexts in docker-compose.trading.yml
   - Maintain existing port assignments (8021, 8022)
   - Update health check paths

3. **Update Documentation**
   - Update all service paths in documentation
   - Revise port allocation tables
   - Update developer guides

### **Next Phase Actions (Updated)**
1. **Advanced Risk Enhancements**
   - Add VaR & volatility-based limits
   - Persist risk counters to DB
2. **Signals Expansion**
   - Integrate ML signal processor & quality metrics
3. **Analytics Service**
   - Performance & attribution endpoints
4. **Auth & Observability Hardening**
   - API key rotation + per-service scopes
   - Structured logging & tracing (OTel)
5. **Data Consistency**
   - Remove legacy `status` column (post-migration) in favor of `execution_status`
6. **Testing & CI**
   - Expand integration test coverage & add CI pipeline

## File Movement Mapping

### **Current → New Location**
```
trading_engine/fixed_mock_trading_engine.py → trading/engines/mock_trading_engine.py
trading_engine/improved_live_trading_engine.py → trading/engines/live_trading_engine.py
trading_engine/enhanced_mock_trading_schema.py → trading/shared/database_schemas.py
trading_engine/TRADING_ENGINE_PORTS.md → trading/TRADING_SERVICES_DOCUMENTATION.md

trading_recommendation_engine/trade_recommendation_service.py → trading/recommendations/recommendation_service.py
trading_recommendation_engine/create_crypto_transactions_schema.py → trading/shared/database_schemas.py
```

### **Docker Configuration Updates**
```yaml
# New docker-compose.trading.yml structure
services:
  mock-trading-engine:
    build:
      context: ./trading/engines
      dockerfile: Dockerfile.mock-trading
    
  trade-recommendations:
    build:
      context: ./trading/recommendations  
      dockerfile: Dockerfile.trade-recommendations
```

## Cross-Cutting Features Implemented

- Unified API Key auth (`X-TRADING-API-KEY`) across active services
- Prometheus-compatible `/metrics` endpoints (engine, recommendations, portfolio, risk, signals)
- Execution status unification: `status` kept for backward compatibility; `execution_status` authoritative
- Enhanced risk evaluation (allocation%, symbol exposure%, daily trade cap)
- Dockerized risk & signals services (new Dockerfiles + requirements)
- Initial integration test scaffold (auth, metrics, risk rejection)

This organization provides a solid foundation for scaling the trading system while maintaining clear boundaries between different types of trading services.
