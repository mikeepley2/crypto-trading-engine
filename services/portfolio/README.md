# 📊 Portfolio Management Service

**Advanced portfolio analytics and intelligent rebalancing with correlation analysis and risk management**

## Service Details
- **Port**: 8026
- **Type**: REST API (FastAPI)
- **Database**: crypto_transactions (portfolio data), crypto_prices (price data)
- **Dependencies**: httpx for price fetching, mysql-connector, pandas, numpy

## 🚀 **Enhanced Features**

### **📈 Core Portfolio Management**
- Real-time portfolio position tracking with live price updates
- Automatic price updates from Coinbase/Binance APIs  
- Holdings management (buy/sell operations)
- Cash balance tracking with deployment optimization
- Comprehensive P&L calculations (realized and unrealized)

### **⚖️ Advanced Portfolio Rebalancing** ⭐ **NEW**
- **Correlation Analysis**: 30-day correlation matrix calculation for portfolio diversification
- **Concentration Risk Management**: Automated detection of over-concentrated positions (>25%)
- **Volatility-Adjusted Sizing**: Dynamic position sizing based on 30-day volatility analysis
- **Sector Diversification**: Asset categorization and category-based concentration limits
- **Momentum-Based Adjustments**: Performance-driven rebalancing recommendations

### **🛡️ Advanced Risk Analytics** ⭐ **NEW**
- **Portfolio Heat Calculation**: Real-time risk exposure monitoring
- **Herfindahl-Hirschman Index**: Concentration risk measurement
- **Correlation Risk Assessment**: Portfolio diversification analysis
- **Multi-Strategy Rebalancing**: 4 distinct rebalancing strategies
- **Risk-Adjusted Recommendations**: Prioritized action items with confidence scores

### **📊 Performance Analytics**
- Portfolio performance metrics calculation with risk adjustments
- Win rate and profit factor analysis with volatility context
- Daily and total return calculations
- Trade statistics and analytics with correlation impact
- Historical performance tracking with risk attribution

### **📊 API Endpoints**

#### **Portfolio Operations**
- `GET /health` - Service health check with database connectivity
- `GET /portfolio` - Complete portfolio summary with risk metrics
- `GET /positions` - All current positions with live prices and allocations
- `GET /performance?days=30` - Performance metrics for specified period
- `POST /holdings/update` - Update holdings after trade execution
- `POST /portfolio/refresh` - Refresh all positions with current prices

#### **🔄 Advanced Rebalancing** ⭐ **NEW**
- `GET /rebalance/analysis` - Comprehensive portfolio rebalancing analysis
- `POST /rebalance/generate-signals` - Generate trading signals based on rebalancing analysis
- `GET /rebalance/recommendations` - Get current rebalancing recommendations without signals
- `GET /risk/concentration` - Portfolio concentration risk analysis

#### **📈 Risk Analytics** ⭐ **NEW**
- `GET /risk/correlation-matrix` - Asset correlation analysis for diversification
- `GET /risk/volatility-analysis` - Portfolio volatility breakdown by asset
- `GET /risk/sector-allocation` - Asset allocation by sector/category
- `GET /risk/portfolio-health` - Overall portfolio health score and recommendations

## 📊 **Advanced Rebalancing Examples**

### **Get Comprehensive Rebalancing Analysis**
```bash
curl http://localhost:8026/rebalance/analysis
```

**Example Response:**
```json
{
  "status": "success",
  "analysis": {
    "portfolio_health_score": 75,
    "total_positions": 8,
    "total_recommendations": 3,
    "concentration_analysis": {
      "hhi": 0.18,
      "concentration_level": "medium",
      "over_concentrated_positions": [
        {
          "symbol": "BTC",
          "current_allocation": 28.5,
          "excess_allocation": 3.5,
          "excess_value": 245.67
        }
      ]
    },
    "average_correlation": 0.45,
    "high_priority_actions": 1,
    "recommendations": [
      {
        "symbol": "BTC",
        "action": "SELL",
        "reason": "concentration_risk",
        "trade_value": 245.67,
        "priority": "high",
        "confidence": 0.85
      }
    ]
  }
}
```

### **Generate Rebalancing Signals**
```bash
curl -X POST http://localhost:8026/rebalance/generate-signals
```

### **Get Concentration Risk Analysis**
```bash
curl http://localhost:8026/risk/concentration
```

**Example Response:**
```json
{
  "status": "success",
  "concentration_analysis": {
    "hhi": 0.18,
    "concentration_level": "medium",
    "over_concentrated_positions": [],
    "over_concentrated_categories": [],
    "num_positions": 6,
    "recommended_action": "maintain"
  }
}
```

## 🔧 **Usage Examples**

### Get Portfolio Summary
```bash
curl http://localhost:8026/portfolio
```

### Get Current Positions
```bash
curl http://localhost:8026/positions
```

### Get Performance Metrics
```bash
curl http://localhost:8026/performance?days=7
```

### Update Holdings (called by trading engines)
```bash
curl -X POST http://localhost:8026/holdings/update \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC",
    "side": "BUY", 
    "quantity": 0.1,
    "price": 45000.0,
    "net_amount": 4500.0,
    "fees": 22.5
  }'
```

## 🔄 **Integration with Trading Engines**

### **Enhanced Integration Points**
Trading engines should integrate with the portfolio service for:
1. **Portfolio Updates**: After each trade execution with risk context
2. **Position Analysis**: Before position sizing with correlation awareness
3. **Cash Management**: Before orders with rebalancing considerations
4. **Risk Assessment**: For strategy evaluation with portfolio health metrics
5. **Rebalancing Signals**: Automated rebalancing signal generation

### **Advanced Integration Example**
```python
import httpx

# Enhanced portfolio integration with rebalancing
async def get_portfolio_with_rebalancing_analysis():
    async with httpx.AsyncClient() as client:
        # Get portfolio summary
        portfolio = await client.get("http://localhost:8026/portfolio")
        
        # Get rebalancing analysis
        rebalancing = await client.get("http://localhost:8026/rebalance/analysis")
        
        # Get concentration risk
        concentration = await client.get("http://localhost:8026/risk/concentration")
        
        return {
            "portfolio": portfolio.json(),
            "rebalancing": rebalancing.json(),
            "risk": concentration.json()
        }

# Check if rebalancing is needed
async def check_rebalancing_needed():
    async with httpx.AsyncClient() as client:
        analysis = await client.get("http://localhost:8026/rebalance/analysis")
        data = analysis.json()
        
        # Check portfolio health score
        health_score = data["analysis"]["portfolio_health_score"]
        high_priority_actions = data["analysis"]["high_priority_actions"]
        
        return {
            "rebalancing_needed": health_score < 70 or high_priority_actions > 0,
            "health_score": health_score,
            "priority_actions": high_priority_actions
        }
```

## 🧠 **Rebalancing Strategies**

### **1. Concentration Risk Reduction**
- **Trigger**: Single position >25% of portfolio
- **Action**: Generate SELL signals to reduce concentration
- **Target**: Reduce to 23% allocation (2% buffer)

### **2. Correlation-Based Diversification**
- **Trigger**: High correlation (>0.7) between major positions
- **Action**: Reduce position in weaker performing asset
- **Target**: Improve portfolio diversification

### **3. Volatility-Adjusted Positioning**
- **Trigger**: High volatility assets with large allocations
- **Action**: Reduce position sizes for high-risk assets
- **Target**: Risk-adjusted position sizing

### **4. Momentum-Based Adjustments**
- **Trigger**: Strong negative momentum in significant positions
- **Action**: Reduce positions with poor momentum
- **Target**: Improve portfolio momentum profile

## 📈 **Risk Metrics**

### **Portfolio Health Score**
- **Range**: 0-100 (higher is better)
- **Factors**: Concentration (-15 per over-concentrated position), correlation (-30 max), category concentration (-10 per over-concentrated category)
- **Target**: >70 for healthy portfolio

### **Concentration Measurements**
- **HHI**: Herfindahl-Hirschman Index for concentration
- **Position Limits**: 25% maximum single position
- **Category Limits**: 40% maximum per asset category

### **Correlation Analysis**
- **Threshold**: 0.7 correlation triggers diversification recommendations
- **Lookback**: 30-day correlation calculation
- **Diversification**: Lower correlation improves portfolio risk profile

## Database Schema

Uses the existing mock trading tables:
- `mock_holdings` - Current positions and entry prices
- `mock_portfolio` - Portfolio summary and cash balance
- `mock_trades` - Trade history for performance calculations
- `mock_performance_history` - Daily portfolio snapshots

## Configuration

### Database Connection
- Host: `host.docker.internal` (Docker environment)
- User: `news_collector`
- Password: `99Rules!`
- Database: `crypto_transactions`

### Price Data Sources
1. **Primary**: Coinbase API (`https://api.coinbase.com/v2/prices/{symbol}-USD/spot`)
2. **Fallback**: Binance API (`https://api.binance.com/api/v3/ticker/price?symbol={symbol}USDT`)

## Running the Service

### Local Development
```bash
cd backend/services/trading/portfolio/
python3 portfolio_service.py
```

### Docker Container
```bash
docker build -t portfolio-service .
docker run -p 8026:8026 portfolio-service
```

## Monitoring and Logging

- Service logs to `portfolio_service.log`
- Structured logging with timestamps
- Health check endpoint for monitoring
- Performance metrics for service optimization

## Future Enhancements

- [ ] Portfolio optimization algorithms
- [ ] Risk metrics calculation (VaR, Sharpe ratio, etc.)
- [ ] Multi-timeframe performance analysis
- [ ] Portfolio rebalancing recommendations
- [ ] Integration with risk management service
- [ ] Real-time portfolio streaming via WebSocket
- [ ] Portfolio backtesting capabilities
