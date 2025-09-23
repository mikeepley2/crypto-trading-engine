# 🛡️ Advanced Risk Management Service

**Comprehensive risk management with volatility-based position sizing, portfolio heat monitoring, and correlation analysis**

## Service Details
- **Port**: 8027 (Enhanced from 8025)
- **Type**: REST API (FastAPI)
- **Database**: crypto_prices (price data), crypto_transactions (portfolio data)
- **Dependencies**: pandas, numpy, mysql-connector, scikit-learn

## 🚀 **Advanced Risk Management Features**

### **📊 Volatility-Based Position Sizing** ⭐ **NEW**
- **Dynamic Sizing**: Position sizes adjusted based on 14-day volatility analysis
- **Risk Normalization**: Higher volatility assets get smaller position sizes
- **Volatility Calculation**: Annualized volatility with daily return analysis
- **Size Bounds**: 30%-200% adjustment range with mathematical optimization

### **🔥 Portfolio Heat Monitoring** ⭐ **NEW**
- **Heat Calculation**: Real-time portfolio risk exposure measurement
- **Risk Contribution**: Position weight × volatility for each asset
- **Heat Thresholds**: 15% maximum portfolio heat with dynamic adjustments
- **Alert System**: Automated warnings when heat exceeds safe levels

### **🔗 Correlation-Based Risk Adjustments** ⭐ **NEW**
- **Correlation Analysis**: 30-day rolling correlation between portfolio assets
- **Diversification Scoring**: Portfolio diversification benefit measurement
- **Position Reduction**: Automatic size reduction for highly correlated positions
- **Threshold Management**: 0.7 correlation threshold with graduated responses

### **🎯 Market Regime Adaptation** ⭐ **NEW**
- **Regime Detection**: Bull/bear/sideways market identification
- **Adaptive Sizing**: Position multipliers based on market conditions
- **Risk Scaling**: Dynamic risk tolerance adjustment by market regime
- **Regime Multipliers**: Bull (1.2x), Bear (0.6x), High Volatility (0.7x)

### **⚡ Intelligent Position Optimization**
- **Multi-Factor Analysis**: Combines volatility, correlation, heat, and regime factors
- **Risk Score Calculation**: Comprehensive 0-1 risk scoring system
- **Position Bounds**: Automatic minimum ($25) and maximum position enforcement
- **Warning System**: Detailed risk warnings and recommendations

## 📈 **API Endpoints**

### **🔍 Core Risk Analysis**
- `GET /health` - Service health with database connectivity check
- `GET /risk/limits` - Current risk management limits and parameters
- `POST /risk/check_trade` - Legacy basic trade risk validation
- `POST /limits/update` - Update risk management parameters (admin)

### **📊 Advanced Position Sizing** ⭐ **NEW**
- `POST /position-size/calculate` - Optimal position size with multi-factor analysis
- `GET /volatility/{symbol}` - Asset volatility metrics and adjustments
- `GET /correlation/{symbol1}/{symbol2}` - Asset correlation analysis
- `POST /portfolio/risk-analysis` - Comprehensive portfolio risk assessment

### **🔥 Portfolio Risk Monitoring** ⭐ **NEW**
- `GET /portfolio/heat` - Real-time portfolio heat calculation
- `GET /risk/concentration` - Portfolio concentration risk analysis
- `GET /risk/diversification` - Portfolio diversification scoring
- `GET /metrics/risk-breakdown` - Detailed risk attribution by asset

## 🧮 **Position Sizing Algorithm**

### **Multi-Factor Position Sizing**
```python
# Comprehensive position sizing calculation
def calculate_optimal_position_size(symbol, base_size, current_positions, market_regime):
    adjustments = {}
    
    # 1. Volatility adjustment (30%-200% range)
    volatility_multiplier = 1.0 / (1.0 + volatility * 10)
    adjusted_size *= volatility_multiplier
    
    # 2. Correlation adjustment (down to 20% if highly correlated)
    correlation_multiplier = 1.0 - (max_correlation - 0.7) * 2
    adjusted_size *= max(0.2, correlation_multiplier)
    
    # 3. Portfolio heat adjustment
    heat_multiplier = get_heat_adjustment(portfolio_heat)
    adjusted_size *= heat_multiplier
    
    # 4. Market regime adjustment
    regime_multiplier = REGIME_MULTIPLIERS[market_regime]
    adjusted_size *= regime_multiplier
    
    # 5. Concentration limits (25% max position)
    adjusted_size = min(adjusted_size, total_portfolio * 0.25)
    
    return {
        'optimal_size': adjusted_size,
        'adjustments': adjustments,
        'risk_level': categorize_risk_level(adjusted_size, base_size),
        'warnings': generate_risk_warnings(symbol, adjusted_size, adjustments)
    }
```

### **Risk Level Categories**
- **Conservative**: <60% of base size (high risk reduction)
- **Reduced**: 60-90% of base size (moderate risk reduction)
- **Normal**: 90-110% of base size (standard sizing)
- **Elevated**: 110-150% of base size (increased opportunity)
- **High**: >150% of base size (maximum opportunity sizing)

## 🔍 **Usage Examples**

### **Calculate Optimal Position Size**
```bash
curl -X POST http://localhost:8027/position-size/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC",
    "base_size": 500.0,
    "current_positions": {
      "ETH": {"value_usd": 800},
      "ADA": {"value_usd": 300}
    },
    "market_regime": "bull_market"
  }'
```

**Example Response:**
```json
{
  "optimal_size": 650.75,
  "adjustments": {
    "base_size": 500.0,
    "volatility_multiplier": 0.85,
    "correlation_multiplier": 1.0,
    "portfolio_heat": 0.12,
    "heat_multiplier": 1.0,
    "regime_multiplier": 1.2,
    "final_size": 650.75
  },
  "risk_level": "elevated",
  "warnings": []
}
```

### **Get Asset Volatility Analysis**
```bash
curl http://localhost:8027/volatility/BTC
```

**Example Response:**
```json
{
  "symbol": "BTC",
  "volatility": 0.78,
  "volatility_adjustment": 0.89,
  "risk_level": "medium",
  "timestamp": "2025-08-25T10:30:00Z"
}
```

### **Analyze Portfolio Risk**
```bash
curl -X POST http://localhost:8027/portfolio/risk-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "BTC": {"value_usd": 1200},
    "ETH": {"value_usd": 800},
    "ADA": {"value_usd": 400}
  }'
```

**Example Response:**
```json
{
  "portfolio_heat": 0.145,
  "concentration_risk": 0.22,
  "correlation_risk": 0.35,
  "risk_score": 0.65,
  "risk_level": "moderate",
  "recommendations": [
    "Monitor portfolio heat - approaching 15% threshold",
    "Consider diversification to reduce correlation risk"
  ],
  "total_positions": 3,
  "total_value": 2400.0
}
```

### **Get Portfolio Heat**
```bash
curl http://localhost:8027/portfolio/heat
```

**Example Response:**
```json
{
  "portfolio_heat": 0.125,
  "max_heat_threshold": 0.15,
  "heat_level": "moderate",
  "positions_analyzed": 4,
  "total_value": 2845.67,
  "timestamp": "2025-08-25T10:30:00Z"
}
```

## 🔧 **Risk Management Parameters**

### **Default Risk Limits**
```python
MAX_PORTFOLIO_HEAT = 0.15          # 15% maximum portfolio at risk
BASE_POSITION_SIZE = 200.0         # Base position size in USD
VOLATILITY_LOOKBACK = 14           # Days for volatility calculation
CORRELATION_THRESHOLD = 0.7        # High correlation threshold
MAX_POSITION_SIZE = 0.25           # 25% maximum single position
MIN_POSITION_SIZE = 25.0           # $25 minimum position size
```

### **Market Regime Multipliers**
```python
REGIME_MULTIPLIERS = {
    'bull_market': 1.2,      # Increase positions in bull markets
    'bear_market': 0.6,      # Reduce positions in bear markets
    'high_volatility': 0.7,  # Reduce during high volatility
    'sideways': 1.0          # Normal sizing in sideways markets
}
```

### **Heat Adjustment Levels**
```python
def get_heat_adjustment(portfolio_heat):
    if portfolio_heat <= 0.075:     # ≤50% of max heat
        return 1.2                   # Low heat - increase positions
    elif portfolio_heat <= 0.12:    # ≤80% of max heat
        return 1.0                   # Normal heat - standard sizing
    elif portfolio_heat <= 0.15:    # ≤100% of max heat
        return 0.8                   # High heat - reduce positions
    else:                           # >100% of max heat
        return 0.5                   # Critical heat - major reduction
```

## 🔄 **Integration with Trading System**

### **Signal Generator Integration**
- **Position Sizing**: Enhanced signal generator calls risk service for optimal sizing
- **Risk Validation**: All signals validated against current risk parameters
- **Portfolio Context**: Risk analysis considers current portfolio composition

### **Portfolio Service Integration**
- **Risk Metrics**: Portfolio service uses risk analysis for health scoring
- **Rebalancing**: Risk service provides input for rebalancing decisions
- **Concentration**: Shared concentration risk analysis and monitoring

### **Trade Execution Integration**
- **Pre-Trade Checks**: All trades validated against risk limits
- **Dynamic Sizing**: Position sizes adjusted based on real-time risk analysis
- **Post-Trade Updates**: Risk metrics updated after trade execution

## 📊 **Risk Monitoring & Alerts**

### **Portfolio Health Monitoring**
- **Heat Warnings**: Alerts when portfolio heat >80% of maximum
- **Concentration Alerts**: Warnings when single position >20%
- **Correlation Notices**: Alerts for high correlation between major positions
- **Volatility Spikes**: Notifications during high volatility periods

### **Risk Level Escalation**
1. **Green (0-30% risk score)**: Normal operations, standard monitoring
2. **Yellow (30-60% risk score)**: Increased monitoring, position size reductions
3. **Orange (60-80% risk score)**: Active risk management, rebalancing recommended
4. **Red (80-100% risk score)**: Emergency protocols, immediate risk reduction

## 🧠 **Advanced Risk Algorithms**

### **Correlation Matrix Calculation**
```python
# 30-day rolling correlation analysis
def calculate_correlation_matrix(symbols):
    # Fetch 30 days of price data
    price_data = get_price_data(symbols, days=30)
    
    # Calculate daily returns
    returns = price_data.pct_change().dropna()
    
    # Calculate correlation matrix
    correlation_matrix = returns.corr()
    
    return correlation_matrix
```

### **Portfolio Heat Formula**
```python
# Portfolio heat calculation
def calculate_portfolio_heat(positions):
    total_heat = 0
    total_value = sum(pos['value_usd'] for pos in positions.values())
    
    for symbol, position in positions.items():
        weight = position['value_usd'] / total_value
        volatility = get_asset_volatility(symbol)
        heat_contribution = weight * volatility
        total_heat += heat_contribution
    
    return total_heat
```

### **Risk Score Calculation**
```python
# Comprehensive risk scoring
def calculate_risk_score(portfolio_heat, concentration_risk, correlation_risk):
    # Weighted risk score (0-1 scale)
    risk_score = (
        portfolio_heat * 0.4 +      # 40% weight to portfolio heat
        concentration_risk * 0.3 +   # 30% weight to concentration
        correlation_risk * 0.3       # 30% weight to correlation
    )
    
    return min(1.0, max(0.0, risk_score))
```

---

**Status**: ✅ **LIVE & OPTIMIZED** - Advanced risk management active, protecting automated trading system with comprehensive multi-factor analysis
