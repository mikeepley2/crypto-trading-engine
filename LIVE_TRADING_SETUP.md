# 🚀 Live Trading Setup Guide

## ✅ Current Status
- **Trade Orchestrator**: ✅ Working and processing recommendations
- **Signal Generation**: ✅ Working (manual signals tested)
- **Database**: ✅ Connected and operational
- **API Integration**: ⚠️ Needs real credentials

## 🔑 Required API Credentials

To enable live trading, you need to configure these environment variables:

### 1. Create `.env.live` file in the project root:

```bash
# Live Trading Configuration
# WARNING: This file contains sensitive API credentials
# Keep this file secure and never commit it to version control

# Coinbase API Configuration (Required for live trading)
COINBASE_API_KEY=your_actual_api_key_here
COINBASE_PRIVATE_KEY=your_actual_private_key_here
COINBASE_BASE_URL=https://api.coinbase.com

# Trading Execution Mode
EXECUTION_MODE=live
TRADE_EXECUTION_ENABLED=true

# Trading Limits (Conservative settings for safety)
MAX_POSITION_SIZE_USD=100.00
MAX_DAILY_TRADES=10
MAX_DAILY_LOSS_USD=200.00
MIN_TRADE_SIZE_USD=25.00
MIN_LIQUIDATION_VALUE_USD=25.00

# Risk Management
STOP_LOSS_PERCENT=0.05
TAKE_PROFIT_PERCENT=0.10

# Database Configuration
DB_HOST=172.22.32.1
DB_USER=news_collector
DB_PASSWORD=99Rules!
DB_NAME_TRANSACTIONS=crypto_transactions
DB_NAME_PRICES=crypto_prices

# Redis Configuration
REDIS_HOST=infra-redis
REDIS_PORT=6379
```

### 2. Get Coinbase Advanced Trade API Credentials

1. **Log into Coinbase Advanced Trade**
2. **Go to Settings → API**
3. **Create a new API key** with these permissions:
   - ✅ View account information
   - ✅ Place orders
   - ✅ View orders
   - ❌ Withdraw funds (for safety)
4. **Copy the API Key and Private Key**

### 3. Update the deployment with new environment variables

```bash
# Update the trade orchestrator deployment
kubectl set env deployment/configurable-trade-orchestrator \
  EXECUTION_MODE=live \
  TRADE_EXECUTION_ENABLED=true \
  COINBASE_API_KEY=your_api_key \
  COINBASE_PRIVATE_KEY=your_private_key \
  -n crypto-trading
```

## 🧪 Testing with Small Amounts

Before full deployment, test with small amounts:

1. **Start with $25 minimum trades**
2. **Set conservative limits**:
   - Max position: $100
   - Max daily trades: 5
   - Max daily loss: $50
3. **Monitor closely** for the first few trades
4. **Gradually increase** limits as confidence grows

## 📊 Current System Status

### ✅ Working Components:
- Trade orchestrator finds and processes recommendations
- Database stores and retrieves trading signals
- Risk management system is active
- Monitoring and logging are operational

### ⚠️ Needs Configuration:
- Real API credentials for live trading
- Signal generator ML model (currently using manual signals)
- Position sizing and risk limits

## 🚨 Safety Features

The system includes multiple safety features:
- **Duplicate trade prevention** (120-second window)
- **Daily loss limits** ($200 max)
- **Position size limits** ($100 max)
- **Stop-loss protection** (5% default)
- **Take-profit targets** (10% default)

## 📈 Next Steps

1. **Configure API credentials** (this guide)
2. **Test with small amounts** ($25 trades)
3. **Monitor performance** for 24-48 hours
4. **Gradually increase** position sizes
5. **Enable continuous signal generation**

## 🔍 Monitoring

Monitor the system with:
```bash
# Check trade orchestrator logs
kubectl logs -n crypto-trading deployment/configurable-trade-orchestrator -f

# Check for executed trades
kubectl exec -n crypto-trading deployment/configurable-trade-orchestrator -- python -c "
import mysql.connector
conn = mysql.connector.connect(host='172.22.32.1', user='news_collector', password='99Rules!', database='crypto_transactions')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM trades WHERE created_at > DATE_SUB(NOW(), INTERVAL 1 HOUR)')
print('Recent trades:', cursor.fetchone()[0])
conn.close()
"
```

---

**⚠️ IMPORTANT**: Always test with small amounts first and monitor closely!



