# 🚀 Crypto Trading Pipeline - Quick Reference Guide

## **Current System Status** ✅

| Component | Status | Port | Health Check |
|-----------|--------|------|--------------|
| Enhanced Signal Generator | ✅ Running | 8025 | `/health` |
| Signal Bridge | ✅ Running | 8022 | `/health` |
| Trade Orchestrator | ✅ Running | 8023 | `/health` |
| Trade Executor Real | ✅ Running | 8024 | `/health` |
| Risk Management | ✅ Running | 8027 | `/health` |
| Ollama LLM | ✅ Running | 11434 | `/api/tags` |
| Health Monitor | ✅ Running | 8030 | `/health` |

## **Quick Commands**

### **Check System Status**
```bash
# Check all pods
kubectl get pods -n crypto-trading

# Check services
kubectl get services -n crypto-trading

# Check deployments
kubectl get deployments -n crypto-trading
```

### **View Logs**
```bash
# Signal generator logs
kubectl logs -f deployment/enhanced-signal-generator -n crypto-trading

# Trade executor logs
kubectl logs -f deployment/trade-executor-real -n crypto-trading

# Risk management logs
kubectl logs -f deployment/risk-management-service -n crypto-trading
```

### **Test Services**
```bash
# Test signal generator
kubectl exec deployment/enhanced-signal-generator -n crypto-trading -- curl http://localhost:8025/health

# Test trade executor
kubectl exec deployment/trade-executor-real -n crypto-trading -- curl http://localhost:8024/health

# Test risk management
kubectl exec deployment/risk-management-service -n crypto-trading -- curl http://localhost:8027/health
```

## **Database Queries**

### **Check Recent Signals**
```sql
SELECT symbol, signal_type, confidence, timestamp 
FROM trading_signals 
ORDER BY timestamp DESC 
LIMIT 10;
```

### **Check Pending Recommendations**
```sql
SELECT id, symbol, signal_type, amount_usd, execution_status, llm_validation
FROM trade_recommendations 
WHERE execution_status = 'PENDING'
ORDER BY created_at DESC 
LIMIT 10;
```

### **Check Executed Trades**
```sql
SELECT id, symbol, signal_type, amount_usd, executed_at
FROM trade_recommendations 
WHERE execution_status = 'EXECUTED'
ORDER BY executed_at DESC 
LIMIT 10;
```

## **Performance Monitoring**

### **Key Metrics**
- **Signal Generation Rate**: 480 signals/hour
- **LLM Validation Rate**: 494 validations/24h
- **Trade Execution**: Real Coinbase API calls
- **Risk Management**: Active volatility analysis
- **System Uptime**: 24/7 operation

### **Health Check Endpoints**
```bash
# All services health
curl http://enhanced-signal-generator:8025/health
curl http://signal-bridge:8022/health
curl http://trade-orchestrator-llm:8023/health
curl http://trade-executor-real:8024/health
curl http://risk-management-service:8027/health
```

## **Troubleshooting**

### **Common Issues & Solutions**

#### **1. No Trades Executing**
```bash
# Check Coinbase API connectivity
kubectl exec deployment/trade-executor-real -n crypto-trading -- curl http://localhost:8024/health

# Check account balance (insufficient funds error)
# Solution: Add funds to Coinbase account
```

#### **2. Signal Generation Stopped**
```bash
# Check database connectivity
kubectl exec deployment/enhanced-signal-generator -n crypto-trading -- python -c "import mysql.connector; print('DB OK')"

# Check ML model availability
kubectl exec deployment/enhanced-signal-generator -n crypto-trading -- ls -la /app/models/
```

#### **3. Risk Management Issues**
```bash
# Check portfolio heat
kubectl exec deployment/risk-management-service -n crypto-trading -- curl http://localhost:8027/portfolio-heat

# Check volatility calculation
kubectl exec deployment/risk-management-service -n crypto-trading -- curl http://localhost:8027/volatility/BTC
```

## **Configuration Files**

### **Key Configuration Files**
- `k8s/crypto-trading-system-fixed.yaml` - Core system
- `k8s/trade-executor-real.yaml` - Real trade execution
- `k8s/risk-management-service.yaml` - Risk management
- `k8s/health-monitor.yaml` - Health monitoring

### **Environment Variables**
```bash
# Database
DB_HOST=172.22.32.1
DB_USER=news_collector
DB_PASSWORD=99Rules!

# Coinbase API
COINBASE_API_KEY=organizations/5f04b9a1-3467-4f94-bb5c-2769d89fe5d6/apiKeys/7dd53cef-f159-45af-947f-a861eeb79204
COINBASE_BASE_URL=https://api.coinbase.com/api/v3/brokerage
EXECUTION_MODE=live
```

## **Deployment Commands**

### **Deploy All Services**
```bash
kubectl apply -f k8s/crypto-trading-system-fixed.yaml
kubectl apply -f k8s/trade-executor-real.yaml
kubectl apply -f k8s/risk-management-service.yaml
kubectl apply -f k8s/health-monitor.yaml
```

### **Restart Services**
```bash
kubectl rollout restart deployment/enhanced-signal-generator -n crypto-trading
kubectl rollout restart deployment/trade-executor-real -n crypto-trading
kubectl rollout restart deployment/risk-management-service -n crypto-trading
```

## **Data Flow Summary**

```
Market Data → ML Signals → Recommendations → LLM Validation → Risk Assessment → Real Trades
     ↓            ↓             ↓               ↓                ↓              ↓
crypto_prices → trading_signals → trade_recommendations → APPROVE/REJECT → Risk Score → Coinbase API
```

## **Key Files**

### **Core System Files**
- `working_signal_generator.py` - Main signal generator
- `balanced_realistic_model_20251005_155755.joblib` - ML model
- `CRYPTO_TRADING_PIPELINE_DOCUMENTATION.md` - Full documentation
- `PIPELINE_FLOW_DIAGRAM.md` - Visual flow diagrams

### **Kubernetes Deployments**
- `k8s/crypto-trading-system-fixed.yaml` - Core services
- `k8s/trade-executor-real.yaml` - Real trade execution
- `k8s/risk-management-service.yaml` - Risk management
- `k8s/health-monitor.yaml` - Health monitoring

## **Emergency Procedures**

### **Stop All Trading**
```bash
# Scale down trade executor
kubectl scale deployment trade-executor-real --replicas=0 -n crypto-trading

# Scale down trade orchestrator
kubectl scale deployment trade-orchestrator-llm --replicas=0 -n crypto-trading
```

### **Restart Trading**
```bash
# Scale up services
kubectl scale deployment trade-executor-real --replicas=1 -n crypto-trading
kubectl scale deployment trade-orchestrator-llm --replicas=1 -n crypto-trading
```

### **Check System Health**
```bash
# Run health checks
kubectl get pods -n crypto-trading | grep -v Running
kubectl get services -n crypto-trading
```

---

**Last Updated**: October 8, 2025  
**System Version**: 2.0.0  
**Status**: Production Ready ✅


