# Branch Strategy and Deployment Guide

## 🌿 **BRANCH STRATEGY**

### **Production Branch: `master`**
- **Purpose:** Stable, production-ready code
- **Status:** ✅ **CURRENTLY PRODUCTION READY**
- **Deployment:** Live trading system with all services operational
- **Services:** 8/9 critical services running and healthy
- **Monitoring:** Full Prometheus/Loki/Grafana stack active

### **Development Branch: `dev`**
- **Purpose:** Active development and testing
- **Status:** ✅ **ACTIVE DEVELOPMENT BRANCH**
- **Usage:** All new features, experiments, and testing
- **Protection:** Production system remains unaffected

## 🚀 **DEPLOYMENT STATUS**

### **✅ PRODUCTION SYSTEM (master branch):**

#### **Core Services Running:**
1. **Signal Generator (Enhanced)** - ML model loaded, Prometheus metrics active
2. **Trade Executor** - Live trading with Coinbase Advanced Trade API
3. **Trade Orchestrator** - Processing recommendations every 30 seconds
4. **LLM Validation** - AI-powered trade validation with Ollama
5. **Risk Management** - Pre-trade risk analysis and position sizing
6. **Ollama Server** - LLM model ready for validation
7. **Grafana** - Monitoring dashboards with Prometheus/Loki datasources
8. **Health Monitor** - System health tracking

#### **Monitoring Infrastructure:**
- **Prometheus** - Metrics collection and storage
- **Loki** - Log aggregation and storage
- **Promtail** - Log collection from all nodes
- **Grafana** - Visualization and dashboards
- **Service Discovery** - Automatic target discovery

### **🔄 DEVELOPMENT SYSTEM (dev branch):**
- **Status:** Ready for new development
- **Protection:** Production system isolated
- **Usage:** All experimental features and testing

## 📊 **SYSTEM ARCHITECTURE**

### **Node Distribution:**
- **cryptoai-k8s-trading-engine-worker2 (Trading Engine):**
  - Signal Generator (Enhanced)
  - Trade Executor
  - Risk Management

- **cryptoai-k8s-trading-engine-worker3 (Analytics):**
  - LLM Validation
  - Trade Orchestrator
  - Ollama Server
  - Grafana
  - Health Monitor

- **cryptoai-k8s-trading-engine-worker (Data Collection):**
  - Health Monitor

## 🎯 **PIPELINE FLOW**

```
Signal Generator ✅ → Database ✅ → Recommendations ✅
    ↓
Trade Orchestrator ✅ → LLM Validation ✅ → Risk Management ✅
    ↓
Trade Executor ✅ → Live Trading ✅
```

## 📈 **MONITORING CAPABILITIES**

### **Prometheus Metrics:**
- Signal generation counters and confidence histograms
- Trade execution metrics and API response times
- Model inference timing and performance
- System health and resource utilization

### **Log Aggregation:**
- Centralized logging with Loki
- Automatic log collection from all pods
- Structured logging with proper labels
- Log retention and storage management

### **Visualization:**
- Grafana dashboards for all services
- Real-time monitoring and alerting
- Historical data analysis
- Performance trend tracking

## 🔧 **DEVELOPMENT WORKFLOW**

### **Working on New Features:**
1. **Switch to dev branch:** `git checkout dev`
2. **Make changes:** Develop and test new features
3. **Test thoroughly:** Ensure no impact on production
4. **Merge to master:** When ready for production deployment

### **Production Deployment:**
1. **Switch to master:** `git checkout master`
2. **Pull latest:** `git pull origin master`
3. **Deploy services:** `kubectl apply -f k8s/`
4. **Monitor deployment:** Check service health and metrics

## 📋 **SERVICE ENDPOINTS**

### **Production Services:**
- **Signal Generator:** `http://localhost:8025/metrics`
- **Trade Executor:** `http://localhost:8024/health`
- **LLM Validation:** `http://localhost:8050/status`
- **Risk Management:** `http://localhost:8027/status`
- **Grafana:** `http://localhost:3000`
- **Prometheus:** `http://localhost:9090`

## 🚨 **IMPORTANT NOTES**

### **Production Protection:**
- **NEVER** make direct changes to master branch
- **ALWAYS** develop on dev branch first
- **TEST** thoroughly before merging to production
- **MONITOR** production system continuously

### **Service Management:**
- All services are deployed via Kubernetes
- Use `kubectl` commands for service management
- Monitor logs and metrics for health checks
- Use Grafana dashboards for system overview

## 📚 **DOCUMENTATION**

### **Key Documents:**
- `FINAL_SYSTEM_STATUS_2025_10_14.md` - Current production status
- `FINAL_SYSTEM_DOCUMENTATION.md` - Complete system documentation
- `PIPELINE_FLOW_DIAGRAM.md` - System architecture and flow
- `QUICK_REFERENCE_GUIDE.md` - Quick commands and references

### **Deployment Files:**
- `k8s/corrected-architecture-deployments.yaml` - Main service deployments
- `k8s/monitoring-stack.yaml` - Prometheus, Loki, Promtail
- `k8s/grafana-monitoring.yaml` - Grafana with datasources
- `k8s/service-monitors.yaml` - Prometheus service discovery

---

**Last Updated:** October 14, 2025
**Production Status:** ✅ **OPERATIONAL**
**Development Status:** ✅ **READY FOR DEVELOPMENT**
**Branch Strategy:** ✅ **IMPLEMENTED**
