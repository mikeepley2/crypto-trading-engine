# Kubernetes Trading Services Naming Convention

## 🏗️ **Architectural Categories**

### **Signal Generation (Existing K8s Services)**
Pattern: `signal-gen-<function>`
- ✅ `signal-gen-orchestrator` - Coordinates ML signal generation
- ✅ `signal-gen-ml-engine` - ML model inference engine  
- ✅ `signal-gen-analytics` - Signal analytics and validation
- ✅ `signal-gen-feature-engine` - Feature engineering pipeline
- ✅ `signal-gen-market-context` - Market context analysis
- ✅ `signal-gen-portfolio` - Portfolio context for signals
- ✅ `signal-gen-risk-mgmt` - Risk assessment for signals

### **Signal Processing (New K8s Services)**
Pattern: `signal-proc-<function>`
- 🔄 `signal-proc-bridge` - **NEW NAME** for `crypto-enhanced-signal-bridge`
  - Function: Converts ML signals into actionable trade recommendations
  - Ports: 8022
  - Technologies: Redis streaming + MySQL fallback

### **Trade Processing (New K8s Services)**  
Pattern: `trade-proc-<function>`
- 🔄 `trade-proc-orchestrator` - **NEW NAME** for `crypto-automated-live-trader`
  - Function: Orchestrates trade recommendation processing and execution
  - Ports: 8023
  - Technologies: Recommendation polling + execution coordination

### **Trade Execution (New K8s Services)**
Pattern: `trade-exec-<function>`
- 🔄 `trade-exec-coinbase` - **NEW NAME** for `aicryptotrading-engines-trade-execution`
  - Function: Direct Coinbase Advanced Trade API integration
  - Ports: 8024
  - Technologies: Coinbase API + portfolio management

### **Infrastructure Services**
Pattern: `infra-<function>`
- 🔄 `infra-redis` - Redis streaming infrastructure
- 🔄 `infra-cors-proxy` - CORS proxy for frontend

## 🔄 **Migration Mapping**

| Current Docker Name | New K8s Name | Category | Function |
|-------------------|-------------|----------|----------|
| `crypto-enhanced-signal-bridge` | `signal-proc-bridge` | Signal Processing | ML signals → Trade recommendations |
| `crypto-automated-live-trader` | `trade-proc-orchestrator` | Trade Processing | Recommendation orchestration |
| `aicryptotrading-engines-trade-execution` | `trade-exec-coinbase` | Trade Execution | Coinbase API integration |
| `crypto-redis` | `infra-redis` | Infrastructure | Redis streaming |

## 📊 **Service Architecture Flow**

```
K8s Signal Generation Services
├── signal-gen-orchestrator
├── signal-gen-ml-engine
├── signal-gen-analytics
├── signal-gen-feature-engine
├── signal-gen-market-context
├── signal-gen-portfolio
└── signal-gen-risk-mgmt
           ↓ (Write signals to database)
K8s Signal Processing Services  
├── signal-proc-bridge ← Reads signals, creates recommendations
           ↓ (REST API calls)
K8s Trade Processing Services
├── trade-proc-orchestrator ← Processes recommendations  
           ↓ (REST API calls)
K8s Trade Execution Services
└── trade-exec-coinbase ← Executes on Coinbase Advanced Trade API
```

## 🎯 **Benefits of New Naming Convention**

1. **Clear Categorization**: Immediately understand service category and function
2. **Consistent Patterns**: All services follow `<category>-<function>` format  
3. **Scalability**: Easy to add new services in each category
4. **Maintainability**: Developers can quickly locate and understand services
5. **Monitoring**: Easier to group and monitor services by category

## 📝 **Implementation Plan**

1. Create K8s manifests with new names
2. Update service discovery URLs 
3. Test inter-service communication
4. Update documentation and monitoring
5. Remove Docker Compose services