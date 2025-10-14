# 🔄 Crypto Trading Pipeline Flow Diagram

## **ASCII Flow Diagram**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Market Data   │───▶│ Enhanced Signal  │───▶│   Signal Bridge │
│   Collection    │    │   Generator      │    │   (Port 8022)   │
│                 │    │   (Port 8025)    │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Database:     │    │   ML Model:      │    │   Database:     │
│ crypto_prices   │    │ balanced_model   │    │ trade_recommend │
│                 │    │ .joblib (51 feat)│    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Ollama LLM    │◀───│ LLM Validation   │◀───│ Trade Orchestr. │
│   Service       │    │   Service        │    │   (Port 8023)   │
│ (Port 11434)    │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Coinbase      │◀───│ Trade Executor   │◀───│ Risk Management │
│   API (Real)    │    │   Real           │    │   Service       │
│                 │    │ (Port 8024)      │    │   (Port 8027)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
                                               ┌─────────────────┐
                                               │ Health Monitor  │
                                               │   (Port 8030)   │
                                               │                 │
                                               └─────────────────┘
```

## **Detailed Data Flow**

### **Step 1: Signal Generation**
```
Market Data → ML Model → Trading Signals → Database
     ↓              ↓           ↓            ↓
crypto_prices → 51 features → BUY/SELL → trading_signals
```

### **Step 2: Recommendation Creation**
```
Trading Signals → Signal Bridge → Trade Recommendations → Database
       ↓              ↓                    ↓                ↓
   BUY/SELL → Convert to trades → $100 positions → trade_recommendations
```

### **Step 3: LLM Validation**
```
Trade Recommendations → Ollama LLM → Validation Results → Database
         ↓                    ↓              ↓              ↓
    $100 trades → AI Analysis → APPROVE/REJECT → llm_validation
```

### **Step 4: Risk Assessment**
```
Validated Trades → Risk Management → Risk Assessment → Trade Executor
        ↓                ↓                ↓                ↓
   APPROVED → Volatility Analysis → Risk Score → Adjusted Amount
```

### **Step 5: Trade Execution**
```
Risk-Approved Trades → Trade Executor → Coinbase API → Real Trades
         ↓                    ↓              ↓            ↓
   $100 adjusted → JWT Auth → Live Orders → Cryptocurrency
```

## **Service Communication Flow**

```
┌─────────────────────────────────────────────────────────────────┐
│                    KUBERNETES CLUSTER                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Enhanced    │  │ Signal      │  │ Trade       │            │
│  │ Signal Gen  │──│ Bridge      │──│ Orchestrator│            │
│  │ :8025       │  │ :8022       │  │ :8023       │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│         │                │                │                   │
│         ▼                ▼                ▼                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Database    │  │ Database    │  │ Risk Mgmt   │            │
│  │ crypto_     │  │ trade_      │  │ Service     │            │
│  │ prices      │  │ recommend   │  │ :8027       │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                 │                              │
│                                 ▼                              │
│                          ┌─────────────┐                      │
│                          │ Trade       │                      │
│                          │ Executor    │                      │
│                          │ Real :8024  │                      │
│                          └─────────────┘                      │
│                                 │                              │
│                                 ▼                              │
│                          ┌─────────────┐                      │
│                          │ Coinbase    │                      │
│                          │ API (Live)  │                      │
│                          └─────────────┘                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## **Key Data Transformations**

### **Signal Generation Process**
```
Raw Market Data → Feature Engineering → ML Prediction → Trading Signal
     ↓                    ↓                    ↓              ↓
Price/Volume → 51 Technical Features → Confidence Score → BUY/SELL/HOLD
```

### **Risk Management Process**
```
Trade Request → Volatility Analysis → Portfolio Heat → Risk Score → Decision
     ↓                ↓                    ↓              ↓          ↓
$100 ETH BUY → 14-day volatility → Current exposure → 0.3 score → APPROVE
```

### **Trade Execution Process**
```
Approved Trade → JWT Token → Coinbase Order → Execution Result → Database Update
     ↓              ↓              ↓              ↓              ↓
$100 ETH BUY → Auth Header → Market Order → Order ID → EXECUTED status
```

## **Performance Metrics Flow**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 480 signals │───▶│ 6,665+ recs │───▶│ 494 LLM val │───▶│ Real trades │
│ per hour    │    │ generated   │    │ in 24h      │    │ executed    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## **Error Handling Flow**

```
Service Error → Health Check → Alert → Logging → Recovery
     ↓              ↓           ↓        ↓          ↓
API Failure → /health endpoint → Log → Database → Retry
```

## **Configuration Flow**

```
Environment Variables → Kubernetes ConfigMaps → Service Configuration → Runtime
         ↓                        ↓                      ↓              ↓
DB_HOST, API_KEY → crypto-trading-config → Service startup → Active service
```


