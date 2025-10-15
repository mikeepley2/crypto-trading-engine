# Unified Monitoring Architecture - Crypto Trading System

## Overview

This document describes the unified monitoring architecture that provides comprehensive observability across the entire crypto trading ecosystem, including data collection, trading engine, and monitoring infrastructure.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           UNIFIED MONITORING ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   DATA COLLECTION NODE    │    │   TRADING ENGINE NODE    │    │   ANALYTICS NODE    │    │   CONTROL PLANE    │
│  (crypto-collectors)      │    │   (crypto-trading)       │    │  (crypto-monitoring) │    │                   │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         │                       │                       │                       │
         ▼                       ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ • Data Collectors │    │ • Signal Generator │    │ • Prometheus    │    │ • Kubernetes API │
│ • Technical Ind.  │    │ • Trade Executor   │    │ • Grafana       │    │ • Cluster Mgmt   │
│ • Sentiment      │    │ • LLM Validation   │    │ • Loki          │    │ • Service Disc.  │
│ • Market Data    │    │ • Risk Management  │    │ • Promtail      │    │ • Resource Mgmt  │
│ • Health Monitor │    │ • Trade Orchestr.  │    │ • Alertmanager  │    │ • Network Policy │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         │                       │                       │                       │
         └───────────────────────┼───────────────────────┼───────────────────────┘
                                 │                       │
                                 ▼                       ▼
                    ┌─────────────────────────────────────────┐
                    │           MONITORING FLOW               │
                    └─────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────────────────────┐
                    │         PROMETHEUS METRICS             │
                    │  • Service Discovery (K8s SD)          │
                    │  • Metrics Collection (/metrics)       │
                    │  • Time Series Storage                 │
                    │  • Query Engine (PromQL)               │
                    └─────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────────────────────┐
                    │           LOKI LOGS                    │
                    │  • Log Aggregation                     │
                    │  • Promtail Collection                 │
                    │  • LogQL Queries                       │
                    │  • Log Storage                         │
                    └─────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────────────────────┐
                    │           GRAFANA DASHBOARDS           │
                    │  • Crypto Trading Engine Dashboard     │
                    │  • Unified System Dashboard            │
                    │  • Real-time Visualization             │
                    │  • Alert Management                    │
                    └─────────────────────────────────────────┘
```

## Service Distribution

### 1. Data Collection Node (crypto-collectors)
**Purpose**: Market data collection and preprocessing
**Services**:
- Data collectors for various exchanges
- Technical indicators calculation
- Sentiment analysis services
- Market data aggregation
- Health monitoring

**Metrics Exposed**:
- Data collection rates
- Data freshness timestamps
- Collection error rates
- API response times

### 2. Trading Engine Node (crypto-trading)
**Purpose**: Core trading logic and execution
**Services**:
- Signal Generator (ML-based)
- Trade Executor (Coinbase integration)
- LLM Validation Service
- Risk Management Service
- Trade Orchestrator

**Metrics Exposed**:
- Signal generation rates and confidence
- Trade execution success/failure rates
- LLM validation decisions and timing
- Risk assessment scores
- Pipeline processing times

### 3. Analytics Node (crypto-monitoring)
**Purpose**: Monitoring and observability infrastructure
**Services**:
- Prometheus (metrics collection)
- Grafana (visualization)
- Loki (log aggregation)
- Promtail (log collection)
- Alertmanager (alerting)

**Metrics Exposed**:
- Infrastructure resource usage
- Service health status
- Alert firing rates
- Dashboard performance

### 4. Control Plane
**Purpose**: Kubernetes cluster management
**Services**:
- Kubernetes API server
- etcd (cluster state)
- Scheduler
- Controller manager
- DNS and networking

## Monitoring Flow

### 1. Metrics Collection Flow
```
Services → /metrics endpoint → Prometheus → Grafana Dashboards
```

**Process**:
1. Services expose metrics on `/metrics` endpoint
2. Prometheus discovers services via Kubernetes service discovery
3. Prometheus scrapes metrics at configured intervals
4. Metrics are stored in Prometheus time-series database
5. Grafana queries Prometheus for dashboard visualization

### 2. Log Collection Flow
```
Pods → Promtail → Loki → Grafana Log Viewer
```

**Process**:
1. Promtail runs as DaemonSet on all nodes
2. Promtail collects logs from pod volumes
3. Logs are enriched with Kubernetes metadata
4. Logs are sent to Loki for storage
5. Grafana queries Loki for log visualization

### 3. Service Discovery Flow
```
Kubernetes Services → Prometheus SD → Automatic Scraping
```

**Process**:
1. Services are annotated with Prometheus scrape configuration
2. Prometheus uses Kubernetes service discovery
3. Services are automatically added to scrape targets
4. Metrics collection begins without manual configuration

## Dashboard Organization

### 1. Crypto Trading Engine Dashboard
**Scope**: crypto-trading namespace only
**Panels**:
- **System Overview**: Service health, signal counts, success rates
- **Signal Generation**: ML model performance, confidence distributions
- **Trade Execution**: Order success rates, execution latency
- **LLM Validation**: Decision rates, processing times
- **Pipeline Flow**: End-to-end processing analysis
- **Logs**: Real-time log viewer

### 2. Unified System Dashboard
**Scope**: All crypto namespaces
**Panels**:
- **Cross-System Health**: All services across all namespaces
- **Data Collection**: Collection rates and data freshness
- **Trading Engine**: Signal generation and execution status
- **Infrastructure**: Resource utilization and performance
- **Alerts & Anomalies**: Active alerts and error trends
- **Unified Logs**: Logs from all systems

## Configuration Details

### Prometheus Configuration
```yaml
scrape_configs:
  - job_name: 'kubernetes-services-metrics'
    kubernetes_sd_configs:
      - role: service
        namespaces:
          names:
            - crypto-collectors
            - crypto-monitoring
            - crypto-trading
```

### Service Annotations
```yaml
metadata:
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8025"
    prometheus.io/path: "/metrics"
```

### Promtail Configuration
```yaml
scrape_configs:
  - job_name: kubernetes-pods
    kubernetes_sd_configs:
      - role: pod
    # Automatically discovers all pods in all namespaces
```

## Cross-Namespace Monitoring Relationships

### Data Flow Dependencies
1. **crypto-collectors** → **crypto-trading**: Market data feeds trading signals
2. **crypto-trading** → **crypto-monitoring**: Trading metrics feed monitoring dashboards
3. **crypto-monitoring** → **All namespaces**: Monitoring infrastructure serves all systems

### Metric Relationships
- **Data Collection Metrics** → **Signal Generation Performance**
- **Signal Generation Metrics** → **Trade Execution Success**
- **Trade Execution Metrics** → **Risk Management Decisions**
- **All Service Metrics** → **Infrastructure Resource Planning**

## Access Patterns

### Development Teams
- **Trading Team**: Crypto Trading Engine Dashboard
- **Data Team**: Data Collection metrics in Unified Dashboard
- **DevOps Team**: Infrastructure metrics and alerts
- **Management**: Unified System Dashboard for overall health

### Operational Access
- **Real-time Monitoring**: Grafana dashboards with 30s refresh
- **Historical Analysis**: Prometheus query interface
- **Log Analysis**: Loki log queries in Grafana
- **Alert Response**: Alertmanager webhook notifications

## Benefits of Unified Architecture

### 1. Single Source of Truth
- All metrics and logs in one place
- Consistent data formats and collection
- Unified query interface (PromQL/LogQL)

### 2. Operational Efficiency
- Automated service discovery
- Centralized configuration management
- Single monitoring infrastructure to maintain

### 3. Professional-Grade Observability
- Industry-standard tools (Prometheus, Grafana, Loki)
- Scalable and production-ready
- Rich visualization and analysis capabilities

### 4. Cross-System Visibility
- End-to-end pipeline monitoring
- Dependency tracking between services
- Holistic system health understanding

## Maintenance and Operations

### Regular Tasks
- Monitor dashboard performance and data freshness
- Review and tune alerting rules
- Update dashboard queries as services evolve
- Scale monitoring infrastructure as needed

### Troubleshooting Workflow
1. Check Unified System Dashboard for overall health
2. Drill down to specific service dashboard
3. Use integrated log viewer for detailed debugging
4. Query Prometheus directly for custom analysis

### Scaling Considerations
- Prometheus storage retention policies
- Loki log retention and storage
- Grafana dashboard performance optimization
- Alert rule complexity and firing rates

## Future Enhancements

### Planned Improvements
- Custom alerting rules for trading-specific metrics
- Advanced anomaly detection using ML
- Integration with external notification systems
- Performance optimization for large-scale deployments

### Extensibility
- Easy addition of new services and namespaces
- Custom dashboard creation for specific use cases
- Integration with external monitoring tools
- Support for additional data sources and exporters
