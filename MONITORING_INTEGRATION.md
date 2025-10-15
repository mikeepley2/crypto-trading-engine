# Crypto Trading Engine - Monitoring Integration Guide

## Overview

The Crypto Trading Engine integrates with the centralized monitoring infrastructure provided by the `crypto-monitoring` project. This integration provides comprehensive observability across all trading services using Prometheus, Loki, and Grafana.

## Architecture

### Monitoring Stack Components

- **Prometheus**: Metrics collection and storage
- **Loki**: Log aggregation and storage  
- **Promtail**: Log collection agent
- **Grafana**: Visualization and dashboards

### Service Discovery

All crypto-trading services are automatically discovered by Prometheus using Kubernetes service annotations:

```yaml
metadata:
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8025"
    prometheus.io/path: "/metrics"
```

## Exposed Metrics

### Signal Generator Service (Port 8025)

- `signals_generated_total{symbol, signal_type}` - Counter of generated signals
- `signal_confidence` - Histogram of signal confidence scores
- `model_inference_time_seconds` - Histogram of ML model inference times
- `model_load_status` - Gauge indicating if ML model is loaded (1=loaded, 0=not loaded)
- `database_query_time_seconds` - Histogram of database query times

### Trade Executor Service (Port 8024)

- `trade_executions_total{status, symbol}` - Counter of trade executions by status
- `trade_execution_time_seconds` - Histogram of trade execution latency
- `coinbase_api_response_time_seconds` - Histogram of Coinbase API response times
- `order_precision_errors_total{symbol}` - Counter of order precision errors
- `balance_check_failures_total{symbol}` - Counter of balance check failures
- `pending_recommendations` - Gauge of pending trade recommendations

### LLM Validation Service (Port 8050)

- `llm_validations_total{decision}` - Counter of LLM validation decisions
- `llm_validation_time_seconds` - Histogram of validation processing time
- `llm_confidence_scores` - Histogram of LLM confidence scores
- `ollama_api_latency_seconds` - Histogram of Ollama API response times

### Trade Orchestrator Service (Port 8023)

- `recommendations_processed_total{status}` - Counter of processed recommendations
- `pipeline_stage_duration_seconds{stage}` - Histogram of pipeline stage durations
- `pending_recommendations_queue_size` - Gauge of queue size
- `orchestrator_cycle_duration_seconds` - Histogram of orchestrator cycle times

### Risk Management Service (Port 8027)

- `risk_assessments_total{decision}` - Counter of risk assessment decisions
- `risk_assessment_time_seconds` - Histogram of risk assessment processing time
- `risk_scores` - Histogram of calculated risk scores
- `rejected_trades_by_risk_total{risk_level}` - Counter of trades rejected by risk level

## Accessing Dashboards

### Grafana Access

1. **Port Forward to Grafana**:
   ```bash
   kubectl port-forward -n crypto-monitoring service/grafana 3000:3000
   ```

2. **Access Grafana**:
   - URL: `http://localhost:3000`
   - Username: `admin`
   - Password: `admin123`

### Available Dashboards

#### 1. Crypto Trading Engine Dashboard
- **Purpose**: Comprehensive monitoring of all trading engine services
- **Location**: `http://localhost:3000/d/crypto-trading-engine`
- **Features**:
  - Service health status
  - Signal generation metrics
  - Trade execution performance
  - LLM validation statistics
  - Pipeline flow analysis
  - Integrated log viewer

#### 2. Unified Crypto System Dashboard
- **Purpose**: Cross-system monitoring across all namespaces
- **Location**: `http://localhost:3000/d/unified-crypto-system`
- **Features**:
  - All namespaces health overview
  - Data collection monitoring
  - Trading engine status
  - Infrastructure metrics
  - Alert and anomaly detection
  - Unified log aggregation

## Log Aggregation

### Log Collection

All crypto-trading pods automatically have their logs collected by Promtail and stored in Loki. Logs are tagged with:

- `namespace`: `crypto-trading`
- `pod`: Pod name
- `container`: Container name
- `app`: Application label

### Log Queries in Grafana

Use the integrated Loki log viewer in dashboards or query directly:

```logql
{namespace="crypto-trading"} |= "ERROR"
{namespace="crypto-trading", app="signal-generator-working"} |= "signal"
{namespace="crypto-trading", app="trade-executor-real"} |= "trade"
```

## Troubleshooting

### Common Issues

#### 1. Metrics Not Appearing

**Symptoms**: Services show as "DOWN" in Grafana
**Solutions**:
- Verify service annotations are present
- Check if `/metrics` endpoint is accessible
- Ensure `prometheus_client` is installed in service containers

#### 2. Logs Not Appearing

**Symptoms**: No logs in Loki log viewer
**Solutions**:
- Verify Promtail is running: `kubectl get pods -n crypto-monitoring | grep promtail`
- Check Promtail logs: `kubectl logs -n crypto-monitoring -l app=promtail`
- Ensure pods have proper labels

#### 3. Dashboard Not Loading

**Symptoms**: Dashboard shows "No data" or fails to load
**Solutions**:
- Verify Prometheus is scraping targets: `http://localhost:9090/targets`
- Check datasource configuration in Grafana
- Ensure metric names match dashboard queries

### Verification Commands

#### Check Service Health
```bash
# Check all crypto-trading services
kubectl get pods -n crypto-trading

# Check service annotations
kubectl get services -n crypto-trading -o yaml | grep -A 5 annotations
```

#### Check Metrics Endpoints
```bash
# Test metrics endpoint
kubectl port-forward -n crypto-trading service/signal-generator-working 8025:8025
curl http://localhost:8025/metrics

# Check Prometheus targets
kubectl port-forward -n crypto-monitoring service/prometheus 9090:9090
# Visit http://localhost:9090/targets
```

#### Check Log Collection
```bash
# Check Promtail status
kubectl get pods -n crypto-monitoring | grep promtail

# Check Promtail logs
kubectl logs -n crypto-monitoring -l app=promtail --tail=50
```

## Service Discovery Mechanism

### How It Works

1. **Service Annotations**: Services are annotated with Prometheus scrape configuration
2. **Kubernetes SD**: Prometheus uses Kubernetes service discovery to find annotated services
3. **Automatic Scraping**: Prometheus automatically scrapes metrics from discovered services
4. **Label Propagation**: Kubernetes labels are automatically added to metrics

### Configuration Details

The Prometheus configuration in `crypto-monitoring` includes:

```yaml
scrape_configs:
  - job_name: 'kubernetes-services-metrics'
    kubernetes_sd_configs:
      - role: service
        namespaces:
          names:
            - crypto-collectors
            - crypto-monitoring
            - crypto-trading  # Our namespace
```

## Integration Benefits

### Unified Observability
- Single monitoring infrastructure for all crypto systems
- Consistent metrics and logging across services
- Centralized alerting and notification

### Professional-Grade Monitoring
- Industry-standard tools (Prometheus, Grafana, Loki)
- Scalable and production-ready
- Rich visualization and analysis capabilities

### Operational Efficiency
- Automated service discovery
- Real-time health monitoring
- Historical data analysis and trending
- Integrated log analysis and debugging

## Next Steps

1. **Deploy Updated Services**: Apply the updated Kubernetes configurations
2. **Verify Integration**: Check that metrics and logs are being collected
3. **Customize Dashboards**: Modify dashboards for specific monitoring needs
4. **Set Up Alerts**: Configure alerting rules for critical metrics
5. **Monitor Performance**: Use dashboards to monitor system performance and health

## Related Documentation

- [Crypto Monitoring Setup Guide](../crypto-monitoring/k8s/MONITORING_SETUP_GUIDE.md)
- [Unified Monitoring Architecture](UNIFIED_MONITORING_ARCHITECTURE.md)
- [Kubernetes Cluster Architecture](KUBERNETES_CLUSTER_ARCHITECTURE_DOCUMENTATION.md)
