# Kubernetes Deployment Guide - Signal Generation Microservices

## 🚀 Quick Start

```bash
# Deploy everything to Kubernetes
./deploy_k8s_signal_microservices.sh
```

## 📋 Prerequisites

### Required Tools
- `kubectl` - Kubernetes CLI
- `docker` - Container engine
- Access to a Kubernetes cluster
- Prometheus and Grafana installed in cluster (optional)

### Cluster Requirements
- Kubernetes 1.19+
- Metrics Server installed (for HPA)
- StorageClass available (for model storage)
- Minimum 4GB RAM, 2 CPU cores available

## 🏗️ Architecture Overview

### Services Deployed
- **Orchestrator** (Port 8025) - Main coordination service
- **ML Engine** (Port 8051) - XGBoost model predictions
- **Feature Engine** (Port 8052) - Technical indicator calculations
- **Market Context** (Port 8053) - Sentiment and momentum analysis
- **Portfolio** (Port 8054) - Position sizing logic
- **Risk Management** (Port 8055) - Risk assessment
- **Analytics** (Port 8056) - Performance tracking

### Kubernetes Resources
- **Namespace**: `crypto-trading`
- **Deployments**: 7 microservices
- **Services**: ClusterIP for internal communication
- **ConfigMaps**: Environment configuration
- **Secrets**: Database credentials
- **PVC**: Model storage (5GB)
- **HPA**: Auto-scaling configuration
- **ServiceMonitors**: Prometheus monitoring

## 🔧 Manual Deployment Steps

### 1. Build Docker Images

```bash
cd backend/services/trading

# Build all service images
for service in orchestrator ml-engine feature-engine market-context portfolio risk-mgmt analytics; do
  docker build -f "signal-generation/$service/Dockerfile" \
               -t "crypto-signal-gen-$service:latest" \
               "signal-generation/$service/"
done
```

### 2. Apply Kubernetes Manifests

```bash
# Apply in order
kubectl apply -f k8s/signal-generation/01-orchestrator.yaml
kubectl apply -f k8s/signal-generation/02-ml-engine.yaml
kubectl apply -f k8s/signal-generation/03-feature-market.yaml
kubectl apply -f k8s/signal-generation/04-portfolio-risk-analytics.yaml
kubectl apply -f k8s/signal-generation/05-monitoring.yaml
kubectl apply -f k8s/signal-generation/06-autoscaling.yaml
```

### 3. Verify Deployment

```bash
# Check pod status
kubectl get pods -n crypto-trading

# Check service endpoints
kubectl get svc -n crypto-trading

# Check HPA status
kubectl get hpa -n crypto-trading
```

## 🔍 Monitoring Setup

### Prometheus Configuration

The services are configured with ServiceMonitors for automatic discovery:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: signal-generation-services
  namespace: crypto-trading
spec:
  selector:
    matchLabels:
      service: orchestrator
  endpoints:
  - port: http
    path: /metrics
```

### Grafana Dashboard

Import the dashboard from `k8s/signal-generation/grafana-dashboard.json`:

1. Open Grafana UI
2. Go to Dashboards → Import
3. Upload `grafana-dashboard.json`
4. Configure data source to point to your Prometheus instance

### Key Metrics Monitored

- **Request Rate**: HTTP requests per second per service
- **Latency**: P50, P95, P99 response times
- **Error Rate**: 4xx/5xx error tracking
- **Resource Usage**: CPU and memory utilization
- **Signal Success Rate**: Trading signal generation success
- **Database Operations**: Query performance and connections
- **ML Predictions**: Model inference rates and accuracy

## 🔧 Configuration

### Environment Variables

Set via ConfigMap and Secrets:

```yaml
# ConfigMap
MYSQL_HOST: "mysql.crypto-trading.svc.cluster.local"
MYSQL_USER: "news_collector"
MYSQL_DATABASE: "crypto_prices"
LOG_LEVEL: "INFO"

# Secret
MYSQL_PASSWORD: "99Rules!"
```

### Resource Limits

Default resource allocation per service:

```yaml
resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

ML Engine has higher limits:
```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "1Gi"
    cpu: "1000m"
```

## 📈 Auto-Scaling

HPA is configured for automatic scaling based on:

- **CPU Utilization**: 70% threshold
- **Memory Utilization**: 80% threshold
- **Scale Up**: Maximum 100% increase every 15 seconds
- **Scale Down**: Maximum 10% decrease every 60 seconds

### Scaling Configuration

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: signal-gen-orchestrator-hpa
spec:
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 🧪 Testing

### Health Checks

```bash
# Port forward to orchestrator
kubectl port-forward svc/signal-gen-orchestrator 8025:8025 -n crypto-trading

# Test health endpoint
curl http://localhost:8025/health

# Test metrics endpoint
curl http://localhost:8025/metrics
```

### Signal Generation Test

```bash
# Generate a test signal
curl -X POST http://localhost:8025/generate_signal \
  -H "Content-Type: application/json" \
  -d '{"symbol": "SOL", "price": 100.0}'
```

### Load Testing

```bash
# Install hey for load testing
go install github.com/rakyll/hey@latest

# Run load test
hey -n 1000 -c 10 -m POST \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTC", "price": 50000.0}' \
  http://localhost:8025/generate_signal
```

## 🐛 Troubleshooting

### Common Issues

1. **Pod CrashLoopBackOff**
   ```bash
   kubectl logs -l app=signal-gen-orchestrator -n crypto-trading
   kubectl describe pod <pod-name> -n crypto-trading
   ```

2. **Service Not Reachable**
   ```bash
   kubectl get endpoints -n crypto-trading
   kubectl get svc -n crypto-trading
   ```

3. **Resource Issues**
   ```bash
   kubectl top pods -n crypto-trading
   kubectl describe nodes
   ```

### Debug Commands

```bash
# View all resources
kubectl get all -n crypto-trading

# Check events
kubectl get events -n crypto-trading --sort-by='.lastTimestamp'

# Exec into pod
kubectl exec -it <pod-name> -n crypto-trading -- /bin/bash

# View logs with follow
kubectl logs -f deployment/signal-gen-orchestrator -n crypto-trading
```

## 🔒 Security Considerations

### Network Policies

Implement network policies to restrict pod-to-pod communication:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: signal-generation-policy
  namespace: crypto-trading
spec:
  podSelector:
    matchLabels:
      app: signal-gen-orchestrator
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: trading-client
    ports:
    - protocol: TCP
      port: 8025
```

### RBAC

Create service accounts with minimal required permissions:

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: signal-generation-sa
  namespace: crypto-trading
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: signal-generation-role
  namespace: crypto-trading
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]
```

## 📊 Performance Tuning

### JVM Tuning (if applicable)

```yaml
env:
- name: JAVA_OPTS
  value: "-Xmx512m -Xms256m -XX:+UseG1GC"
```

### Database Connection Pooling

```python
# Optimized connection pool settings
db_config = {
    'pool_size': 10,
    'max_overflow': 20,
    'pool_pre_ping': True,
    'pool_recycle': 3600
}
```

## 🚀 Production Deployment

### Pre-deployment Checklist

- [ ] Resource quotas configured
- [ ] Monitoring and alerting set up
- [ ] Backup strategy for model files
- [ ] Network policies implemented
- [ ] Security scanning completed
- [ ] Load testing performed
- [ ] Disaster recovery plan documented

### Rolling Updates

```bash
# Update image for a service
kubectl set image deployment/signal-gen-orchestrator \
  orchestrator=crypto-signal-gen-orchestrator:v1.1.0 \
  -n crypto-trading

# Watch rollout status
kubectl rollout status deployment/signal-gen-orchestrator -n crypto-trading
```

### Backup and Recovery

```bash
# Export configurations
kubectl get all -n crypto-trading -o yaml > signal-generation-backup.yaml

# Backup model files from PVC
kubectl exec -it <ml-engine-pod> -n crypto-trading -- \
  tar czf /tmp/models-backup.tar.gz /app/models
```

## 📞 Support and Maintenance

### Regular Maintenance Tasks

1. **Monitor resource usage** and adjust limits
2. **Review logs** for errors and performance issues
3. **Update dependencies** and security patches
4. **Scale services** based on load patterns
5. **Backup model files** and configurations

### Emergency Procedures

```bash
# Emergency scale down
kubectl scale deployment --all --replicas=0 -n crypto-trading

# Emergency rollback
kubectl rollout undo deployment/signal-gen-orchestrator -n crypto-trading

# Force pod restart
kubectl delete pods -l app=signal-gen-orchestrator -n crypto-trading
```