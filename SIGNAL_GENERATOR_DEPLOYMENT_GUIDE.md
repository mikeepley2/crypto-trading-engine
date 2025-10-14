# Signal Generator Deployment Guide

## Overview
This guide documents the process for deploying the ML-backed signal generator to Kubernetes without fallback modes.

## Key Requirements
- **NO FALLBACKS**: The signal generator must only work with the ML model
- **Model File**: `balanced_realistic_model_20251005_155755.joblib` (2MB)
- **Dependencies**: All Python packages must be pre-installed
- **Kubernetes**: Kind cluster deployment

## Deployment Process

### 1. Create Docker Image
The signal generator uses a custom Docker image that includes:
- All Python dependencies pre-installed
- The ML model file baked into the image
- The signal generator script

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn mysql-connector-python requests numpy pandas scikit-learn xgboost joblib

# Create models directory
RUN mkdir -p /app/models

# Copy the model file
COPY balanced_realistic_model_20251005_155755.joblib /app/models/

# Copy the signal generator script
COPY working_signal_generator.py /app/

# Expose port
EXPOSE 8025

# Run the signal generator
CMD ["python", "/app/working_signal_generator.py"]
```

### 2. Build and Load Image
```bash
# Build the Docker image
docker build -t signal-generator:latest -f Dockerfile.signal-generator .

# Load the image into the kind cluster
docker save signal-generator:latest | docker exec -i cryptoai-multinode-control-plane ctr -n k8s.io images import -
```

### 3. Kubernetes Deployment
**Key Configuration:**
- `imagePullPolicy: Never` - Use local image
- Environment variables from ConfigMap and Secrets
- Health checks on port 8025
- Resource limits: 200m CPU, 256Mi memory

**YAML:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: signal-generator
  namespace: crypto-trading
spec:
  replicas: 1
  selector:
    matchLabels:
      app: signal-generator
  template:
    metadata:
      labels:
        app: signal-generator
    spec:
      containers:
      - name: signal-generator
        image: signal-generator:latest
        imagePullPolicy: Never
        ports:
        - containerPort: 8025
        envFrom:
        - configMapRef:
            name: crypto-trading-config
        - secretRef:
            name: mysql-credentials
        resources:
          limits:
            cpu: 200m
            memory: 256Mi
          requests:
            cpu: 50m
            memory: 128Mi
        livenessProbe:
          httpGet:
            path: /health
            port: 8025
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8025
          initialDelaySeconds: 10
          periodSeconds: 10
```

### 4. Critical Code Changes
**Remove ALL fallback logic from `working_signal_generator.py`:**

1. **load_model() function:**
   - Remove fallback mode assignment
   - Add model path: `/app/models/balanced_realistic_model_20251005_155755.joblib`
   - Raise RuntimeError if no model found

2. **generate_signal() function:**
   - Remove fallback mode checks
   - Raise RuntimeError if model is None or "fallback"

3. **Signal generation cycle:**
   - Remove fallback mode logic
   - Skip symbols without features instead of using fallback

## Verification Steps

### 1. Check Model Loading
```bash
kubectl logs deployment/signal-generator -n crypto-trading | grep "ML model loaded"
```

### 2. Verify ML Predictions
```bash
kubectl logs deployment/signal-generator -n crypto-trading | grep "Model prediction"
```

### 3. Check Database Records
```bash
kubectl exec deployment/signal-generator -n crypto-trading -- python -c "
import mysql.connector
import os

conn = mysql.connector.connect(
    host=os.getenv('DB_HOST', '172.22.32.1'),
    user=os.getenv('DB_USER', 'news_collector'),
    password=os.getenv('DB_PASSWORD', '99Rules!'),
    database=os.getenv('DB_NAME_PRICES', 'crypto_prices')
)

cursor = conn.cursor()
cursor.execute('''
    SELECT id, symbol, signal_type, confidence, model_version, timestamp 
    FROM trading_signals 
    ORDER BY id DESC 
    LIMIT 5
''')

print('Recent signals:')
for row in cursor.fetchall():
    print(f'{row[0]} | {row[1]} | {row[2]} | {row[3]:.3f} | {row[4]} | {row[5]}')

conn.close()
"
```

**Expected Output:**
- Model version: `xgboost_ml_model` (not `fallback_mode`)
- Confidence scores: Real ML values (0.6-0.8 range)
- Signal types: BUY/SELL based on ML predictions

## Troubleshooting

### Model Not Loading
- Check if model file exists in container: `kubectl exec deployment/signal-generator -n crypto-trading -- ls -la /app/models/`
- Verify image was loaded: `docker exec cryptoai-multinode-control-plane ctr -n k8s.io images list | grep signal-generator`

### Fallback Mode Still Active
- Check logs for "fallback mode" messages
- Verify code changes were applied to `working_signal_generator.py`
- Rebuild and reload Docker image

### No Signals Generated
- Check database connectivity
- Verify `ml_features_materialized` table has recent data
- Check symbol filtering logic

## Rebuild Process
When making changes to the signal generator:

1. **Update code** in `working_signal_generator.py`
2. **Rebuild image**: `docker build -t signal-generator:latest -f Dockerfile.signal-generator .`
3. **Reload image**: `docker save signal-generator:latest | docker exec -i cryptoai-multinode-control-plane ctr -n k8s.io images import -`
4. **Restart deployment**: `kubectl rollout restart deployment/signal-generator -n crypto-trading`
5. **Verify**: Check logs and database records

## Important Notes
- **Never use fallback modes** - the service must fail if ML model is unavailable
- **Model file is 2MB** - too large for ConfigMaps, must be in Docker image
- **Kind cluster** requires manual image loading via `ctr import`
- **Health checks** ensure service is ready before traffic routing
- **Resource limits** prevent memory issues with ML model loading


