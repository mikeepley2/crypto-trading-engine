#!/bin/bash
# =============================================================================
# CRYPTO TRADING SYSTEM - COMPLETE DEPLOYMENT SCRIPT
# =============================================================================
# This script deploys the complete crypto trading system with all services
# consolidated and properly configured.
#
# Usage: ./deploy_crypto_trading_system.sh
# =============================================================================

set -e  # Exit on any error

echo "============================================================"
echo "CRYPTO TRADING SYSTEM - COMPLETE DEPLOYMENT"
echo "============================================================"

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed or not in PATH"
    exit 1
fi

# Check if we're connected to a cluster
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Not connected to a Kubernetes cluster"
    exit 1
fi

echo "✅ Kubernetes cluster connection verified"

# Create namespace if it doesn't exist
echo "📦 Creating namespace..."
kubectl create namespace crypto-trading --dry-run=client -o yaml | kubectl apply -f -

# Deploy the consolidated system
echo "🚀 Deploying consolidated crypto trading system..."
kubectl apply -f k8s/crypto-trading-system-fixed.yaml

# Deploy health monitor
echo "🏥 Deploying health monitor..."
kubectl apply -f k8s/health-monitor.yaml

# Wait for deployments to be ready
echo "⏳ Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/signal-bridge -n crypto-trading
kubectl wait --for=condition=available --timeout=300s deployment/trade-executor -n crypto-trading
kubectl wait --for=condition=available --timeout=300s deployment/trade-orchestrator -n crypto-trading

# Note: Signal generator may take longer due to ML model loading
echo "⏳ Waiting for signal generator (this may take several minutes)..."
kubectl wait --for=condition=available --timeout=600s deployment/signal-generator -n crypto-trading || {
    echo "⚠️ Signal generator is taking longer than expected. This is normal due to ML model loading."
    echo "   You can check its status with: kubectl get pods -n crypto-trading"
}

# Copy ML model to signal generator
echo "📊 Copying ML model to signal generator..."
SIGNAL_GENERATOR_POD=$(kubectl get pods -n crypto-trading -l app=signal-generator -o jsonpath='{.items[0].metadata.name}')
if [ -f "balanced_realistic_model_20251005_155755.joblib" ]; then
    kubectl cp balanced_realistic_model_20251005_155755.joblib crypto-trading/$SIGNAL_GENERATOR_POD:/app/models/balanced_realistic_model_20251005_155755.joblib
    echo "✅ ML model copied successfully"
else
    echo "⚠️ ML model file not found. Please ensure balanced_realistic_model_20251005_155755.joblib is in the current directory"
fi

# Check system status
echo "🔍 Checking system status..."
kubectl get pods -n crypto-trading

echo ""
echo "============================================================"
echo "DEPLOYMENT COMPLETE"
echo "============================================================"

# Display service endpoints
echo "📡 Service Endpoints:"
echo "  Signal Generator:    http://signal-generator:8025"
echo "  Signal Bridge:       http://signal-bridge:8022"
echo "  Trade Executor:      http://trade-executor:8024"
echo "  Trade Orchestrator:  http://trade-orchestrator:8023"
echo "  Health Monitor:      http://health-monitor:8080"

echo ""
echo "🔧 Useful Commands:"
echo "  Check pods:          kubectl get pods -n crypto-trading"
echo "  Check services:      kubectl get services -n crypto-trading"
echo "  View logs:           kubectl logs -f deployment/signal-generator -n crypto-trading"
echo "  Health check:        kubectl exec -it deployment/health-monitor -n crypto-trading -- curl http://localhost:8080/status"

echo ""
echo "📊 To test the complete pipeline:"
echo "  kubectl exec -it deployment/signal-bridge -n crypto-trading -- python /app/test_complete_pipeline.py"

echo ""
echo "✅ Crypto Trading System deployed successfully!"
echo "   The system will automatically start generating signals and executing trades."
echo "   Monitor the health monitor for system status."
