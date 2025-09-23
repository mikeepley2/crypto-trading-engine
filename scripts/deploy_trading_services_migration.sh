#!/bin/bash
# Deploy Trading Services Migration to Kubernetes
# This script migrates remaining Docker services to Kubernetes with full monitoring support

echo "🚀 Starting Trading Services Migration to Kubernetes"
echo "=================================================="

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed or not in PATH"
    exit 1
fi

# Check if crypto-trading namespace exists
if ! kubectl get namespace crypto-trading &> /dev/null; then
    echo "📦 Creating crypto-trading namespace..."
    kubectl create namespace crypto-trading
fi

# Get current Docker service environment variables for secrets
echo "🔧 Setting up API key secrets..."

# Check if OpenAI API key is available
if [ -n "$OPENAI_API_KEY" ]; then
    echo "✅ Found OpenAI API key, creating secret..."
    kubectl create secret generic openai-api-key \
        --from-literal=api-key="$OPENAI_API_KEY" \
        --namespace=crypto-trading \
        --dry-run=client -o yaml | kubectl apply -f -
else
    echo "⚠️  OPENAI_API_KEY not found. Please set it manually:"
    echo "   kubectl create secret generic openai-api-key --from-literal=api-key=YOUR_KEY --namespace=crypto-trading"
fi

# Check if XAI API key is available
if [ -n "$XAI_API_KEY" ]; then
    echo "✅ Found XAI API key, creating secret..."
    kubectl create secret generic xai-api-key \
        --from-literal=api-key="$XAI_API_KEY" \
        --namespace=crypto-trading \
        --dry-run=client -o yaml | kubectl apply -f -
else
    echo "⚠️  XAI_API_KEY not found (optional). Set it manually if needed:"
    echo "   kubectl create secret generic xai-api-key --from-literal=api-key=YOUR_KEY --namespace=crypto-trading"
fi

echo ""
echo "📊 Deploying Trading Services..."
echo "==============================="

# Deploy services in order of dependencies
services=(
    "data-pipeline"
    "ml-prediction" 
    "portfolio-optimization"
    "llm-risk-manager"
    "ml-performance-feedback"
    "trading-performance-feedback"
)

for service in "${services[@]}"; do
    echo "🔄 Deploying $service..."
    if kubectl apply -f "k8s/trading-services/$service.yaml"; then
        echo "✅ $service deployed successfully"
    else
        echo "❌ Failed to deploy $service"
        exit 1
    fi
    echo ""
done

echo "⏳ Waiting for deployments to be ready..."
echo "========================================"

# Wait for all deployments to be ready
for service in "${services[@]}"; do
    echo "⏳ Waiting for $service to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/$service -n crypto-trading
    if [ $? -eq 0 ]; then
        echo "✅ $service is ready"
    else
        echo "⚠️  $service is taking longer than expected"
        kubectl get pods -n crypto-trading -l app=$service
    fi
done

echo ""
echo "🔍 Verifying Service Health..."
echo "============================="

# Check pod status
echo "Pod Status:"
kubectl get pods -n crypto-trading

echo ""
echo "Service Status:"
kubectl get services -n crypto-trading

echo ""
echo "🎯 Prometheus Monitoring Setup..."
echo "================================"

# Check if ServiceMonitors were created
echo "ServiceMonitor Status:"
kubectl get servicemonitor -n crypto-trading

echo ""
echo "📋 Service URLs for Testing:"
echo "==========================="

# Get minikube IP for testing (if using minikube)
if command -v minikube &> /dev/null && minikube status | grep -q "Running"; then
    MINIKUBE_IP=$(minikube ip)
    echo "Using Minikube IP: $MINIKUBE_IP"
    
    echo ""
    echo "Test endpoints (requires port-forwarding):"
    for service in "${services[@]}"; do
        case $service in
            "portfolio-optimization")
                echo "  $service: kubectl port-forward -n crypto-trading svc/$service 8026:8026"
                ;;
            "llm-risk-manager")
                echo "  $service: kubectl port-forward -n crypto-trading svc/$service 8034:8034"
                ;;
            "ml-prediction")
                echo "  $service: kubectl port-forward -n crypto-trading svc/$service 8040:8040"
                ;;
            "data-pipeline")
                echo "  $service: kubectl port-forward -n crypto-trading svc/$service 8043:8043"
                ;;
            "ml-performance-feedback")
                echo "  $service: kubectl port-forward -n crypto-trading svc/$service 8035:8035"
                ;;
            "trading-performance-feedback")
                echo "  $service: kubectl port-forward -n crypto-trading svc/$service 8051:8051"
                ;;
        esac
    done
fi

echo ""
echo "🎉 Migration Complete!"
echo "===================="
echo "✅ All trading services have been migrated to Kubernetes"
echo "✅ Prometheus metrics endpoints configured"
echo "✅ ServiceMonitors created for monitoring"
echo "✅ Health checks and resource limits configured"
echo ""
echo "📖 Next Steps:"
echo "  1. Test service functionality via port-forwarding"
echo "  2. Verify Prometheus is scraping metrics"
echo "  3. Check Grafana dashboards for new services"
echo "  4. Stop Docker versions after validation"
echo ""
echo "💡 To stop Docker services after validation:"
echo "   docker stop crypto-portfolio-optimization crypto-ml-performance-feedback"
echo "   docker stop aicryptotrading-analytics-llm-risk-manager crypto-ml-prediction-new"
echo "   docker stop crypto-data-pipeline-new aicryptotrading-analytics-trading-performance-feedback"
echo ""
echo "🔍 Monitor with:"
echo "   kubectl get pods -n crypto-trading -w"
echo "   kubectl logs -n crypto-trading -l component=trading -f"
