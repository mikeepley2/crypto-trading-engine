#!/bin/bash
# Deploy Signal Generation Microservices to Kubernetes
# This script deploys the complete microservices architecture with monitoring

set -e

echo "🚀 Deploying Signal Generation Microservices to Kubernetes..."

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="$SCRIPT_DIR/k8s/signal-generation"
NAMESPACE="crypto-trading"
DOCKER_REGISTRY="localhost:5000"  # Change this to your registry

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
print_status "Checking prerequisites..."

if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed or not in PATH"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check if kubectl can connect to cluster
if ! kubectl cluster-info &> /dev/null; then
    print_error "Cannot connect to Kubernetes cluster"
    exit 1
fi

print_success "Prerequisites check passed"

# Build and tag Docker images
print_status "Building Docker images..."

services=(
    "orchestrator"
    "ml-engine"
    "feature-engine"
    "market-context"
    "portfolio"
    "risk-mgmt"
    "analytics"
)

cd "$SCRIPT_DIR/backend/services/trading"

for service in "${services[@]}"; do
    print_status "Building $service..."
    
    # Build the image
    docker build -f "signal-generation/$service/Dockerfile" \
                 -t "crypto-signal-gen-$service:latest" \
                 "signal-generation/$service/"
    
    # Tag for registry if using one
    if [ "$DOCKER_REGISTRY" != "localhost:5000" ]; then
        docker tag "crypto-signal-gen-$service:latest" \
                   "$DOCKER_REGISTRY/crypto-signal-gen-$service:latest"
        
        print_status "Pushing $service to registry..."
        docker push "$DOCKER_REGISTRY/crypto-signal-gen-$service:latest"
    fi
    
    print_success "Built $service"
done

# Apply Kubernetes manifests
print_status "Applying Kubernetes manifests..."

cd "$SCRIPT_DIR"

# Apply in order
manifests=(
    "01-orchestrator.yaml"
    "02-ml-engine.yaml"
    "03-feature-market.yaml"
    "04-portfolio-risk-analytics.yaml"
    "05-monitoring.yaml"
    "06-autoscaling.yaml"
)

for manifest in "${manifests[@]}"; do
    print_status "Applying $manifest..."
    kubectl apply -f "$K8S_DIR/$manifest"
done

print_success "All manifests applied"

# Wait for deployments to be ready
print_status "Waiting for deployments to be ready..."

deployments=(
    "signal-gen-orchestrator"
    "signal-gen-ml-engine"
    "signal-gen-feature-engine"
    "signal-gen-market-context"
    "signal-gen-portfolio"
    "signal-gen-risk-mgmt"
    "signal-gen-analytics"
)

for deployment in "${deployments[@]}"; do
    print_status "Waiting for $deployment..."
    kubectl wait --for=condition=available --timeout=300s \
                 deployment/"$deployment" -n "$NAMESPACE"
    print_success "$deployment is ready"
done

# Verify pod status
print_status "Checking pod status..."
kubectl get pods -n "$NAMESPACE" -l app=signal-gen-orchestrator

# Test the orchestrator service
print_status "Testing orchestrator service..."

# Port forward for testing
kubectl port-forward svc/signal-gen-orchestrator 8025:8025 -n "$NAMESPACE" &
PORT_FORWARD_PID=$!

# Wait a moment for port forward to establish
sleep 5

# Test health endpoint
if curl -f -s "http://localhost:8025/health" > /dev/null; then
    print_success "Orchestrator health check passed"
else
    print_warning "Orchestrator health check failed"
fi

# Test signal generation
print_status "Testing signal generation..."
test_response=$(curl -s -X POST "http://localhost:8025/generate_signal" \
    -H "Content-Type: application/json" \
    -d '{"symbol": "SOL", "price": 100.0}' || echo "ERROR")

if [[ "$test_response" == *"signal"* ]]; then
    print_success "Signal generation test passed"
else
    print_warning "Signal generation test failed. Response: $test_response"
fi

# Clean up port forward
kill $PORT_FORWARD_PID 2>/dev/null || true

# Show service information
print_status "Deployment Summary:"
echo ""
echo "📊 Services deployed:"
kubectl get services -n "$NAMESPACE" -l app | grep signal-gen

echo ""
echo "🚀 Pods running:"
kubectl get pods -n "$NAMESPACE" -l app | grep signal-gen

echo ""
echo "📈 HPA status:"
kubectl get hpa -n "$NAMESPACE"

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "📋 Useful commands:"
echo "  View pods:         kubectl get pods -n $NAMESPACE"
echo "  View services:     kubectl get svc -n $NAMESPACE"
echo "  View logs:         kubectl logs -l app=signal-gen-orchestrator -n $NAMESPACE"
echo "  Scale deployment:  kubectl scale deployment signal-gen-orchestrator --replicas=3 -n $NAMESPACE"
echo "  Port forward:      kubectl port-forward svc/signal-gen-orchestrator 8025:8025 -n $NAMESPACE"
echo ""
echo "🔗 Access URLs (after port forwarding):"
echo "  Orchestrator:      http://localhost:8025"
echo "  Health Check:      http://localhost:8025/health"
echo "  Metrics:           http://localhost:8025/metrics"
echo "  Generate Signal:   POST http://localhost:8025/generate_signal"
echo ""
echo "📊 Monitoring:"
echo "  - Prometheus metrics are exposed on /metrics endpoints"
echo "  - ServiceMonitors are configured for automatic discovery"
echo "  - HPA is configured for auto-scaling based on CPU/Memory"
echo "  - Loki-compatible JSON logging is enabled"