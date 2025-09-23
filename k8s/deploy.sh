#!/bin/bash

# Kubernetes Trading Services Deployment Script
# Deploys signal processing, trade processing, and trade execution services
# with proper naming convention and service dependencies

set -e

NAMESPACE="crypto-trading"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Deploying Trading Services to Kubernetes"
echo "============================================"
echo ""

# Function to check if namespace exists, create if not
ensure_namespace() {
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        echo "📁 Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
    else
        echo "✅ Namespace exists: $NAMESPACE"
    fi
}

# Function to wait for deployment to be ready
wait_for_deployment() {
    local deployment_name=$1
    local timeout=${2:-300}
    
    echo "⏳ Waiting for deployment $deployment_name to be ready..."
    if kubectl wait --for=condition=available --timeout="${timeout}s" deployment/"$deployment_name" -n "$NAMESPACE"; then
        echo "✅ $deployment_name is ready"
    else
        echo "❌ $deployment_name failed to become ready within ${timeout}s"
        return 1
    fi
}

# Function to check service health
check_service_health() {
    local service_name=$1
    local port=$2
    local max_attempts=${3:-30}
    local attempt=1
    
    echo "🏥 Checking health of $service_name on port $port..."
    
    # Port forward in background
    kubectl port-forward -n "$NAMESPACE" service/"$service_name" "$port:$port" >/dev/null 2>&1 &
    local pf_pid=$!
    
    # Give port-forward time to establish
    sleep 3
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://localhost:$port/health" >/dev/null 2>&1; then
            echo "✅ $service_name health check passed"
            kill $pf_pid 2>/dev/null || true
            return 0
        fi
        
        echo "🔄 Attempt $attempt/$max_attempts failed, retrying in 5s..."
        sleep 5
        ((attempt++))
    done
    
    echo "❌ $service_name health check failed after $max_attempts attempts"
    kill $pf_pid 2>/dev/null || true
    return 1
}

# Function to deploy and verify a service
deploy_service() {
    local service_file=$1
    local service_name=$2
    local port=$3
    
    echo ""
    echo "📦 Deploying $service_name..."
    echo "   File: $service_file"
    echo "   Port: $port"
    
    # Apply the manifest
    kubectl apply -f "$SCRIPT_DIR/$service_file"
    
    # Wait for deployment to be ready
    wait_for_deployment "$service_name"
    
    # Check health endpoint if port is provided
    if [ -n "$port" ]; then
        check_service_health "$service_name" "$port"
    fi
}

# Main deployment flow
main() {
    echo "🔍 Pre-deployment checks..."
    
    # Ensure namespace exists
    ensure_namespace
    
    # Check if kubectl is available and connected
    if ! kubectl cluster-info >/dev/null 2>&1; then
        echo "❌ kubectl is not connected to a cluster"
        exit 1
    fi
    
    echo "✅ kubectl connected to cluster"
    echo ""
    
    # Deploy infrastructure services first
    echo "🏗️ PHASE 1: Infrastructure Services"
    echo "===================================="
    deploy_service "infra-redis.yaml" "infra-redis" ""
    
    # Deploy signal processing services
    echo ""
    echo "🔄 PHASE 2: Signal Processing Services"
    echo "======================================"
    deploy_service "signal-proc-bridge.yaml" "signal-proc-bridge" "8022"
    
    # Deploy trade processing services
    echo ""
    echo "🎯 PHASE 3: Trade Processing Services"  
    echo "====================================="
    deploy_service "trade-proc-orchestrator.yaml" "trade-proc-orchestrator" "8023"
    
    # Deploy trade execution services
    echo ""
    echo "💰 PHASE 4: Trade Execution Services"
    echo "====================================="
    deploy_service "trade-exec-coinbase.yaml" "trade-exec-coinbase" "8024"
    
    echo ""
    echo "🎉 DEPLOYMENT COMPLETE!"
    echo "======================"
    
    # Show final status
    echo ""
    echo "📊 Final Service Status:"
    kubectl get pods -n "$NAMESPACE" -l category -o wide
    
    echo ""
    echo "🌐 Service Endpoints:"
    kubectl get services -n "$NAMESPACE" -l category
    
    echo ""
    echo "🔗 Service Architecture:"
    echo "   K8s Signal Generation → signal-proc-bridge:8022 → trade-proc-orchestrator:8023 → trade-exec-coinbase:8024 → Coinbase API"
    
    echo ""
    echo "🏥 Health Check Commands:"
    echo "   kubectl port-forward -n $NAMESPACE service/signal-proc-bridge 8022:8022"
    echo "   kubectl port-forward -n $NAMESPACE service/trade-proc-orchestrator 8023:8023"  
    echo "   kubectl port-forward -n $NAMESPACE service/trade-exec-coinbase 8024:8024"
    
    echo ""
    echo "✅ All trading services deployed with new naming convention!"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "status")
        echo "📊 Trading Services Status:"
        kubectl get pods -n "$NAMESPACE" -l category
        echo ""
        kubectl get services -n "$NAMESPACE" -l category
        ;;
    "clean")
        echo "🧹 Cleaning up trading services..."
        kubectl delete -f "$SCRIPT_DIR/" --ignore-not-found=true
        echo "✅ Cleanup complete"
        ;;
    "help"|"-h"|"--help")
        echo "Trading Services Deployment Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  deploy  - Deploy all trading services (default)"
        echo "  status  - Show current deployment status"  
        echo "  clean   - Remove all trading services"
        echo "  help    - Show this help message"
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac