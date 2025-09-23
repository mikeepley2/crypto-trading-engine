#!/bin/bash
# Containerized Test Runner for Trading Services

set -e

echo "🚀 Starting Trading Services Containerized Tests"
echo "================================================"

# Navigate to trading services directory
cd "$(dirname "$0")"

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ docker-compose.yml not found in $(pwd)"
    exit 1
fi

# Set required environment variables
export TRADING_API_KEY="${TRADING_API_KEY:-test-key}"
export DB_HOST="host.docker.internal"
export DB_USER="news_collector"  
export DB_PASSWORD="99Rules!"
export TRADING_DB_NAME="crypto_transactions"
export PRICES_DB_NAME="crypto_prices"

echo "🔧 Environment configured:"
echo "   API Key: ${TRADING_API_KEY}"
echo "   DB Host: ${DB_HOST}"
echo "   DB User: ${DB_USER}"

# Start services
echo ""
echo "🐳 Starting Docker services..."
docker-compose up -d

# Wait a moment for services to initialize
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo ""
echo "🏥 Checking service health..."
for port in 8022 8025 8026 8028; do
    if curl -f "http://localhost:${port}/health" >/dev/null 2>&1; then
        echo "   ✅ Service on port ${port} is healthy"
    else
        echo "   ⚠️  Service on port ${port} not responding"
    fi
done

# Run containerized tests
echo ""
echo "🧪 Running containerized integration tests..."
python -m pytest tests/test_containerized_services.py -v

# Store exit code
TEST_EXIT_CODE=$?

# Show logs if tests failed
if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ Tests failed! Service logs:"
    docker-compose logs --tail=50
fi

# Optional: Keep services running for manual testing
if [ "${KEEP_SERVICES:-}" = "true" ]; then
    echo ""
    echo "🔄 Services will remain running (KEEP_SERVICES=true)"
    echo "   To stop: docker-compose down"
    echo ""
    echo "🌐 Service URLs:"
    echo "   Recommendations: http://localhost:8022"
    echo "   Risk:            http://localhost:8025" 
    echo "   Portfolio:       http://localhost:8026"
    echo "   Signals:         http://localhost:8028"
else
    echo ""
    echo "🧹 Stopping services..."
    docker-compose down
fi

exit $TEST_EXIT_CODE
