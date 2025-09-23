# Containerized Trading Services Testing

## Overview
This directory contains two types of integration tests:

1. **Unit-style tests** (`test_integrated_trading_services.py`) - Test FastAPI apps directly via Python imports
2. **Containerized tests** (`test_containerized_services.py`) - Test actual Docker containers via HTTP

## Running Unit-style Tests

```bash
# Install basic dependencies
pip install pytest fastapi httpx mysql-connector-python

# Run unit-style tests (faster, no containers needed)
pytest backend/services/trading/tests/test_integrated_trading_services.py -v
```

## Running Containerized Tests

### Prerequisites
1. **Docker & Docker Compose** installed
2. **MySQL running** on Windows host (accessible via `host.docker.internal`)
3. **Environment setup**:
   ```bash
   export TRADING_API_KEY=test-key
   export DB_HOST=host.docker.internal
   export DB_USER=news_collector
   export DB_PASSWORD=99Rules!
   ```

### Install Dependencies
```bash
pip install -r requirements-containerized.txt
```

### Build and Test Services
```bash
# Build Docker images
cd backend/services/trading
docker-compose build

# Run containerized integration tests
pytest tests/test_containerized_services.py -v -s

# Or start services with Kubernetes and test
kubectl apply -f k8s/crypto-trading/
pytest tests/test_containerized_services.py::TestContainerizedServices::test_all_services_health -v
```

## Test Coverage

### Unit-style Tests (`test_integrated_trading_services.py`)
- ✅ Health endpoint responses
- ✅ Metrics endpoint Prometheus format
- ✅ Risk service trade rejection
- ✅ Recommendation status synchronization

### Containerized Tests (`test_containerized_services.py`)
- ✅ All services start and respond to health checks
- ✅ Metrics endpoints accessible via HTTP
- ✅ Risk service trade validation over network
- ✅ Signals service data retrieval
- ✅ Full recommendation CRUD flow
- ✅ Portfolio service operations

## Service Ports
- **Recommendations**: 8022
- **Risk Management**: 8025  
- **Portfolio**: 8026
- **Signals**: 8028

## Troubleshooting

### Services won't start
```bash
# Check container logs
docker-compose logs recommendations
docker-compose logs risk
docker-compose logs signals
docker-compose logs portfolio

# Rebuild images
docker-compose build --no-cache
```

### Database connection issues
- Ensure MySQL is running on Windows host
- Verify `news_collector` user has proper permissions
- Check `host.docker.internal` resolves from containers

### Port conflicts
```bash
# Check what's using the ports
netstat -tulpn | grep -E ':(8022|8025|8026|8028)'

# Stop conflicting services
docker-compose down
```

## Development Workflow

1. **Unit tests first** - Rapid development cycle
2. **Containerized tests** - Integration validation
3. **Manual testing** - Via curl or Postman on running containers

```bash
# Example manual test
curl -H "X-TRADING-API-KEY: test-key" http://localhost:8022/health
curl -H "X-TRADING-API-KEY: test-key" http://localhost:8025/metrics
```
