# Signal Generation Orchestrator Service

## Overview
This microservice coordinates all signal generation services and maintains backward compatibility with the original API. Handles end-to-end signal generation requests.

- **API Endpoints:**
  - `/generate_signal`: POST, returns trading signal for a symbol
  - `/signal/{symbol}`: GET, returns signal for a symbol
  - `/batch_signals`: POST, returns signals for multiple symbols
  - `/health`: GET, service health check
  - `/service_status`: GET, returns health of all microservices

## Environment Variables
- `DATABASE_HOST`, `DATABASE_PORT`, `DATABASE_USER`, `DATABASE_PASSWORD`, `DATABASE_NAME`, `ORCHESTRATOR_PORT`
- `ML_ENGINE_URL`, `FEATURE_ENGINE_URL`, `MARKET_CONTEXT_URL`, `PORTFOLIO_URL`, `RISK_MGMT_URL`, `ANALYTICS_URL`, `TRADING_ENGINE_URL`

## Usage
Send a POST request to `/generate_signal` with symbol. See API docs for details.

## Maintainer
- Contact: mikeepley2
