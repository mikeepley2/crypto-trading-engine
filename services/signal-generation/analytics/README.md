# Signal Generation Analytics Service

## Overview
This microservice tracks strategy performance, signal metrics, and service health for the trading system. Provides dashboard and reporting data.

- **API Endpoints:**
  - `/analytics`: POST, returns analytics data
  - `/dashboard_data`: GET, returns dashboard summary
  - `/service_health`: GET, returns health of all services
  - `/health`: GET, service health check

## Environment Variables
- `DATABASE_HOST`, `DATABASE_PORT`, `DATABASE_USER`, `DATABASE_PASSWORD`, `DATABASE_NAME`, `ANALYTICS_PORT`
- `ML_ENGINE_URL`, `FEATURE_ENGINE_URL`, `MARKET_CONTEXT_URL`, `PORTFOLIO_URL`, `RISK_MGMT_URL`, `ORCHESTRATOR_URL`, `TRADING_ENGINE_URL`

## Usage
Send a POST request to `/analytics` with time range. See API docs for details.

## Maintainer
- Contact: mikeepley2
