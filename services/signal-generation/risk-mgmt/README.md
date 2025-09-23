# Signal Generation Risk Management Service

## Overview
This microservice provides risk controls, selloff protection, and risk adjustments for trading signals. Integrates with market context and trading engine.

- **API Endpoints:**
  - `/assess`: POST, returns risk assessment for a signal
  - `/market_conditions`: GET, returns current market conditions
  - `/protection_status`: GET, returns selloff/recovery status
  - `/health`: GET, service health check

## Environment Variables
- `DATABASE_HOST`, `DATABASE_PORT`, `DATABASE_USER`, `DATABASE_PASSWORD`, `DATABASE_NAME`, `RISK_MGMT_PORT`
- `TRADING_ENGINE_URL`, `MARKET_CONTEXT_URL`

## Usage
Send a POST request to `/assess` with signal details. See API docs for details.

## Maintainer
- Contact: mikeepley2
