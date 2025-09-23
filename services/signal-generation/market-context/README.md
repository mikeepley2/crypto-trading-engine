# Signal Generation Market Context

## Overview
This microservice provides market sentiment, momentum, and regime analysis for trading signals. Integrates with external sentiment and momentum services.

- **API Endpoints:**
  - `/analyze`: POST, returns market context analysis
  - `/sentiment`: GET, returns sentiment data
  - `/momentum`: GET, returns momentum data
  - `/health`: GET, service health check

## Environment Variables
- `DATABASE_HOST`, `DATABASE_PORT`, `DATABASE_USER`, `DATABASE_PASSWORD`, `DATABASE_NAME`, `MARKET_CONTEXT_PORT`
- `SENTIMENT_SERVICE_URL`, `MOMENTUM_DETECTOR_URL`, `MARKET_SELLOFF_SERVICE_URL`

## Usage
Send a POST request to `/analyze` with symbol. See API docs for details.

## Maintainer
- Contact: mikeepley2
