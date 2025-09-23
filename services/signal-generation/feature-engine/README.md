# Signal Generation Feature Engine

## Overview
This microservice computes technical indicators, time features, and market features for trading signals. It is designed for high-throughput feature engineering.

- **API Endpoints:**
  - `/engineer_features`: POST, returns engineered features for a symbol
  - `/health`: GET, service health check
  - `/feature_list`: GET, returns supported features

## Environment Variables
- `DATABASE_HOST`, `DATABASE_PORT`, `DATABASE_USER`, `DATABASE_PASSWORD`, `DATABASE_NAME`, `FEATURE_ENGINE_PORT`

## Usage
Send a POST request to `/engineer_features` with symbol and price. See API docs for details.

## Maintainer
- Contact: mikeepley2
