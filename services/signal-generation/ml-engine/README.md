# Signal Generation ML Engine

## Overview
This microservice provides XGBoost-based machine learning predictions for trading signals. It loads pre-trained models and exposes REST API endpoints for prediction and model info.

- **API Endpoints:**
  - `/predict`: POST, returns prediction and confidence for input features
  - `/health`: GET, service health check
  - `/model_info`: GET, model metadata

## Environment Variables
- `DATABASE_HOST`, `DATABASE_PORT`, `DATABASE_USER`, `DATABASE_PASSWORD`, `DATABASE_NAME`, `ML_ENGINE_PORT`

## Usage
Send a POST request to `/predict` with features and symbol. See API docs for details.

## Maintainer
- Contact: mikeepley2
