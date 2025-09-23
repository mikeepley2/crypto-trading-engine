"""Simple API key authentication dependency for FastAPI services."""
from __future__ import annotations
import os
from fastapi import HTTPException, status, Request
from typing import Optional

API_KEY_ENV = "TRADING_API_KEY"
API_KEY_HEADER = "X-TRADING-API-KEY"

class ApiKeyValidator:
    def __init__(self):
        self._cached_key = os.getenv(API_KEY_ENV)

    def __call__(self, request: Request):  # raw header access avoids Pydantic optional Header type issues
        if not self._cached_key:
            return
        provided = request.headers.get(API_KEY_HEADER)
        if not provided or provided != self._cached_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")

api_key_required = ApiKeyValidator()

__all__ = ["api_key_required", "API_KEY_HEADER"]
