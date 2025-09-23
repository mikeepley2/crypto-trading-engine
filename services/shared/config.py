"""Shared configuration module for trading services.
Centralizes environment variable parsing and provides defaults.
"""
from __future__ import annotations
import os
from functools import lru_cache
from typing import Dict, Any

# Default constants
_DEFAULT_DB_HOST = "localhost"  # Changed from host.docker.internal for direct Python execution
_DEFAULT_DB_USER = "news_collector"
_DEFAULT_DB_PASSWORD = "99Rules!"
_DEFAULT_DB_TRANSACTIONS = "crypto_transactions"
_DEFAULT_DB_PRICES = "crypto_prices"

@lru_cache(maxsize=1)
def get_db_configs() -> Dict[str, Dict[str, Any]]:
    return {
        "transactions": {
            "host": os.getenv("MYSQL_HOST", _DEFAULT_DB_HOST),
            "user": os.getenv("MYSQL_USER", _DEFAULT_DB_USER),
            "password": os.getenv("MYSQL_PASSWORD", _DEFAULT_DB_PASSWORD),
            "database": os.getenv("MYSQL_DATABASE_TRANSACTIONS", _DEFAULT_DB_TRANSACTIONS),
            "port": int(os.getenv("MYSQL_PORT", 3306)),
            "autocommit": True,
        },
        "prices": {
            "host": os.getenv("MYSQL_HOST", _DEFAULT_DB_HOST),
            "user": os.getenv("MYSQL_USER", _DEFAULT_DB_USER),
            "password": os.getenv("MYSQL_PASSWORD", _DEFAULT_DB_PASSWORD),
            "database": os.getenv("MYSQL_DATABASE_PRICES", _DEFAULT_DB_PRICES),
            "port": int(os.getenv("MYSQL_PORT", 3306)),
            "autocommit": True,
        }
    }

@lru_cache(maxsize=1)
def get_trading_limits() -> Dict[str, Any]:
    return {
        "trading_fee_percent": float(os.getenv("TRADING_FEE_PERCENT", 0.5)),
        "min_trade_amount": float(os.getenv("MIN_TRADE_AMOUNT", 5.0)),
        "max_trade_amount": float(os.getenv("MAX_TRADE_AMOUNT", 10000.0)),
        "default_position_size_percent": float(os.getenv("DEFAULT_POSITION_SIZE_PERCENT", 2.0)),
        "max_position_size_percent": float(os.getenv("MAX_POSITION_SIZE_PERCENT", 10.0)),
        "daily_trade_limit": int(os.getenv("DAILY_TRADE_LIMIT", 10)),
        "max_portfolio_allocation": float(os.getenv("MAX_PORTFOLIO_ALLOCATION", 80.0)),
    }

@lru_cache(maxsize=1)
def get_service_urls() -> Dict[str, str]:
    return {
    "portfolio": os.getenv("PORTFOLIO_SERVICE_URL", "http://portfolio-service:8026"),
    "risk": os.getenv("RISK_SERVICE_URL", "http://risk-service:8025"),
        # signals-service deprecated; prefer signal-generation-engine usage directly
        "signals": os.getenv("SIGNALS_SERVICE_URL", ""),
        # mock_engine now routes to aicryptotrading-engines-trade-execution (EXECUTION_MODE=mock)
    "mock_engine": os.getenv("MOCK_TRADING_ENGINE_URL", "http://trade-execution-engine:8024"),
    }

@lru_cache(maxsize=1)
def feature_flags() -> Dict[str, bool]:
    return {
        "enforce_portfolio_service": os.getenv("ENFORCE_PORTFOLIO_SERVICE", "true").lower() == "true",
        "use_risk_service": os.getenv("USE_RISK_SERVICE", "true").lower() == "true",
    # signals-service deprecated
    "use_signals_service": False,
    "enable_prometheus_text": os.getenv("ENABLE_PROMETHEUS_TEXT", "true").lower() == "true",
    }

@lru_cache(maxsize=1)
def api_key() -> str | None:
    return os.getenv("TRADING_API_KEY") or None

__all__ = [
    "get_db_configs",
    "get_trading_limits",
    "get_service_urls",
    "feature_flags",
    "api_key",
]
