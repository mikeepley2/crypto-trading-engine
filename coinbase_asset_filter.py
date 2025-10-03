#!/usr/bin/env python3
"""
Asset Filter Configuration for Coinbase Advanced Trade API

This module implements asset filtering to prevent failed trade attempts.
Uses a configuration-based approach to avoid database schema dependencies.

Based on analysis of actual trade rejections and Coinbase API documentation.
"""

import logging
from typing import Set

# Trading thresholds
MINIMUM_TRADE_SIZE_USD = 5.00  # Coinbase minimum to prevent rejections
CLEANUP_THRESHOLD_USD = 5.00   # Threshold for small position cleanup

# Known unsupported assets on Coinbase Advanced Trade API
UNSUPPORTED_ASSETS = {
    'RNDR': 'Trading not enabled for RNDR-USD on Coinbase Advanced Trade API - confirmed failure',
    'RENDER': 'Alternative symbol for RNDR - also unsupported',
}

logger = logging.getLogger(__name__)

def is_asset_supported(symbol: str) -> bool:
    """
    Check if an asset is supported for trading on Coinbase Advanced Trade API
    
    Args:
        symbol: The cryptocurrency symbol (e.g., 'BTC', 'ETH', 'RNDR')
        
    Returns:
        bool: True if supported, False if not supported
    """
    if not symbol:
        return False
        
    symbol = symbol.upper().strip()
    
    # Check against known unsupported assets
    if symbol in UNSUPPORTED_ASSETS:
        logger.debug(f"Asset {symbol} is not supported: {UNSUPPORTED_ASSETS[symbol]}")
        return False
    
    # Default to supported for all other assets
    return True

def get_unsupported_assets() -> Set[str]:
    """Get set of assets not supported by Coinbase Advanced Trade API"""
    return set(UNSUPPORTED_ASSETS.keys())

def get_trade_size_minimum() -> float:
    """Get the minimum trade size for Coinbase"""
    return MINIMUM_TRADE_SIZE_USD

def should_trade_amount(amount_usd: float) -> bool:
    """Check if trade amount meets minimum requirements"""
    return amount_usd >= MINIMUM_TRADE_SIZE_USD

def get_cleanup_threshold() -> float:
    """Get the threshold for small position cleanup"""
    return CLEANUP_THRESHOLD_USD

def filter_unsupported_symbols(symbols: list) -> list:
    """Filter out unsupported symbols from a list"""
    return [symbol for symbol in symbols if is_asset_supported(symbol)]
