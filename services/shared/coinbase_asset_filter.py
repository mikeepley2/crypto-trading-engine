#!/usr/bin/env python3
"""
Asset Filter Configuration for Coinbase Advanced Trade API

This module implements asset filtering to prevent failed trade attempts.
Uses a configuration-based approach to avoid database schema dependencies.

Based on analysis of actual trade rejections and Coinbase API documentation.
"""

import logging
import mysql.connector
from typing import Dict, Set, Optional
from datetime import datetime, timedelta

# Trading thresholds
MINIMUM_TRADE_SIZE_USD = 5.00  # Coinbase minimum to prevent rejections
CLEANUP_THRESHOLD_USD = 5.00   # Threshold for small position cleanup

# Legacy fallback for unsupported assets (now using database columns)
# This is kept as fallback only if database is unavailable
LEGACY_UNSUPPORTED_ASSETS = {
    'RNDR': 'Trading not enabled for RNDR-USD on Coinbase Advanced Trade API - confirmed failure',
    'RENDER': 'Alternative symbol for RNDR - also unsupported',
}

# Database configuration for checking asset existence (read-only)
DB_CONFIG = {
    'host': 'host.docker.internal',
    'user': 'news_collector',
    'password': '99Rules!',
    'database': 'crypto_prices'
}

# Cache for asset validation
_asset_validation_cache = {}
_cache_expiry = None
_cache_duration = timedelta(hours=1)  # Cache for 1 hour

logger = logging.getLogger(__name__)

def _get_database_connection():
    """Get a database connection with error handling"""
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Exception as e:
        logger.warning(f"Database connection failed: {e}")
        return None

def _refresh_asset_cache():
    """Refresh the asset support cache from database using exchange support columns"""
    global _asset_validation_cache, _cache_expiry
    
    conn = _get_database_connection()
    if not conn:
        logger.warning("Using legacy configuration due to database connection failure")
        _asset_validation_cache = dict(LEGACY_UNSUPPORTED_ASSETS)
        _cache_expiry = datetime.now() + _cache_duration
        return
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Get all assets with their exchange support status
        cursor.execute("""
            SELECT symbol, name, is_active, coinbase_supported, exchange_support_updated_at
            FROM crypto_assets 
            WHERE is_active = 1
        """)
        
        assets = cursor.fetchall()
        
        # Build cache of unsupported assets from database
        _asset_validation_cache = {}
        supported_count = 0
        unsupported_count = 0
        
        for asset in assets:
            symbol = asset['symbol']
            if not asset.get('coinbase_supported', False):
                _asset_validation_cache[symbol] = f"Not supported on Coinbase (database: exchange_support_updated_at={asset.get('exchange_support_updated_at', 'N/A')})"
                unsupported_count += 1
            else:
                supported_count += 1
        
        logger.info(f"Loaded exchange support from database: {supported_count} supported, {unsupported_count} unsupported assets")
        
        # Add legacy fallback assets if not in database
        for symbol, reason in LEGACY_UNSUPPORTED_ASSETS.items():
            if symbol not in [asset['symbol'] for asset in assets]:
                _asset_validation_cache[symbol] = f"Legacy fallback: {reason}"
                logger.info(f"Added legacy unsupported asset: {symbol}")
        
        _cache_expiry = datetime.now() + _cache_duration
        
        logger.info(f"Asset filter initialized from database with {len(_asset_validation_cache)} unsupported assets")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Failed to refresh asset cache from database: {e}")
        # Use legacy configuration as fallback
        _asset_validation_cache = dict(LEGACY_UNSUPPORTED_ASSETS)
        _cache_expiry = datetime.now() + _cache_duration
        
        if conn:
            conn.close()

def _ensure_cache_fresh():
    """Ensure the asset cache is fresh"""
    global _cache_expiry
    
    if _cache_expiry is None or datetime.now() > _cache_expiry:
        _refresh_asset_cache()

def get_unsupported_assets() -> Set[str]:
    """Get set of assets not supported by Coinbase Advanced Trade API"""
    _ensure_cache_fresh()
    return set(_asset_validation_cache.keys())

def get_supported_assets() -> Set[str]:
    """Get set of assets supported by Coinbase Advanced Trade API (from database)"""
    _ensure_cache_fresh()
    
    conn = _get_database_connection()
    if not conn:
        logger.warning("Cannot get supported assets - database unavailable")
        return set()
    
    try:
        cursor = conn.cursor()
        # Get assets that are explicitly marked as supported on Coinbase
        cursor.execute("""
            SELECT symbol 
            FROM crypto_assets 
            WHERE is_active = 1 AND coinbase_supported = TRUE
        """)
        supported_symbols = {row[0] for row in cursor.fetchall()}
        
        # Filter out any None or empty symbols for robustness
        supported = {s for s in supported_symbols if s and isinstance(s, str)}
        
        logger.debug(f"Database-filtered supported assets: {len(supported)} symbols")
        
        cursor.close()
        conn.close()
        
        return supported
        
    except Exception as e:
        logger.error(f"Failed to get supported assets from database: {e}")
        try:
            if conn:
                conn.close()
        except Exception:
            pass
        return set()

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
    
    # First check cache for known unsupported assets
    _ensure_cache_fresh()
    if symbol in _asset_validation_cache:
        reason = _asset_validation_cache[symbol]
        logger.debug(f"Asset {symbol} is not supported: {reason}")
        return False
    
    # Check database directly for definitive answer
    conn = _get_database_connection()
    if not conn:
        logger.warning(f"Cannot verify {symbol} support - database unavailable, assuming unsupported")
        return False
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT coinbase_supported 
            FROM crypto_assets 
            WHERE symbol = %s AND is_active = 1
        """, (symbol,))
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if result is None:
            logger.debug(f"Asset {symbol} not found in database, assuming unsupported")
            return False
            
        is_supported = bool(result[0])
        logger.debug(f"Asset {symbol} database support status: {is_supported}")
        return is_supported
        
    except Exception as e:
        logger.error(f"Failed to check {symbol} support in database: {e}")
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        return False

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

def log_filtered_symbol(symbol: str, reason: str = "not supported") -> str:
    """Generate log message for filtered symbols"""
    return f"[ASSET_FILTER] Excluding {symbol}: {reason} on Coinbase Advanced Trade API"

def set_asset_exchange_support(symbol: str, coinbase: bool = None, binance_us: bool = None, kucoin: bool = None) -> bool:
    """
    Update exchange support status for an asset in the database
    
    Args:
        symbol: The cryptocurrency symbol
        coinbase: Whether supported on Coinbase (None = no change)
        binance_us: Whether supported on Binance.US (None = no change) 
        kucoin: Whether supported on KuCoin (None = no change)
        
    Returns:
        bool: True if successful, False if failed
    """
    if not symbol:
        return False
        
    symbol = symbol.upper().strip()
    
    conn = _get_database_connection()
    if not conn:
        logger.error("Cannot update exchange support - database unavailable")
        return False
    
    try:
        cursor = conn.cursor()
        
        # Build dynamic update query
        updates = []
        params = []
        
        if coinbase is not None:
            updates.append("coinbase_supported = %s")
            params.append(coinbase)
            
        if binance_us is not None:
            updates.append("binance_us_supported = %s")
            params.append(binance_us)
            
        if kucoin is not None:
            updates.append("kucoin_supported = %s")
            params.append(kucoin)
        
        if not updates:
            logger.warning(f"No exchange support updates specified for {symbol}")
            cursor.close()
            conn.close()
            return False
        
        updates.append("exchange_support_updated_at = NOW()")
        params.append(symbol)
        
        query = f"""
            UPDATE crypto_assets 
            SET {', '.join(updates)}
            WHERE symbol = %s AND is_active = 1
        """
        
        cursor.execute(query, params)
        rows_affected = cursor.rowcount
        conn.commit()
        
        cursor.close()
        conn.close()
        
        if rows_affected > 0:
            logger.info(f"Updated exchange support for {symbol}: coinbase={coinbase}, binance_us={binance_us}, kucoin={kucoin}")
            # Invalidate cache to force refresh
            global _cache_expiry
            _cache_expiry = None
            return True
        else:
            logger.warning(f"No active asset found with symbol {symbol}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to update exchange support for {symbol}: {e}")
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        return False

def get_exchange_support_status(symbol: str) -> Dict[str, Optional[bool]]:
    """
    Get exchange support status for an asset
    
    Args:
        symbol: The cryptocurrency symbol
        
    Returns:
        Dict with keys: coinbase_supported, binance_us_supported, kucoin_supported, updated_at
    """
    if not symbol:
        return {}
        
    symbol = symbol.upper().strip()
    
    conn = _get_database_connection()
    if not conn:
        logger.error("Cannot get exchange support - database unavailable")
        return {}
    
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT coinbase_supported, binance_us_supported, kucoin_supported, exchange_support_updated_at
            FROM crypto_assets 
            WHERE symbol = %s AND is_active = 1
        """, (symbol,))
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return result or {}
        
    except Exception as e:
        logger.error(f"Failed to get exchange support for {symbol}: {e}")
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        return {}
