#!/usr/bin/env python3
"""
Symbol Standardization Utility
Provides consistent symbol format handling across all trading services
"""

import re
import logging
from typing import List, Dict, Optional, Set

logger = logging.getLogger(__name__)

class SymbolStandardizer:
    """
    Centralizes all symbol format handling for the trading system
    
    Key Rules:
    1. Database Storage: Always use base symbols (BTC, ETH, DOGE)
    2. Coinbase API: Always use trading pairs (BTC-USD, ETH-USD) 
    3. Signal Processing: Normalize all inputs to base symbols
    4. Portfolio Queries: Handle both formats seamlessly
    """
    
    # Known cryptocurrency symbols - expand as needed
    SUPPORTED_SYMBOLS = {
        'BTC', 'ETH', 'DOGE', 'ADA', 'SOL', 'LINK', 'AVAX', 'DOT', 'UNI', 'XRP',
        'LTC', 'BCH', 'ETC', 'ATOM', 'MATIC', 'ALGO', 'VET', 'FIL', 'TRX', 'ICP',
        'AAVE', 'MKR', 'CRV', 'COMP', 'SNX', 'SUSHI', 'YFI', 'UMA', 'BAL', 'REN',
        'ZRX', 'ENJ', 'MANA', 'SAND', 'GALA', 'CHZ', 'BAT', 'THETA', 'ZIL', 'HOT',
        'USDC', 'USDT', 'DAI', 'BUSD'  # Stablecoins
    }
    
    # Common trading pair suffixes
    TRADING_SUFFIXES = {'-USD', '-USDT', '-USDC', '-BTC', '-ETH'}
    
    @classmethod
    def normalize_symbol(cls, symbol: str) -> str:
        """
        Normalize any symbol format to base symbol for database storage
        
        Examples:
        - 'BTC-USD' -> 'BTC'
        - 'ETH-USDT' -> 'ETH'  
        - 'BTC' -> 'BTC'
        - 'btc' -> 'BTC'
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        
        # Convert to uppercase
        symbol = symbol.upper().strip()
        
        # Remove common trading pair suffixes
        for suffix in cls.TRADING_SUFFIXES:
            if symbol.endswith(suffix):
                base_symbol = symbol[:-len(suffix)]
                if base_symbol in cls.SUPPORTED_SYMBOLS:
                    return base_symbol
                # If base isn't recognized, fall through to validation
                break
        
        # Validate the symbol
        if symbol not in cls.SUPPORTED_SYMBOLS:
            logger.warning(f"Unknown symbol: {symbol}. Adding to supported list.")
            cls.SUPPORTED_SYMBOLS.add(symbol)
        
        return symbol
    
    @classmethod
    def to_coinbase_product_id(cls, symbol: str, quote_currency: str = 'USD') -> str:
        """
        Convert base symbol to Coinbase trading pair format
        
        Examples:
        - 'BTC' -> 'BTC-USD'
        - 'BTC-USD' -> 'BTC-USD' (idempotent)
        - 'ETH' -> 'ETH-USD'
        """
        base_symbol = cls.normalize_symbol(symbol)
        return f"{base_symbol}-{quote_currency}"
    
    @classmethod
    def from_coinbase_product_id(cls, product_id: str) -> str:
        """
        Extract base symbol from Coinbase trading pair
        
        Examples:
        - 'BTC-USD' -> 'BTC'
        - 'ETH-USDT' -> 'ETH'
        """
        return cls.normalize_symbol(product_id)
    
    @classmethod
    def standardize_portfolio_symbols(cls, positions: List[Dict]) -> List[Dict]:
        """
        Standardize symbol formats in portfolio position data
        
        Args:
            positions: List of position dictionaries with 'currency' or 'symbol' field
            
        Returns:
            List with normalized base symbols
        """
        standardized_positions = []
        
        for position in positions:
            if isinstance(position, dict):
                # Handle different field names
                symbol_field = None
                if 'currency' in position:
                    symbol_field = 'currency'
                elif 'symbol' in position:
                    symbol_field = 'symbol'
                elif 'asset' in position:
                    symbol_field = 'asset'
                
                if symbol_field and position[symbol_field]:
                    # Create copy and normalize symbol
                    normalized_position = position.copy()
                    normalized_position[symbol_field] = cls.normalize_symbol(position[symbol_field])
                    standardized_positions.append(normalized_position)
                else:
                    # Keep original if no recognizable symbol field
                    standardized_positions.append(position)
            else:
                standardized_positions.append(position)
        
        return standardized_positions
    
    @classmethod
    def validate_symbol(cls, symbol: str) -> bool:
        """
        Check if symbol is valid and supported
        """
        try:
            normalized = cls.normalize_symbol(symbol)
            return normalized in cls.SUPPORTED_SYMBOLS
        except ValueError:
            return False
    
    @classmethod
    def get_supported_symbols(cls) -> Set[str]:
        """
        Get set of all supported base symbols
        """
        return cls.SUPPORTED_SYMBOLS.copy()
    
    @classmethod
    def batch_normalize_symbols(cls, symbols: List[str]) -> List[str]:
        """
        Normalize a list of symbols
        """
        return [cls.normalize_symbol(symbol) for symbol in symbols if symbol]
    
    @classmethod
    def is_base_symbol(cls, symbol: str) -> bool:
        """
        Check if symbol is already in base format (no trading pair suffix)
        """
        if not symbol:
            return False
        
        symbol = symbol.upper().strip()
        
        # Check if it has any trading pair suffix
        for suffix in cls.TRADING_SUFFIXES:
            if symbol.endswith(suffix):
                return False
        
        return True
    
    @classmethod
    def standardize_signal_data(cls, signal: Dict) -> Dict:
        """
        Standardize symbol format in signal data dictionary
        """
        if not isinstance(signal, dict):
            return signal
        
        standardized_signal = signal.copy()
        
        # Normalize symbol field if present
        if 'symbol' in standardized_signal and standardized_signal['symbol']:
            standardized_signal['symbol'] = cls.normalize_symbol(standardized_signal['symbol'])
        
        return standardized_signal
    
    @classmethod
    def standardize_trade_data(cls, trade: Dict) -> Dict:
        """
        Standardize symbol format in trade data dictionary
        """
        if not isinstance(trade, dict):
            return trade
        
        standardized_trade = trade.copy()
        
        # Normalize symbol field if present
        if 'symbol' in standardized_trade and standardized_trade['symbol']:
            standardized_trade['symbol'] = cls.normalize_symbol(standardized_trade['symbol'])
        
        return standardized_trade


# Global instance for easy access
symbol_standardizer = SymbolStandardizer()


def normalize_symbol(symbol: str) -> str:
    """Convenience function for quick symbol normalization"""
    return symbol_standardizer.normalize_symbol(symbol)


def to_coinbase_format(symbol: str) -> str:
    """Convenience function for Coinbase API format"""
    return symbol_standardizer.to_coinbase_product_id(symbol)


def standardize_portfolio(positions: List[Dict]) -> List[Dict]:
    """Convenience function for portfolio standardization"""
    return symbol_standardizer.standardize_portfolio_symbols(positions)


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_symbols = [
        'BTC', 'BTC-USD', 'btc', 'btc-usd',
        'ETH', 'ETH-USD', 'eth-usdt',
        'DOGE', 'DOGE-USD', 'doge',
        'InvalidSymbol-USD', 'XYZ'
    ]
    
    print("Symbol Normalization Tests:")
    print("-" * 50)
    
    for symbol in test_symbols:
        try:
            normalized = normalize_symbol(symbol)
            coinbase_format = to_coinbase_format(symbol)
            print(f"{symbol:15} -> {normalized:8} -> {coinbase_format}")
        except Exception as e:
            print(f"{symbol:15} -> ERROR: {e}")
    
    print("\nPortfolio Standardization Test:")
    print("-" * 50)
    
    test_portfolio = [
        {'currency': 'BTC-USD', 'balance': 0.5, 'value_usd': 50000},
        {'currency': 'ETH', 'balance': 2.0, 'value_usd': 6000},
        {'symbol': 'DOGE-USD', 'balance': 1000, 'value_usd': 100}
    ]
    
    standardized_portfolio = standardize_portfolio(test_portfolio)
    for position in standardized_portfolio:
        print(position)