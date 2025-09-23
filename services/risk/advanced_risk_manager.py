#!/usr/bin/env python3
"""
Advanced Risk Management Module

Implements comprehensive risk management including:
1. Volatility-based position sizing
2. Portfolio heat monitoring
3. Correlation-based adjustments
4. Dynamic risk limits
5. Position concentration limits
6. Market regime risk adjustments

Integrated into the trading services ecosystem.
"""

import numpy as np
import pandas as pd
import mysql.connector
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import json
import os

logger = logging.getLogger(__name__)

class AdvancedRiskManager:
    def __init__(self, db_config: Dict = None):
        if db_config is None:
            self.db_config = {
                'host': os.environ.get('DATABASE_HOST', 'host.docker.internal'),
                'user': os.environ.get('DATABASE_USER', 'news_collector'),
                'password': os.environ.get('DATABASE_PASSWORD', '99Rules!'),
                'database': 'crypto_prices',  # Use prices DB for historical data
                'port': int(os.environ.get('DATABASE_PORT', 3306))
            }
        else:
            self.db_config = db_config
        
        # Risk parameters
        self.MAX_PORTFOLIO_HEAT = 0.15  # Max 15% portfolio at risk
        self.BASE_POSITION_SIZE = 200.0  # Base position size in USD
        self.VOLATILITY_LOOKBACK = 14  # Days for volatility calculation
        self.CORRELATION_THRESHOLD = 0.7  # Max correlation before reducing positions
        self.MAX_POSITION_SIZE = 0.25  # Max 25% in any single position
        self.MIN_POSITION_SIZE = 25.0   # Minimum position size in USD
        
        # Market regime adjustments
        self.REGIME_MULTIPLIERS = {
            'bull_market': 1.2,     # Increase position sizes in bull markets
            'bear_market': 0.6,     # Reduce position sizes in bear markets
            'high_volatility': 0.7, # Reduce positions during high volatility
            'sideways': 1.0         # Normal sizing in sideways markets
        }
    
    def calculate_optimal_position_size(self, symbol: str, base_size: float, 
                                      current_positions: Dict = None, 
                                      market_regime: str = 'sideways') -> Dict:
        """Calculate optimal position size considering multiple risk factors"""
        try:
            if current_positions is None:
                current_positions = {}
            
            # Start with base size
            adjusted_size = base_size
            adjustments = {'base_size': base_size}
            
            # 1. Volatility adjustment
            volatility_multiplier = self.get_volatility_adjustment(symbol)
            adjusted_size *= volatility_multiplier
            adjustments['volatility_multiplier'] = volatility_multiplier
            
            # 2. Correlation adjustment
            correlation_multiplier = self.get_correlation_adjustment(symbol, current_positions)
            adjusted_size *= correlation_multiplier
            adjustments['correlation_multiplier'] = correlation_multiplier
            
            # 3. Portfolio heat adjustment
            portfolio_heat = self.calculate_portfolio_heat(current_positions)
            heat_multiplier = self.get_heat_adjustment(portfolio_heat)
            adjusted_size *= heat_multiplier
            adjustments['portfolio_heat'] = portfolio_heat
            adjustments['heat_multiplier'] = heat_multiplier
            
            # 4. Market regime adjustment
            regime_multiplier = self.REGIME_MULTIPLIERS.get(market_regime, 1.0)
            adjusted_size *= regime_multiplier
            adjustments['regime_multiplier'] = regime_multiplier
            
            # 5. Position concentration limits
            total_portfolio_value = sum(pos.get('value_usd', 0) for pos in current_positions.values())
            if total_portfolio_value > 0:
                max_position_value = total_portfolio_value * self.MAX_POSITION_SIZE
                adjusted_size = min(adjusted_size, max_position_value)
            
            # 6. Minimum and maximum bounds
            adjusted_size = max(self.MIN_POSITION_SIZE, min(adjusted_size, base_size * 3.0))
            adjustments['final_size'] = adjusted_size
            
            return {
                'optimal_size': adjusted_size,
                'adjustments': adjustments,
                'risk_level': self.categorize_risk_level(adjusted_size, base_size),
                'warnings': self.generate_risk_warnings(symbol, adjusted_size, adjustments)
            }
            
        except Exception as e:
            logger.error(f"Error calculating optimal position size for {symbol}: {e}")
            return {
                'optimal_size': base_size,
                'adjustments': {'base_size': base_size, 'error': str(e)},
                'risk_level': 'unknown',
                'warnings': [f"Error in risk calculation: {e}"]
            }
    
    def get_volatility_adjustment(self, symbol: str) -> float:
        """Calculate volatility-based position size multiplier"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            
            # Get historical price data for volatility calculation
            query = """
            SELECT current_price, timestamp_iso
            FROM ml_features_materialized 
            WHERE symbol = %s 
            AND timestamp_iso >= DATE_SUB(NOW(), INTERVAL %s DAY)
            ORDER BY timestamp_iso ASC
            """
            
            df = pd.read_sql(query, connection, params=[symbol, self.VOLATILITY_LOOKBACK])
            connection.close()
            
            if len(df) < 5:
                logger.warning(f"Insufficient data for volatility calculation: {symbol}")
                return 0.8  # Conservative default
            
            # Calculate daily returns and volatility
            df['returns'] = df['current_price'].pct_change().dropna()
            volatility = df['returns'].std() * np.sqrt(24)  # Annualized volatility
            
            # Volatility-based adjustment
            # High volatility = smaller position, Low volatility = larger position
            volatility_multiplier = 1.0 / (1.0 + volatility * 10)  # Adjust factor
            volatility_multiplier = max(0.3, min(2.0, volatility_multiplier))  # Cap between 30%-200%
            
            logger.debug(f"📊 {symbol} volatility: {volatility:.3f}, multiplier: {volatility_multiplier:.3f}")
            return volatility_multiplier
            
        except Exception as e:
            logger.error(f"Error calculating volatility adjustment for {symbol}: {e}")
            return 0.8  # Conservative default
    
    def get_correlation_adjustment(self, symbol: str, current_positions: Dict) -> float:
        """Calculate correlation-based position size adjustment"""
        try:
            if not current_positions:
                return 1.0
            
            connection = mysql.connector.connect(**self.db_config)
            
            # Get correlation with existing positions
            correlations = []
            for existing_symbol in current_positions.keys():
                if existing_symbol == symbol:
                    continue
                
                correlation = self.calculate_asset_correlation(symbol, existing_symbol, connection)
                if correlation > self.CORRELATION_THRESHOLD:
                    correlations.append(correlation)
            
            connection.close()
            
            if not correlations:
                return 1.0
            
            # Reduce position size based on highest correlation
            max_correlation = max(correlations)
            correlation_adjustment = 1.0 - (max_correlation - self.CORRELATION_THRESHOLD) * 2
            correlation_adjustment = max(0.2, correlation_adjustment)  # Minimum 20% position
            
            logger.debug(f"📊 {symbol} max correlation: {max_correlation:.3f}, adjustment: {correlation_adjustment:.3f}")
            return correlation_adjustment
            
        except Exception as e:
            logger.error(f"Error calculating correlation adjustment for {symbol}: {e}")
            return 1.0
    
    def calculate_asset_correlation(self, symbol1: str, symbol2: str, connection) -> float:
        """Calculate correlation between two assets"""
        try:
            # Get price data for both symbols
            query = """
            SELECT DATE(timestamp_iso) as date, symbol, current_price
            FROM ml_features_materialized 
            WHERE symbol IN (%s, %s)
            AND timestamp_iso >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            ORDER BY timestamp_iso ASC
            """
            
            df = pd.read_sql(query, connection, params=[symbol1, symbol2])
            
            if df.empty:
                return 0.0
            
            # Pivot to get price series for each symbol
            price_data = df.pivot(index='date', columns='symbol', values='current_price')
            
            if symbol1 not in price_data.columns or symbol2 not in price_data.columns:
                return 0.0
            
            # Calculate returns
            returns = price_data.pct_change().dropna()
            
            if len(returns) < 5:
                return 0.0
            
            # Calculate correlation
            correlation = returns[symbol1].corr(returns[symbol2])
            return abs(correlation) if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating correlation between {symbol1} and {symbol2}: {e}")
            return 0.0
    
    def calculate_portfolio_heat(self, positions: Dict) -> float:
        """Calculate current portfolio heat (total risk exposure)"""
        try:
            total_value = sum(pos.get('value_usd', 0) for pos in positions.values())
            if total_value == 0:
                return 0.0
            
            portfolio_heat = 0.0
            
            for symbol, position in positions.items():
                position_value = position.get('value_usd', 0)
                position_weight = position_value / total_value
                
                # Get volatility for this position
                volatility = self.get_asset_volatility(symbol)
                
                # Risk contribution = position_weight * volatility
                position_risk = position_weight * volatility
                portfolio_heat += position_risk
            
            return portfolio_heat
            
        except Exception as e:
            logger.error(f"Error calculating portfolio heat: {e}")
            return 0.0
    
    def get_asset_volatility(self, symbol: str) -> float:
        """Get recent volatility for an asset"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            
            query = """
            SELECT current_price 
            FROM ml_features_materialized 
            WHERE symbol = %s 
            AND timestamp_iso >= DATE_SUB(NOW(), INTERVAL 14 DAY)
            ORDER BY timestamp_iso ASC
            """
            
            df = pd.read_sql(query, connection, params=[symbol])
            connection.close()
            
            if len(df) < 5:
                return 0.3  # Default volatility
            
            returns = df['current_price'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(24)  # Annualized
            
            return min(2.0, max(0.1, volatility))  # Cap between 10%-200%
            
        except Exception as e:
            logger.error(f"Error getting volatility for {symbol}: {e}")
            return 0.3
    
    def get_heat_adjustment(self, portfolio_heat: float) -> float:
        """Calculate position size adjustment based on portfolio heat"""
        if portfolio_heat <= self.MAX_PORTFOLIO_HEAT * 0.5:
            return 1.2  # Low heat, can increase positions
        elif portfolio_heat <= self.MAX_PORTFOLIO_HEAT * 0.8:
            return 1.0  # Normal heat
        elif portfolio_heat <= self.MAX_PORTFOLIO_HEAT:
            return 0.8  # High heat, reduce positions
        else:
            return 0.5  # Critical heat, significantly reduce positions
    
    def categorize_risk_level(self, final_size: float, base_size: float) -> str:
        """Categorize the risk level of the final position size"""
        ratio = final_size / base_size
        
        if ratio >= 1.5:
            return 'high'
        elif ratio >= 1.1:
            return 'elevated'
        elif ratio >= 0.9:
            return 'normal'
        elif ratio >= 0.6:
            return 'reduced'
        else:
            return 'conservative'
    
    def generate_risk_warnings(self, symbol: str, final_size: float, adjustments: Dict) -> List[str]:
        """Generate risk warnings based on adjustments"""
        warnings = []
        
        # Check for significant volatility adjustment
        vol_mult = adjustments.get('volatility_multiplier', 1.0)
        if vol_mult < 0.5:
            warnings.append(f"High volatility detected for {symbol}, position size reduced significantly")
        
        # Check for correlation issues
        corr_mult = adjustments.get('correlation_multiplier', 1.0)
        if corr_mult < 0.8:
            warnings.append(f"High correlation with existing positions detected for {symbol}")
        
        # Check for portfolio heat
        portfolio_heat = adjustments.get('portfolio_heat', 0.0)
        if portfolio_heat > self.MAX_PORTFOLIO_HEAT * 0.8:
            warnings.append(f"Portfolio heat elevated ({portfolio_heat:.1%}), reducing position sizes")
        
        # Check for very small positions
        if final_size < self.MIN_POSITION_SIZE * 1.5:
            warnings.append(f"Position size for {symbol} is very small, consider skipping")
        
        return warnings
    
    def get_portfolio_risk_metrics(self, positions: Dict) -> Dict:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            if not positions:
                return {
                    'portfolio_heat': 0.0,
                    'concentration_risk': 0.0,
                    'correlation_risk': 0.0,
                    'risk_level': 'low',
                    'recommendations': []
                }
            
            total_value = sum(pos.get('value_usd', 0) for pos in positions.values())
            portfolio_heat = self.calculate_portfolio_heat(positions)
            
            # Calculate concentration risk (Herfindahl index)
            allocations = [pos.get('value_usd', 0) / total_value for pos in positions.values() if total_value > 0]
            concentration_risk = sum(allocation ** 2 for allocation in allocations)
            
            # Calculate average correlation
            symbols = list(positions.keys())
            correlation_risk = self.calculate_average_correlation(symbols)
            
            # Overall risk level
            risk_score = (portfolio_heat * 0.4 + concentration_risk * 0.3 + correlation_risk * 0.3)
            
            if risk_score >= 0.8:
                risk_level = 'critical'
            elif risk_score >= 0.6:
                risk_level = 'high'
            elif risk_score >= 0.4:
                risk_level = 'moderate'
            elif risk_score >= 0.2:
                risk_level = 'low'
            else:
                risk_level = 'very_low'
            
            # Generate recommendations
            recommendations = []
            if portfolio_heat > self.MAX_PORTFOLIO_HEAT:
                recommendations.append("Reduce position sizes to lower portfolio heat")
            if concentration_risk > 0.5:
                recommendations.append("Diversify holdings to reduce concentration risk")
            if correlation_risk > 0.7:
                recommendations.append("Reduce correlation by diversifying across asset categories")
            
            return {
                'portfolio_heat': portfolio_heat,
                'concentration_risk': concentration_risk,
                'correlation_risk': correlation_risk,
                'risk_score': risk_score,
                'risk_level': risk_level,
                'recommendations': recommendations,
                'total_positions': len(positions),
                'total_value': total_value
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk metrics: {e}")
            return {
                'portfolio_heat': 0.0,
                'concentration_risk': 0.0,
                'correlation_risk': 0.0,
                'risk_level': 'unknown',
                'recommendations': [f"Error calculating risk metrics: {e}"]
            }
    
    def calculate_average_correlation(self, symbols: List[str]) -> float:
        """Calculate average correlation across portfolio assets"""
        try:
            if len(symbols) < 2:
                return 0.0
            
            connection = mysql.connector.connect(**self.db_config)
            correlations = []
            
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    correlation = self.calculate_asset_correlation(symbol1, symbol2, connection)
                    if not np.isnan(correlation):
                        correlations.append(abs(correlation))
            
            connection.close()
            
            return np.mean(correlations) if correlations else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating average correlation: {e}")
            return 0.0
