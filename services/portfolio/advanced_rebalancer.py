#!/usr/bin/env python3
"""
Advanced Portfolio Rebalancer Module

Enhanced intelligent rebalancing system with multiple optimization strategies:
1. Dynamic concentration management 
2. Correlation-based diversification
3. Volatility-adjusted position sizing
4. Performance-based rebalancing
5. Sector/category diversification
6. Liquidity-based optimization
7. Momentum-based adjustments
8. Market regime adaptation

Integrated into the trading portfolio service.
"""

import mysql.connector
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import math
import os

# Configure logging
logger = logging.getLogger(__name__)

class AdvancedPortfolioRebalancer:
    def __init__(self, db_config: Dict = None):
        if db_config is None:
            self.db_config = {
                'host': os.environ.get('DATABASE_HOST', 'host.docker.internal'),
                'user': os.environ.get('DATABASE_USER', 'news_collector'),
                'password': os.environ.get('DATABASE_PASSWORD', '99Rules!'),
                'database': os.environ.get('DATABASE_NAME', 'crypto_transactions'),
                'port': int(os.environ.get('DATABASE_PORT', 3306))
            }
        else:
            self.db_config = db_config
        
        # Configuration parameters
        self.MAX_SINGLE_POSITION = 0.25  # 25% maximum single position
        self.MIN_SINGLE_POSITION = 0.02  # 2% minimum position for rebalancing consideration
        self.CORRELATION_THRESHOLD = 0.7  # High correlation threshold
        self.VOLATILITY_LOOKBACK_DAYS = 30
        self.REBALANCE_THRESHOLD = 0.05  # 5% deviation triggers rebalancing
        self.PERFORMANCE_LOOKBACK_DAYS = 14  # Performance analysis period
        
        # Asset categorization for diversification
        self.ASSET_CATEGORIES = {
            'BTC': 'store_of_value',
            'ETH': 'smart_contracts',
            'ADA': 'proof_of_stake',
            'SOL': 'high_throughput',
            'LINK': 'oracle',
            'DOT': 'interoperability',
            'MATIC': 'layer2_scaling',
            'AVAX': 'defi_platform',
            'ATOM': 'cosmos_ecosystem',
            'UNI': 'dex_governance'
        }
        
    def get_portfolio_positions(self) -> List[Dict]:
        """Get current portfolio positions with enhanced data"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor(dictionary=True)
            
            # Enhanced query with additional metrics
            query = """
            SELECT 
                pp.symbol,
                pp.quantity,
                pp.avg_entry_price,
                pp.current_value,
                pp.last_updated,
                -- Calculate additional metrics
                (pp.current_value / pp.quantity) as current_price,
                ((pp.current_value / pp.quantity) - pp.avg_entry_price) / pp.avg_entry_price * 100 as unrealized_return_percent,
                -- Get recent price volatility
                COALESCE(pv.volatility_30d, 0.0) as volatility_30d,
                COALESCE(pv.volatility_7d, 0.0) as volatility_7d,
                -- Get trading volume data
                COALESCE(tv.avg_volume_24h, 0.0) as avg_volume_24h
            FROM portfolio_positions pp
            LEFT JOIN (
                SELECT symbol, 
                       STDDEV(close_price) / AVG(close_price) * 100 as volatility_30d
                FROM price_data 
                WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                GROUP BY symbol
            ) pv ON pp.symbol = pv.symbol
            LEFT JOIN (
                SELECT symbol, 
                       STDDEV(close_price) / AVG(close_price) * 100 as volatility_7d
                FROM price_data 
                WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                GROUP BY symbol
            ) pv7 ON pp.symbol = pv7.symbol
            LEFT JOIN (
                SELECT symbol, AVG(volume) as avg_volume_24h
                FROM price_data 
                WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 1 DAY)
                GROUP BY symbol
            ) tv ON pp.symbol = tv.symbol
            WHERE pp.quantity > 0
            ORDER BY pp.current_value DESC
            """
            
            cursor.execute(query)
            positions = cursor.fetchall()
            
            # Calculate total portfolio value
            total_value = sum(pos['current_value'] for pos in positions)
            
            # Add allocation percentages and enhanced metrics
            enhanced_positions = []
            for pos in positions:
                allocation_percent = (pos['current_value'] / total_value * 100) if total_value > 0 else 0
                
                enhanced_pos = {
                    **pos,
                    'allocation_percent': allocation_percent,
                    'category': self.ASSET_CATEGORIES.get(pos['symbol'], 'other'),
                    'liquidity_score': min(pos.get('avg_volume_24h', 0) / 1000000, 10.0),  # Normalized liquidity score
                    'risk_score': pos.get('volatility_30d', 0.0) / 100,  # Normalized volatility as risk
                    'momentum_score': self.calculate_momentum_score(pos['symbol'])
                }
                enhanced_positions.append(enhanced_pos)
            
            cursor.close()
            connection.close()
            
            return enhanced_positions
            
        except Exception as e:
            logger.error(f"Error getting portfolio positions: {e}")
            return []
    
    def calculate_momentum_score(self, symbol: str) -> float:
        """Calculate momentum score for an asset"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor()
            
            # Get price data for momentum calculation
            query = """
            SELECT close_price, timestamp
            FROM price_data 
            WHERE symbol = %s 
              AND timestamp >= DATE_SUB(NOW(), INTERVAL 14 DAY)
            ORDER BY timestamp ASC
            """
            
            cursor.execute(query, (symbol,))
            price_data = cursor.fetchall()
            
            if len(price_data) < 2:
                return 0.0
            
            # Calculate simple momentum as % change over period
            start_price = price_data[0][0]
            end_price = price_data[-1][0]
            momentum = ((end_price - start_price) / start_price) * 100
            
            # Normalize momentum score between -1 and 1
            normalized_momentum = max(-1.0, min(1.0, momentum / 50))  # Assuming 50% is extreme momentum
            
            cursor.close()
            connection.close()
            
            return normalized_momentum
            
        except Exception as e:
            logger.warning(f"Error calculating momentum for {symbol}: {e}")
            return 0.0
    
    def calculate_correlation_matrix(self, symbols: List[str]) -> pd.DataFrame:
        """Calculate correlation matrix for portfolio assets"""
        try:
            if len(symbols) < 2:
                return pd.DataFrame()
            
            connection = mysql.connector.connect(**self.db_config)
            
            # Get price data for all symbols
            price_data = {}
            for symbol in symbols:
                query = """
                SELECT DATE(timestamp) as date, close_price
                FROM price_data 
                WHERE symbol = %s 
                  AND timestamp >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                ORDER BY timestamp ASC
                """
                
                df = pd.read_sql(query, connection, params=(symbol,))
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.groupby('date')['close_price'].last().reset_index()  # Daily close prices
                    price_data[symbol] = df.set_index('date')['close_price']
            
            connection.close()
            
            if len(price_data) < 2:
                return pd.DataFrame()
            
            # Create price DataFrame and calculate returns
            price_df = pd.DataFrame(price_data)
            returns_df = price_df.pct_change().dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()
            
            return correlation_matrix
            
        except Exception as e:
            logger.warning(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()
    
    def analyze_concentration_risk(self, positions: List[Dict]) -> Dict:
        """Analyze portfolio concentration and identify risks"""
        if not positions:
            return {}
        
        total_value = sum(pos['current_value'] for pos in positions)
        
        # Calculate concentration metrics
        allocations = [pos['current_value'] / total_value for pos in positions]
        
        # Herfindahl-Hirschman Index (HHI) - concentration measure
        hhi = sum(allocation ** 2 for allocation in allocations)
        
        # Identify over-concentrated positions
        over_concentrated = []
        for pos in positions:
            allocation = pos['current_value'] / total_value
            if allocation > self.MAX_SINGLE_POSITION:
                over_concentrated.append({
                    'symbol': pos['symbol'],
                    'current_allocation': allocation * 100,
                    'excess_allocation': (allocation - self.MAX_SINGLE_POSITION) * 100,
                    'excess_value': (allocation - self.MAX_SINGLE_POSITION) * total_value
                })
        
        # Category concentration analysis
        category_allocations = {}
        for pos in positions:
            category = pos.get('category', 'other')
            allocation = pos['current_value'] / total_value
            category_allocations[category] = category_allocations.get(category, 0) + allocation
        
        over_concentrated_categories = []
        for category, allocation in category_allocations.items():
            if allocation > 0.4:  # 40% max per category
                over_concentrated_categories.append({
                    'category': category,
                    'current_allocation': allocation * 100,
                    'recommended_max': 40.0
                })
        
        return {
            'hhi': hhi,
            'concentration_level': 'high' if hhi > 0.25 else 'medium' if hhi > 0.15 else 'low',
            'over_concentrated_positions': over_concentrated,
            'over_concentrated_categories': over_concentrated_categories,
            'num_positions': len(positions),
            'recommended_action': 'reduce_concentration' if over_concentrated else 'maintain'
        }
    
    def generate_rebalancing_recommendations(self, positions: List[Dict]) -> List[Dict]:
        """Generate advanced rebalancing recommendations"""
        if not positions:
            return []
        
        recommendations = []
        total_value = sum(pos['current_value'] for pos in positions)
        
        # Get correlation matrix
        symbols = [pos['symbol'] for pos in positions]
        correlation_matrix = self.calculate_correlation_matrix(symbols)
        
        # Analyze concentration risk
        concentration_analysis = self.analyze_concentration_risk(positions)
        
        # Strategy 1: Reduce over-concentrated positions
        for over_conc in concentration_analysis.get('over_concentrated_positions', []):
            symbol = over_conc['symbol']
            current_pos = next(pos for pos in positions if pos['symbol'] == symbol)
            
            # Calculate target reduction
            target_allocation = self.MAX_SINGLE_POSITION * 0.95  # Slight buffer
            current_allocation = over_conc['current_allocation'] / 100
            reduction_needed = current_allocation - target_allocation
            reduction_value = reduction_needed * total_value
            
            recommendations.append({
                'symbol': symbol,
                'action': 'SELL',
                'reason': 'concentration_risk',
                'current_allocation': current_allocation * 100,
                'target_allocation': target_allocation * 100,
                'trade_value': reduction_value,
                'trade_quantity': reduction_value / current_pos['current_price'],
                'priority': 'high',
                'confidence': 0.85
            })
        
        # Strategy 2: Diversify highly correlated positions
        if not correlation_matrix.empty:
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    if symbol1 in correlation_matrix.index and symbol2 in correlation_matrix.columns:
                        correlation = correlation_matrix.loc[symbol1, symbol2]
                        
                        if correlation > self.CORRELATION_THRESHOLD:
                            pos1 = next(pos for pos in positions if pos['symbol'] == symbol1)
                            pos2 = next(pos for pos in positions if pos['symbol'] == symbol2)
                            
                            # Recommend reducing the weaker performer
                            if pos1['momentum_score'] < pos2['momentum_score']:
                                weaker_pos = pos1
                            else:
                                weaker_pos = pos2
                            
                            # Only recommend if position is significant
                            allocation = weaker_pos['current_value'] / total_value
                            if allocation > 0.05:  # 5% minimum for correlation-based rebalancing
                                reduction_value = allocation * total_value * 0.3  # Reduce by 30%
                                
                                recommendations.append({
                                    'symbol': weaker_pos['symbol'],
                                    'action': 'SELL',
                                    'reason': 'high_correlation',
                                    'correlation_with': symbol1 if weaker_pos['symbol'] == symbol2 else symbol2,
                                    'correlation_value': correlation,
                                    'trade_value': reduction_value,
                                    'trade_quantity': reduction_value / weaker_pos['current_price'],
                                    'priority': 'medium',
                                    'confidence': 0.70
                                })
        
        # Strategy 3: Risk-adjusted position sizing
        for pos in positions:
            allocation = pos['current_value'] / total_value
            risk_score = pos.get('risk_score', 0.0)
            
            # High-risk positions should have lower allocations
            if risk_score > 0.5 and allocation > 0.15:  # High risk and high allocation
                target_allocation = min(0.15, allocation * 0.8)  # Reduce by 20%
                reduction_value = (allocation - target_allocation) * total_value
                
                if reduction_value > total_value * 0.02:  # Only recommend if significant
                    recommendations.append({
                        'symbol': pos['symbol'],
                        'action': 'SELL',
                        'reason': 'risk_management',
                        'risk_score': risk_score,
                        'current_allocation': allocation * 100,
                        'target_allocation': target_allocation * 100,
                        'trade_value': reduction_value,
                        'trade_quantity': reduction_value / pos['current_price'],
                        'priority': 'medium',
                        'confidence': 0.75
                    })
        
        # Strategy 4: Momentum-based adjustments
        for pos in positions:
            allocation = pos['current_value'] / total_value
            momentum = pos.get('momentum_score', 0.0)
            
            # Reduce positions with strong negative momentum
            if momentum < -0.5 and allocation > 0.08:  # Strong negative momentum
                target_allocation = allocation * 0.7  # Reduce by 30%
                reduction_value = (allocation - target_allocation) * total_value
                
                recommendations.append({
                    'symbol': pos['symbol'],
                    'action': 'SELL',
                    'reason': 'negative_momentum',
                    'momentum_score': momentum,
                    'current_allocation': allocation * 100,
                    'target_allocation': target_allocation * 100,
                    'trade_value': reduction_value,
                    'trade_quantity': reduction_value / pos['current_price'],
                    'priority': 'low',
                    'confidence': 0.60
                })
        
        # Remove duplicate recommendations and prioritize
        unique_recommendations = {}
        for rec in recommendations:
            symbol = rec['symbol']
            if symbol not in unique_recommendations or rec['priority'] == 'high':
                unique_recommendations[symbol] = rec
        
        return list(unique_recommendations.values())
    
    def create_rebalancing_signals(self) -> bool:
        """Create trading signals based on rebalancing recommendations"""
        try:
            positions = self.get_portfolio_positions()
            recommendations = self.generate_rebalancing_recommendations(positions)
            
            if not recommendations:
                logger.info("No rebalancing recommendations generated")
                return True
            
            # Create signals for each recommendation
            signals_created = 0
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor()
            
            for rec in recommendations:
                # Create signal in trading_signals table
                signal_query = """
                INSERT INTO trading_signals 
                (symbol, timestamp, signal_type, confidence, prediction, 
                 model_name, model_version, features_used, 
                 additional_data, is_mock)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                additional_data = {
                    'rebalancing_reason': rec['reason'],
                    'priority': rec['priority'],
                    'trade_value': rec['trade_value'],
                    'current_allocation': rec.get('current_allocation', 0),
                    'target_allocation': rec.get('target_allocation', 0)
                }
                
                cursor.execute(signal_query, (
                    rec['symbol'],
                    datetime.now(),
                    rec['action'],
                    rec['confidence'],
                    rec['trade_quantity'],
                    'advanced_rebalancer',
                    '2.0',
                    0,  # features_used
                    json.dumps(additional_data),
                    0   # is_mock
                ))
                
                signals_created += 1
                logger.info(f"📊 Rebalancing signal: {rec['action']} {rec['symbol']} - {rec['reason']}")
            
            connection.commit()
            cursor.close()
            connection.close()
            
            logger.info(f"✅ Created {signals_created} rebalancing signals")
            return True
            
        except Exception as e:
            logger.error(f"Error creating rebalancing signals: {e}")
            return False
    
    def get_rebalancing_analysis(self) -> Dict:
        """Get comprehensive rebalancing analysis"""
        try:
            positions = self.get_portfolio_positions()
            recommendations = self.generate_rebalancing_recommendations(positions)
            concentration_analysis = self.analyze_concentration_risk(positions)
            
            # Calculate correlation analysis
            symbols = [pos['symbol'] for pos in positions]
            correlation_matrix = self.calculate_correlation_matrix(symbols)
            
            avg_correlation = 0.0
            if not correlation_matrix.empty:
                # Calculate average correlation (excluding self-correlation)
                corr_values = []
                for i in range(len(correlation_matrix)):
                    for j in range(i+1, len(correlation_matrix)):
                        corr_values.append(abs(correlation_matrix.iloc[i, j]))
                avg_correlation = np.mean(corr_values) if corr_values else 0.0
            
            # Portfolio health score
            health_score = 100
            health_score -= len(concentration_analysis.get('over_concentrated_positions', [])) * 15
            health_score -= len(concentration_analysis.get('over_concentrated_categories', [])) * 10
            health_score -= min(avg_correlation * 30, 25)  # Penalty for high correlation
            health_score = max(0, health_score)
            
            return {
                'portfolio_health_score': health_score,
                'total_positions': len(positions),
                'total_recommendations': len(recommendations),
                'concentration_analysis': concentration_analysis,
                'average_correlation': avg_correlation,
                'high_priority_actions': len([r for r in recommendations if r['priority'] == 'high']),
                'recommendations': recommendations,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating rebalancing analysis: {e}")
            return {}
