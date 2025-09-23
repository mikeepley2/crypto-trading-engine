#!/usr/bin/env python3
"""
Dynamic Position Manager
Handles intelligent position rebalancing for strong signals and dynamic position sizing
"""

import logging
import mysql.connector
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

logger = logging.getLogger(__name__)

class DynamicPositionManager:
    def __init__(self, trades_db_config: Dict, signals_db_config: Dict, trading_engine_url: str):
        self.trades_db_config = trades_db_config
        self.signals_db_config = signals_db_config
        self.trading_engine_url = trading_engine_url
        
        # Default configuration - will be loaded from database
        self.config = {
            'enable_dynamic_rebalancing': True,
            'strong_signal_threshold': 0.85,  # Signals above this are considered "strong"
            'very_strong_signal_threshold': 0.95,  # Signals above this get maximum position scaling
            'max_position_multiplier': 2.0,  # Strong signals can be up to 2x normal position size
            'weak_position_score_threshold': 0.3,  # Positions below this score are candidates for selling
            'rebalancing_lookback_hours': 72,  # Hours to look back for position performance
            'min_position_age_hours': 4,  # Minimum age before a position can be sold for rebalancing
            'max_rebalancing_percentage': 50.0,  # Maximum % of portfolio to rebalance at once
            'position_strength_weights': {
                'recent_performance': 0.4,  # 40% based on recent P&L
                'signal_confidence': 0.3,   # 30% based on original signal confidence
                'position_age': 0.2,        # 20% based on how long we've held it
                'relative_size': 0.1        # 10% based on position size relative to target
            }
        }
        
        self.load_configuration()
        
    def load_configuration(self):
        """Load dynamic position management configuration from database"""
        try:
            conn = mysql.connector.connect(**self.trades_db_config)
            cursor = conn.cursor(dictionary=True)
            
            # Load dynamic position management settings
            cursor.execute("""
                SELECT parameter_name, parameter_value 
                FROM dynamic_position_config 
                WHERE enabled = 1
            """)
            
            configs = cursor.fetchall()
            
            for config in configs:
                param_name = config['parameter_name']
                param_value = config['parameter_value']
                
                # Parse value based on parameter type
                if param_name.startswith('enable_'):
                    self.config[param_name] = bool(int(param_value))
                elif 'threshold' in param_name or 'multiplier' in param_name or 'percentage' in param_name:
                    self.config[param_name] = float(param_value)
                elif 'hours' in param_name:
                    self.config[param_name] = int(param_value)
                elif param_name == 'position_strength_weights':
                    self.config[param_name] = json.loads(param_value)
            
            cursor.close()
            conn.close()
            
            logger.info(f"Loaded dynamic position configuration: rebalancing={self.config['enable_dynamic_rebalancing']}")
            
        except Exception as e:
            logger.warning(f"Error loading dynamic position config, using defaults: {e}")
            # Create default configuration in database
            self.create_default_configuration()
    
    def create_default_configuration(self):
        """Create default configuration table and values"""
        try:
            conn = mysql.connector.connect(**self.trades_db_config)
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dynamic_position_config (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    parameter_name VARCHAR(100) NOT NULL UNIQUE,
                    parameter_value TEXT NOT NULL,
                    description TEXT,
                    enabled BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                )
            """)
            
            # Insert default configurations
            default_configs = [
                ('enable_dynamic_rebalancing', '1', 'Enable intelligent position rebalancing for strong signals'),
                ('strong_signal_threshold', '0.85', 'Confidence threshold for signals to be considered strong'),
                ('very_strong_signal_threshold', '0.95', 'Confidence threshold for maximum position scaling'),
                ('max_position_multiplier', '2.0', 'Maximum position size multiplier for very strong signals'),
                ('weak_position_score_threshold', '0.3', 'Score threshold below which positions are candidates for selling'),
                ('rebalancing_lookback_hours', '72', 'Hours to look back for position performance analysis'),
                ('min_position_age_hours', '4', 'Minimum hours a position must be held before rebalancing'),
                ('max_rebalancing_percentage', '50.0', 'Maximum percentage of portfolio to rebalance at once'),
                ('position_strength_weights', json.dumps(self.config['position_strength_weights']), 'Weights for position strength scoring algorithm')
            ]
            
            for param_name, param_value, description in default_configs:
                cursor.execute("""
                    INSERT IGNORE INTO dynamic_position_config 
                    (parameter_name, parameter_value, description) 
                    VALUES (%s, %s, %s)
                """, (param_name, param_value, description))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info("Created default dynamic position management configuration")
            
        except Exception as e:
            logger.error(f"Error creating default configuration: {e}")
    
    def get_current_portfolio(self) -> Dict:
        """Get current portfolio from trading engine"""
        try:
            response = requests.get(f"{self.trading_engine_url}/portfolio", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Portfolio API returned {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Error fetching portfolio: {e}")
            return {}
    
    def calculate_position_strength_score(self, symbol: str, position_data: Dict) -> float:
        """Calculate strength score for an existing position"""
        try:
            weights = self.config['position_strength_weights']
            lookback_hours = self.config['rebalancing_lookback_hours']
            
            # Get position performance data
            conn = mysql.connector.connect(**self.trades_db_config)
            cursor = conn.cursor(dictionary=True)
            
            lookback_time = datetime.now() - timedelta(hours=lookback_hours)
            
            # Get recent trades for this symbol
            cursor.execute("""
                SELECT * FROM trades 
                WHERE symbol = %s AND timestamp >= %s 
                ORDER BY timestamp DESC
            """, (symbol, lookback_time))
            
            recent_trades = cursor.fetchall()
            
            # Get original signal confidence for most recent buy
            cursor.execute("""
                SELECT confidence FROM trading_signals 
                WHERE symbol = %s AND signal_type IN ('BUY', 'STRONG_BUY') 
                ORDER BY timestamp DESC LIMIT 1
            """, (symbol,))
            
            signal_data = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            # Calculate components
            scores = {}
            
            # 1. Recent Performance (40%)
            if recent_trades:
                total_pnl = 0
                buy_cost = 0
                for trade in recent_trades:
                    if trade['action'] == 'BUY':
                        buy_cost += float(trade['size_usd'])
                    elif trade['action'] == 'SELL':
                        total_pnl += float(trade['size_usd'])
                
                current_value = float(position_data.get('value', 0))
                if buy_cost > 0:
                    performance_ratio = (current_value + total_pnl) / buy_cost
                    scores['recent_performance'] = min(1.0, max(0.0, (performance_ratio - 0.9) / 0.2))  # Scale 0.9-1.1 to 0-1
                else:
                    scores['recent_performance'] = 0.5
            else:
                scores['recent_performance'] = 0.5
            
            # 2. Signal Confidence (30%)
            if signal_data:
                scores['signal_confidence'] = float(signal_data['confidence'])
            else:
                scores['signal_confidence'] = 0.5
            
            # 3. Position Age (20%) - newer positions score higher
            if recent_trades:
                last_buy = max([trade['timestamp'] for trade in recent_trades if trade['action'] == 'BUY'])
                age_hours = (datetime.now() - last_buy).total_seconds() / 3600
                # Score decreases with age, 0-24 hours = 1.0, 24-168 hours = linear decay to 0.2
                if age_hours <= 24:
                    scores['position_age'] = 1.0
                elif age_hours <= 168:  # 1 week
                    scores['position_age'] = 1.0 - (age_hours - 24) / 144 * 0.8
                else:
                    scores['position_age'] = 0.2
            else:
                scores['position_age'] = 0.5
            
            # 4. Relative Size (10%) - positions closer to target size score higher
            portfolio = self.get_current_portfolio()
            total_value = portfolio.get('total_portfolio_value', 1)
            position_percentage = float(position_data.get('value', 0)) / total_value * 100
            target_percentage = 20.0  # Assuming 20% target allocation
            size_deviation = abs(position_percentage - target_percentage) / target_percentage
            scores['relative_size'] = max(0.0, 1.0 - size_deviation)
            
            # Calculate weighted score
            total_score = sum(scores[component] * weights[component] for component in scores)
            
            logger.info(f"Position strength for {symbol}: {total_score:.3f} (perf:{scores['recent_performance']:.2f}, conf:{scores['signal_confidence']:.2f}, age:{scores['position_age']:.2f}, size:{scores['relative_size']:.2f})")
            
            return total_score
            
        except Exception as e:
            logger.error(f"Error calculating position strength for {symbol}: {e}")
            return 0.5  # Default neutral score
    
    def identify_weak_positions(self, portfolio: Dict, required_amount: float) -> List[Tuple[str, float, float]]:
        """Identify weak positions that should be sold to fund new strong position"""
        if not self.config['enable_dynamic_rebalancing']:
            return []
        
        weak_positions = []
        positions = portfolio.get('positions', [])
        
        for position in positions:
            symbol = position.get('currency', '')
            if not symbol:
                continue
                
            position_value = float(position.get('value', 0))
            if position_value < 1.0:  # Skip very small positions
                continue
            
            # Check minimum age requirement
            if not self.is_position_old_enough(symbol):
                continue
            
            # Calculate position strength score
            strength_score = self.calculate_position_strength_score(symbol, position)
            
            if strength_score < self.config['weak_position_score_threshold']:
                weak_positions.append((symbol, strength_score, position_value))
        
        # Sort by strength score (weakest first)
        weak_positions.sort(key=lambda x: x[1])
        
        # Select positions to sell to meet required amount
        selected_positions = []
        total_selected_value = 0
        max_rebalancing = portfolio.get('total_portfolio_value', 0) * self.config['max_rebalancing_percentage'] / 100
        
        for symbol, score, value in weak_positions:
            if total_selected_value >= required_amount:
                break
            if total_selected_value + value > max_rebalancing:
                break
                
            selected_positions.append((symbol, score, value))
            total_selected_value += value
        
        logger.info(f"Identified {len(selected_positions)} weak positions worth ${total_selected_value:.2f} for rebalancing")
        return selected_positions
    
    def is_position_old_enough(self, symbol: str) -> bool:
        """Check if position is old enough to be rebalanced"""
        try:
            conn = mysql.connector.connect(**self.trades_db_config)
            cursor = conn.cursor()
            
            min_age = datetime.now() - timedelta(hours=self.config['min_position_age_hours'])
            
            cursor.execute("""
                SELECT timestamp FROM trades 
                WHERE symbol = %s AND action = 'BUY' 
                ORDER BY timestamp DESC LIMIT 1
            """, (symbol,))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                last_buy_time = result[0]
                return last_buy_time <= min_age
            
            return True  # If no recent buy found, assume it's old enough
            
        except Exception as e:
            logger.error(f"Error checking position age for {symbol}: {e}")
            return False
    
    def calculate_dynamic_position_size(self, signal_confidence: float, base_position_size: float) -> float:
        """Calculate position size based on signal strength"""
        if not self.config['enable_dynamic_rebalancing']:
            return base_position_size
        
        if signal_confidence >= self.config['very_strong_signal_threshold']:
            # Very strong signals get maximum multiplier
            multiplier = self.config['max_position_multiplier']
        elif signal_confidence >= self.config['strong_signal_threshold']:
            # Strong signals get scaled multiplier
            strength_ratio = (signal_confidence - self.config['strong_signal_threshold']) / \
                           (self.config['very_strong_signal_threshold'] - self.config['strong_signal_threshold'])
            multiplier = 1.0 + strength_ratio * (self.config['max_position_multiplier'] - 1.0)
        else:
            # Normal signals get standard position size
            multiplier = 1.0
        
        dynamic_size = base_position_size * multiplier
        
        logger.info(f"Dynamic position sizing: confidence={signal_confidence:.3f}, multiplier={multiplier:.2f}, size=${dynamic_size:.2f}")
        
        return dynamic_size
    
    def generate_rebalancing_signals(self, strong_signal: Dict, required_funding: float) -> List[Dict]:
        """Generate SELL signals for weak positions to fund strong BUY signal"""
        if not self.config['enable_dynamic_rebalancing']:
            return []
        
        portfolio = self.get_current_portfolio()
        if not portfolio:
            return []
        
        weak_positions = self.identify_weak_positions(portfolio, required_funding)
        
        rebalancing_signals = []
        
        for symbol, strength_score, position_value in weak_positions:
            # Create SELL signal for weak position
            sell_signal = {
                'symbol': symbol,
                'signal_type': 'SELL',
                'confidence': 0.75 + (1.0 - strength_score) * 0.2,  # Higher confidence for weaker positions
                'reasoning': f"Rebalancing: Selling weak position (strength: {strength_score:.2f}) to fund strong {strong_signal['symbol']} signal (confidence: {strong_signal['confidence']:.2f})",
                'rebalancing_for': strong_signal['symbol'],
                'rebalancing_confidence': strong_signal['confidence'],
                'position_strength': strength_score,
                'size_usd': position_value,
                'urgency': 'high' if strong_signal['confidence'] >= self.config['very_strong_signal_threshold'] else 'medium'
            }
            
            rebalancing_signals.append(sell_signal)
        
        logger.info(f"Generated {len(rebalancing_signals)} rebalancing SELL signals for strong {strong_signal['symbol']} BUY")
        
        return rebalancing_signals
    
    def get_configuration(self) -> Dict:
        """Get current configuration for dashboard display"""
        return self.config.copy()
    
    def update_configuration(self, updates: Dict) -> bool:
        """Update configuration from dashboard"""
        try:
            conn = mysql.connector.connect(**self.trades_db_config)
            cursor = conn.cursor()
            
            for param_name, param_value in updates.items():
                if param_name in self.config:
                    # Convert value to string for database storage
                    if isinstance(param_value, dict):
                        value_str = json.dumps(param_value)
                    else:
                        value_str = str(param_value)
                    
                    cursor.execute("""
                        UPDATE dynamic_position_config 
                        SET parameter_value = %s, updated_at = CURRENT_TIMESTAMP 
                        WHERE parameter_name = %s
                    """, (value_str, param_name))
                    
                    # Update local config
                    self.config[param_name] = param_value
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Updated dynamic position configuration: {list(updates.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return False
    
    def consolidate_small_positions(self, min_position_value_usd: float = 10.0) -> List[Dict]:
        """
        Consolidate small positions by selling them and combining into stronger positions
        
        Args:
            min_position_value_usd: Minimum USD value to keep a position
            
        Returns:
            List of consolidation signals (SELL signals for small positions)
        """
        try:
            portfolio = self.get_current_portfolio()
            if not portfolio:
                logger.info("No portfolio data available for consolidation")
                return []
            
            small_positions = []
            total_consolidation_value = 0.0
            
            # Identify small positions
            for symbol, position_data in portfolio.items():
                if symbol == 'USD':  # Skip cash
                    continue
                    
                position_value = position_data.get('value_usd', 0.0)
                if position_value > 0 and position_value < min_position_value_usd:
                    # Check if position is old enough to consolidate
                    if self.is_position_old_enough(symbol):
                        small_positions.append({
                            'symbol': symbol,
                            'value_usd': position_value,
                            'quantity': position_data.get('quantity', 0.0),
                            'avg_price': position_data.get('avg_price', 0.0)
                        })
                        total_consolidation_value += position_value
            
            if not small_positions:
                logger.info("No small positions found for consolidation")
                return []
            
            logger.info(f"Found {len(small_positions)} small positions worth ${total_consolidation_value:.2f} for consolidation")
            
            # Generate SELL signals for small positions
            consolidation_signals = []
            
            for position in small_positions:
                sell_signal = {
                    'symbol': position['symbol'],
                    'signal_type': 'SELL',
                    'confidence': 0.80,  # High confidence for consolidation
                    'reasoning': f"Portfolio consolidation: Position value ${position['value_usd']:.2f} below minimum threshold ${min_position_value_usd:.2f}",
                    'consolidation': True,
                    'consolidation_value': position['value_usd'],
                    'size_usd': position['value_usd'],
                    'urgency': 'low',  # Low urgency for consolidation
                    'position_age_check': True  # Indicates we checked position age
                }
                
                consolidation_signals.append(sell_signal)
            
            logger.info(f"Generated {len(consolidation_signals)} consolidation SELL signals worth ${total_consolidation_value:.2f}")
            return consolidation_signals
            
        except Exception as e:
            logger.error(f"Error in consolidate_small_positions: {e}")
            return []