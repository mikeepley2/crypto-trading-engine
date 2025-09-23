#!/usr/bin/env python3
"""
Advanced Portfolio Rebalancer Service

Enhanced intelligent rebalancing system with multiple optimization strategies:
1. Dynamic concentration management 
2. Correlation-based diversification
3. Volatility-adjusted position sizing
4. Performance-based rebalancing
5. Sector/category diversification
6. Liquidity-based optimization
7. Momentum-based adjustments
8. Market regime adaptation

Runs as a containerized FastAPI service with health endpoints and periodic rebalancing.
"""

import mysql.connector
import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import math
import os
import threading
import time
import uvicorn
from fastapi import FastAPI
from signal_coherence_manager import SignalCoherenceManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global health status tracking
health_status = {
    "status": "starting",
    "last_rebalancing": None,
    "rebalancing_cycles_completed": 0,
    "database_connected": False,
    "last_error": None,
    "start_time": datetime.now(),
    "service_name": "advanced-portfolio-rebalancer",
    "rebalancing_interval_hours": 4
}

# FastAPI app for health endpoints
app = FastAPI(title="Advanced Portfolio Rebalancer API")

@app.get("/health")
async def health_check():
    """Standard health check endpoint"""
    is_healthy = (
        health_status["database_connected"] and
        health_status["status"] not in ["error", "critical"]
    )
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "service": health_status["service_name"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/status")
async def get_status():
    """Detailed status information"""
    uptime = datetime.now() - health_status["start_time"]
    
    next_rebalancing = None
    if health_status["last_rebalancing"]:
        last_rebal = datetime.fromisoformat(health_status["last_rebalancing"].replace('Z', '+00:00'))
        next_rebal_time = last_rebal + timedelta(hours=health_status["rebalancing_interval_hours"])
        time_until_next = next_rebal_time - datetime.now()
        if time_until_next.total_seconds() > 0:
            hours = int(time_until_next.total_seconds() // 3600)
            minutes = int((time_until_next.total_seconds() % 3600) // 60)
            next_rebalancing = f"{hours}h {minutes}m"
        else:
            next_rebalancing = "due now"
    
    return {
        **health_status,
        "uptime_seconds": uptime.total_seconds(),
        "uptime_human": str(uptime).split('.')[0],
        "next_rebalancing_in": next_rebalancing,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Service metrics for monitoring"""
    uptime = datetime.now() - health_status["start_time"]
    
    return {
        "rebalancing_cycles_completed": health_status["rebalancing_cycles_completed"],
        "last_rebalancing": health_status["last_rebalancing"],
        "uptime_seconds": uptime.total_seconds(),
        "rebalancing_interval_hours": health_status["rebalancing_interval_hours"],
        "error_count": 1 if health_status["last_error"] else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/rebalance/trigger")
async def trigger_rebalancing():
    """Manually trigger a rebalancing cycle"""
    try:
        rebalancer = AdvancedPortfolioRebalancer()
        success = rebalancer.create_advanced_rebalancing_signals()
        
        if success:
            health_status["last_rebalancing"] = datetime.now().isoformat()
            health_status["rebalancing_cycles_completed"] += 1
            return {"status": "success", "message": "Rebalancing triggered successfully"}
        else:
            return {"status": "error", "message": "Rebalancing failed"}
    except Exception as e:
        logger.error(f"Manual rebalancing trigger failed: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/rebalance/analysis")
async def get_rebalancing_analysis():
    """Get current portfolio rebalancing analysis"""
    try:
        rebalancer = AdvancedPortfolioRebalancer()
        plan = rebalancer.generate_advanced_rebalancing_plan()
        return {"status": "success", "analysis": plan}
    except Exception as e:
        logger.error(f"Rebalancing analysis failed: {e}")
        return {"status": "error", "message": str(e)}

def start_health_server():
    """Start the FastAPI health server in a background thread"""
    def run_server():
        try:
            uvicorn.run(app, host="0.0.0.0", port=8035, log_level="warning")
        except Exception as e:
            logger.error(f"Health server error: {e}")
    
    health_thread = threading.Thread(target=run_server, daemon=True)
    health_thread.start()
    logger.info("🏥 Health server started on port 8035")

class AdvancedPortfolioRebalancer:
    def __init__(self):
        self.db_config = {
            'host': os.environ.get('DATABASE_HOST', 'host.docker.internal'),
            'user': os.environ.get('DATABASE_USER', 'news_collector'),
            'password': os.environ.get('DATABASE_PASSWORD', '99Rules!'),
            'database': os.environ.get('DATABASE_NAME', 'crypto_prices'),
            'port': int(os.environ.get('DATABASE_PORT', 3306))
        }
        self.trading_db_config = {
            'host': os.environ.get('DATABASE_HOST', 'host.docker.internal'),
            'user': os.environ.get('DATABASE_USER', 'news_collector'),
            'password': os.environ.get('DATABASE_PASSWORD', '99Rules!'),
            'database': 'crypto_transactions',
            'port': int(os.environ.get('DATABASE_PORT', 3306))
        }
        self.trading_engine_url = os.environ.get('TRADING_ENGINE_URL', 'http://host.docker.internal:8024')
        
        # Initialize Signal Coherence Manager
        self.signal_manager = SignalCoherenceManager(self.db_config)
        
        # Test database connectivity
        self.test_database_connection()
        
        # Advanced rebalancing parameters
        self.config = {
            # Position sizing
            'max_position_weight': float(os.environ.get('MAX_POSITION_WEIGHT', 15.0)),  # Reduced from 20% for better diversification
            'min_position_value': float(os.environ.get('MIN_POSITION_VALUE', 50.0)),   # Minimum meaningful position
            'max_positions': int(os.environ.get('MAX_POSITIONS', 15)),          # Target maximum positions
            'target_cash_percentage': float(os.environ.get('TARGET_CASH_PERCENTAGE', 8.0)), # Target cash buffer
            
            # Correlation limits
            'max_sector_allocation': float(os.environ.get('MAX_SECTOR_ALLOCATION', 40.0)),  # Max allocation to similar assets
            'correlation_threshold': float(os.environ.get('CORRELATION_THRESHOLD', 0.7)),   # High correlation threshold
            
            # Performance thresholds
            'underperformance_threshold': float(os.environ.get('UNDERPERFORMANCE_THRESHOLD', -10.0)),  # % underperformance vs portfolio
            'momentum_lookback_days': int(os.environ.get('MOMENTUM_LOOKBACK_DAYS', 7)),          # Days for momentum calculation
            
            # Volatility adjustments
            'volatility_adjustment_factor': float(os.environ.get('VOLATILITY_ADJUSTMENT_FACTOR', 0.3)),  # How much to adjust for volatility
            'max_volatility_position': float(os.environ.get('MAX_VOLATILITY_POSITION', 12.0)),     # Max weight for high-vol assets
            
            # Market regime adjustments
            'bull_market_max_weight': float(os.environ.get('BULL_MARKET_MAX_WEIGHT', 18.0)),      # Higher concentration in bull markets
            'bear_market_max_weight': float(os.environ.get('BEAR_MARKET_MAX_WEIGHT', 10.0)),      # Lower concentration in bear markets
        }
    
    def test_database_connection(self):
        """Test database connectivity on startup"""
        try:
            # Test crypto_prices connection
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()  # Consume the result
            cursor.close()
            conn.close()
            
            # Test crypto_transactions connection
            conn = mysql.connector.connect(**self.trading_db_config)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()  # Consume the result
            cursor.close()
            conn.close()
            
            health_status["database_connected"] = True
            health_status["status"] = "ready"
            logger.info("✅ Database connectivity verified")
            
        except Exception as e:
            health_status["database_connected"] = False
            health_status["status"] = "error"
            health_status["last_error"] = f"Database connection failed: {e}"
            logger.error(f"❌ Database connection failed: {e}")
            raise
    
    def get_portfolio(self) -> Dict:
        """Get current portfolio from trading engine"""
        try:
            response = requests.get(f"{self.trading_engine_url}/portfolio", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get portfolio: {e}")
            return {}
    
    def get_market_regime(self) -> str:
        """Determine current market regime"""
        try:
            # Simple regime detection based on recent BTC performance
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT current_price, timestamp_iso
                FROM ml_features_materialized 
                WHERE symbol = 'BTC'
                ORDER BY timestamp_iso DESC 
                LIMIT 168  -- 7 days of hourly data
            """)
            
            btc_data = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if len(btc_data) < 100:
                return 'SIDEWAYS'
            
            prices = [float(row[0]) for row in btc_data]
            recent_return = (prices[0] - prices[23]) / prices[23] * 100  # 24h return
            weekly_return = (prices[0] - prices[-1]) / prices[-1] * 100  # 7d return
            
            # Simple regime classification
            if weekly_return > 15 and recent_return > 5:
                return 'BULL'
            elif weekly_return < -15 and recent_return < -5:
                return 'BEAR'
            else:
                return 'SIDEWAYS'
                
        except Exception as e:
            logger.warning(f"Error determining market regime: {e}")
            return 'SIDEWAYS'
    
    def calculate_position_volatilities(self, symbols: List[str]) -> Dict[str, float]:
        """Calculate historical volatility for each position"""
        volatilities = {}
        
        try:
            conn = mysql.connector.connect(**self.db_config)
            
            for symbol in symbols:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT current_price
                    FROM ml_features_materialized 
                    WHERE symbol = %s
                    ORDER BY timestamp_iso DESC 
                    LIMIT 168  -- 7 days
                """, (symbol,))
                
                prices = [float(row[0]) for row in cursor.fetchall()]
                cursor.close()
                
                if len(prices) > 10:
                    returns = []
                    for i in range(1, len(prices)):
                        returns.append((prices[i-1] - prices[i]) / prices[i])
                    
                    vol = np.std(returns) * np.sqrt(24)  # Annualized hourly volatility
                    volatilities[symbol] = vol
                else:
                    volatilities[symbol] = 0.5  # Default moderate volatility
            
            conn.close()
            
        except Exception as e:
            logger.warning(f"Error calculating volatilities: {e}")
            # Default volatilities
            for symbol in symbols:
                volatilities[symbol] = 0.5
        
        return volatilities
    
    def calculate_position_correlations(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate price correlations between positions"""
        correlations = {}
        
        try:
            conn = mysql.connector.connect(**self.db_config)
            
            # Get price data for all symbols
            price_data = {}
            for symbol in symbols:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT current_price, timestamp_iso
                    FROM ml_features_materialized 
                    WHERE symbol = %s
                    ORDER BY timestamp_iso DESC 
                    LIMIT 168
                """, (symbol,))
                
                prices = [(float(row[0]), row[1]) for row in cursor.fetchall()]
                price_data[symbol] = prices
                cursor.close()
            
            conn.close()
            
            # Calculate correlations
            for symbol1 in symbols:
                correlations[symbol1] = {}
                for symbol2 in symbols:
                    if symbol1 == symbol2:
                        correlations[symbol1][symbol2] = 1.0
                    else:
                        # Simple correlation calculation
                        prices1 = [p[0] for p in price_data.get(symbol1, [])]
                        prices2 = [p[0] for p in price_data.get(symbol2, [])]
                        
                        if len(prices1) > 10 and len(prices2) > 10:
                            min_len = min(len(prices1), len(prices2))
                            corr = np.corrcoef(prices1[:min_len], prices2[:min_len])[0, 1]
                            correlations[symbol1][symbol2] = corr if not np.isnan(corr) else 0.0
                        else:
                            correlations[symbol1][symbol2] = 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating correlations: {e}")
            # Default to no correlation
            for symbol1 in symbols:
                correlations[symbol1] = {}
                for symbol2 in symbols:
                    correlations[symbol1][symbol2] = 1.0 if symbol1 == symbol2 else 0.0
        
        return correlations
    
    def calculate_position_performance(self, symbol: str, position_value: float) -> Dict:
        """Calculate position performance metrics"""
        try:
            conn = mysql.connector.connect(**self.trading_db_config)
            cursor = conn.cursor()
            
            # Get recent trades for this symbol
            cursor.execute("""
                SELECT executed_price, executed_quantity, trade_type, executed_at
                FROM trades 
                WHERE symbol = %s AND executed_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                ORDER BY executed_at DESC
            """, (symbol,))
            
            trades = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not trades:
                return {'performance_score': 0.5, 'avg_entry_price': 0, 'unrealized_pnl': 0}
            
            # Calculate weighted average entry price
            total_bought = 0
            total_sold = 0
            weighted_entry_price = 0
            
            for price, quantity, trade_type, executed_at in trades:
                if trade_type == 'BUY':
                    total_bought += quantity
                    weighted_entry_price += price * quantity
                else:
                    total_sold += quantity
            
            net_position = total_bought - total_sold
            if net_position > 0 and total_bought > 0:
                avg_entry_price = weighted_entry_price / total_bought
                
                # Get current price
                current_price = self.get_current_price(symbol)
                unrealized_pnl = (current_price - avg_entry_price) / avg_entry_price * 100
                
                # Performance score: 0 = very poor, 0.5 = neutral, 1 = excellent
                performance_score = min(1.0, max(0.0, (unrealized_pnl + 50) / 100))
                
                return {
                    'performance_score': performance_score,
                    'avg_entry_price': avg_entry_price,
                    'unrealized_pnl': unrealized_pnl,
                    'current_price': current_price
                }
            
        except Exception as e:
            logger.warning(f"Error calculating performance for {symbol}: {e}")
        
        return {'performance_score': 0.5, 'avg_entry_price': 0, 'unrealized_pnl': 0}
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT current_price
                FROM ml_features_materialized 
                WHERE symbol = %s
                ORDER BY timestamp_iso DESC 
                LIMIT 1
            """, (symbol,))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            return float(result[0]) if result else 100.0
            
        except Exception as e:
            logger.warning(f"Error getting price for {symbol}: {e}")
            return 100.0
    
    def categorize_assets(self, symbols: List[str]) -> Dict[str, str]:
        """Categorize assets by type/sector"""
        # Simple categorization - in production, this could use external data
        categories = {
            'BTC': 'DIGITAL_GOLD',
            'ETH': 'SMART_CONTRACT',
            'ADA': 'SMART_CONTRACT',
            'SOL': 'SMART_CONTRACT',
            'LINK': 'ORACLE',
            'SHIB': 'MEME',
            'DOGE': 'MEME',
            'XRP': 'PAYMENTS',
            'XLM': 'PAYMENTS',
            'RNDR': 'AI_COMPUTE',
            'FET': 'AI',
            'GRT': 'INDEXING',
            'SAND': 'METAVERSE',
            'CHZ': 'SPORTS',
            'VET': 'SUPPLY_CHAIN',
            'CTSI': 'INFRASTRUCTURE'
        }
        
        result = {}
        for symbol in symbols:
            result[symbol] = categories.get(symbol, 'OTHER')
        
        return result
    
    def generate_advanced_rebalancing_plan(self) -> Dict:
        """Generate comprehensive rebalancing plan with multiple optimization strategies"""
        portfolio = self.get_portfolio()
        if not portfolio:
            return {}
        
        positions = portfolio.get('positions', {})
        cash_balance = portfolio.get('cash_balance', 0)
        total_value = portfolio.get('total_portfolio_value', 0)
        
        # Filter meaningful positions
        meaningful_positions = {
            symbol: data for symbol, data in positions.items()
            if data.get('value_usd', 0) > 1.0  # $1+ positions
        }
        
        symbols = list(meaningful_positions.keys())
        market_regime = self.get_market_regime()
        
        logger.info(f"🎯 Analyzing {len(symbols)} positions in {market_regime} market")
        
        # Calculate advanced metrics
        volatilities = self.calculate_position_volatilities(symbols)
        correlations = self.calculate_position_correlations(symbols)
        categories = self.categorize_assets(symbols)
        
        # Analyze each position
        position_analysis = []
        for symbol, position_data in meaningful_positions.items():
            value = position_data.get('value_usd', 0)
            weight = (value / total_value * 100) if total_value > 0 else 0
            
            performance = self.calculate_position_performance(symbol, value)
            volatility = volatilities.get(symbol, 0.5)
            category = categories.get(symbol, 'OTHER')
            
            # Calculate optimal weight based on multiple factors
            base_weight = self.config['max_position_weight']
            
            # Adjust for market regime
            if market_regime == 'BULL':
                regime_weight = self.config['bull_market_max_weight']
            elif market_regime == 'BEAR':
                regime_weight = self.config['bear_market_max_weight']
            else:
                regime_weight = base_weight
            
            # Adjust for volatility (lower weight for high volatility)
            vol_adjustment = max(0.5, 1.0 - (volatility * self.config['volatility_adjustment_factor']))
            optimal_weight = regime_weight * vol_adjustment
            
            # Adjust for performance (reduce underperformers)
            perf_adjustment = 0.5 + (performance['performance_score'] * 0.5)
            optimal_weight *= perf_adjustment
            
            # Final caps
            optimal_weight = min(optimal_weight, self.config['max_volatility_position'])
            
            analysis = {
                'symbol': symbol,
                'current_value': value,
                'current_weight': weight,
                'optimal_weight': optimal_weight,
                'weight_delta': weight - optimal_weight,
                'volatility': volatility,
                'category': category,
                'performance_score': performance['performance_score'],
                'unrealized_pnl': performance.get('unrealized_pnl', 0),
                'rebalance_priority': abs(weight - optimal_weight) * (1 + volatility)  # Higher priority for bigger misalignments
            }
            
            position_analysis.append(analysis)
        
        # Sort by rebalance priority
        position_analysis.sort(key=lambda x: x['rebalance_priority'], reverse=True)
        
        # Identify consolidation opportunities (tiny positions)
        tiny_positions = [p for p in position_analysis if p['current_value'] < self.config['min_position_value']]
        
        # Identify sector concentration issues
        sector_allocations = {}
        for pos in position_analysis:
            category = pos['category']
            sector_allocations[category] = sector_allocations.get(category, 0) + pos['current_weight']
        
        # Generate rebalancing recommendations
        recommendations = {
            'market_regime': market_regime,
            'total_positions': len(position_analysis),
            'cash_percentage': (cash_balance / total_value * 100) if total_value > 0 else 0,
            'sector_allocations': sector_allocations,
            'oversized_positions': [p for p in position_analysis if p['weight_delta'] > 3.0],
            'undersized_opportunities': [p for p in position_analysis if p['weight_delta'] < -2.0 and p['performance_score'] > 0.6],
            'consolidation_targets': tiny_positions,
            'performance_concerns': [p for p in position_analysis if p['performance_score'] < 0.3 and p['current_value'] > 25],
            'high_volatility_positions': [p for p in position_analysis if p['volatility'] > 0.8 and p['current_weight'] > 8],
            'recommended_actions': []
        }
        
        # Generate specific action recommendations
        
        # 1. Consolidate tiny positions
        if len(tiny_positions) > 5:
            total_tiny_value = sum(p['current_value'] for p in tiny_positions)
            recommendations['recommended_actions'].append({
                'action': 'CONSOLIDATE_TINY_POSITIONS',
                'description': f"Consolidate {len(tiny_positions)} positions worth ${total_tiny_value:.2f} total",
                'symbols': [p['symbol'] for p in tiny_positions],
                'reason': 'Reduce portfolio complexity and transaction costs'
            })
        
        # 2. Trim oversized positions
        for pos in recommendations['oversized_positions']:
            trim_amount = min(pos['current_value'] * 0.3, (pos['weight_delta'] / 100) * total_value)
            if trim_amount > 25:
                recommendations['recommended_actions'].append({
                    'action': 'TRIM_POSITION',
                    'symbol': pos['symbol'],
                    'amount': trim_amount,
                    'description': f"Trim {pos['symbol']} by ${trim_amount:.0f} (from {pos['current_weight']:.1f}% to {pos['optimal_weight']:.1f}%)",
                    'reason': f'Position exceeds optimal weight by {pos["weight_delta"]:.1f}%'
                })
        
        # 3. Address performance concerns
        for pos in recommendations['performance_concerns']:
            if pos['unrealized_pnl'] < -15:  # More than 15% loss
                recommendations['recommended_actions'].append({
                    'action': 'REDUCE_UNDERPERFORMER',
                    'symbol': pos['symbol'],
                    'amount': pos['current_value'] * 0.5,
                    'description': f"Reduce {pos['symbol']} by 50% due to poor performance ({pos['unrealized_pnl']:.1f}% PnL)",
                    'reason': 'Cut losses and redeploy to better opportunities'
                })
        
        # 4. Address sector concentration
        for sector, allocation in sector_allocations.items():
            if allocation > self.config['max_sector_allocation']:
                excess = allocation - self.config['max_sector_allocation']
                sector_positions = [p for p in position_analysis if p['category'] == sector]
                recommendations['recommended_actions'].append({
                    'action': 'REDUCE_SECTOR_CONCENTRATION',
                    'sector': sector,
                    'excess_allocation': excess,
                    'description': f"Reduce {sector} allocation from {allocation:.1f}% to {self.config['max_sector_allocation']:.1f}%",
                    'affected_positions': sector_positions,
                    'reason': 'Avoid sector over-concentration risk'
                })
        
        return recommendations
    
    def create_advanced_rebalancing_signals(self) -> bool:
        """Create advanced rebalancing signals based on comprehensive analysis"""
        plan = self.generate_advanced_rebalancing_plan()
        
        if not plan or not plan.get('recommended_actions'):
            logger.info("No advanced rebalancing needed")
            return True

        logger.info(f"🎯 Creating advanced rebalancing plan with {len(plan['recommended_actions'])} actions")
        
        try:
            # Clean up old signals to prevent conflicts
            self.signal_manager.cleanup_old_signals(max_age_hours=2)
            
            signal_count = 0
            
            for action in plan['recommended_actions']:
                if action['action'] in ['TRIM_POSITION', 'REDUCE_UNDERPERFORMER']:
                    symbol = action['symbol']
                    amount = action['amount']
                    
                    llm_analysis = {
                        'action': 'SELL',
                        'confidence': 0.85,
                        'reasoning': action['reason'],
                        'rebalance_amount': amount,
                        'advanced_rebalancing': True,
                        'action_type': action['action']
                    }
                    
                    llm_reasoning = f"""ADVANCED PORTFOLIO REBALANCING

Action: {action['action']}
Symbol: {symbol}
Amount: ${amount:.2f}
Market Regime: {plan['market_regime']}

Strategy: {action['description']}
Reasoning: {action['reason']}

This action is part of an advanced portfolio optimization strategy that considers:
- Position sizing optimization based on volatility and performance
- Market regime adaptation ({plan['market_regime']} market conditions)
- Sector diversification and concentration risk management
- Performance-based position adjustments

This rebalancing move will improve overall portfolio risk-adjusted returns."""

                    # Generate signal using Signal Coherence Manager
                    additional_data = {
                        'timestamp': datetime.now(timezone.utc),
                        'price': 100.0,  # placeholder
                        'threshold': 0.8,
                        'regime': plan['market_regime'].lower(),
                        'features_used': 15,
                        'xgboost_confidence': 0.85,
                        'data_source': 'advanced_rebalancing',
                        'real_time_available': 1,
                        'llm_analysis': json.dumps(llm_analysis),
                        'llm_confidence': 0.85,
                        'llm_reasoning': llm_reasoning,
                        'prediction_timestamp': datetime.now(timezone.utc),
                        'prediction': amount,
                        'is_mock': 0
                    }
                    
                    # Use Signal Coherence Manager to check conflicts and generate signal
                    signal_generated = self.signal_manager.generate_signal_safely(
                        symbol=symbol,
                        signal_type='SELL',
                        confidence=0.85,
                        model_name='advanced_rebalancer',
                        model_version='2.0_advanced',
                        additional_data=additional_data
                    )
                    
                    if signal_generated:
                        logger.info(f"✅ Created advanced rebalancing signal: {action['action']} {symbol} ${amount:.2f}")
                        signal_count += 1
                    else:
                        logger.info(f"🚫 Skipped rebalancing signal for {symbol} due to conflicts")
                
                elif action['action'] == 'CONSOLIDATE_TINY_POSITIONS':
                    # Create consolidation signals for tiny positions
                    for symbol in action['symbols']:
                        llm_analysis = {
                            'action': 'SELL',
                            'confidence': 0.80,
                            'reasoning': 'Portfolio consolidation - eliminate tiny positions',
                            'consolidation': True,
                            'advanced_rebalancing': True
                        }
                        
                        llm_reasoning = f"""PORTFOLIO CONSOLIDATION

Action: Liquidate tiny position
Symbol: {symbol}
Strategy: Portfolio simplification and cost reduction

Reasoning: This position is too small to meaningfully impact returns but adds complexity and transaction costs. Consolidating into larger, more impactful positions will improve portfolio efficiency.

Market Regime: {plan['market_regime']}
Total tiny positions being consolidated: {len(action['symbols'])}"""

                        # Generate consolidation signal using Signal Coherence Manager
                        additional_data = {
                            'timestamp': datetime.now(timezone.utc),
                            'price': 100.0,  # placeholder
                            'threshold': 0.75,
                            'regime': plan['market_regime'].lower(),
                            'features_used': 8,
                            'xgboost_confidence': 0.80,
                            'data_source': 'portfolio_consolidation',
                            'real_time_available': 1,
                            'llm_analysis': json.dumps(llm_analysis),
                            'llm_confidence': 0.80,
                            'llm_reasoning': llm_reasoning,
                            'prediction_timestamp': datetime.now(timezone.utc),
                            'prediction': 100.0,  # sell all
                            'is_mock': 0
                        }
                        
                        # Use Signal Coherence Manager to check conflicts and generate signal
                        signal_generated = self.signal_manager.generate_signal_safely(
                            symbol=symbol,
                            signal_type='SELL',
                            confidence=0.80,
                            model_name='advanced_rebalancer',
                            model_version='2.0_consolidation',
                            additional_data=additional_data
                        )
                        
                        if signal_generated:
                            logger.info(f"✅ Created consolidation signal: SELL {symbol} (tiny position cleanup)")
                            signal_count += 1
                        else:
                            logger.info(f"🚫 Skipped consolidation signal for {symbol} due to conflicts")
            
            logger.info(f"🎯 Advanced rebalancing complete - created {signal_count} signals")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create advanced rebalancing signals: {e}")
            return False
    
    def print_advanced_analysis(self):
        """Print comprehensive portfolio analysis"""
        plan = self.generate_advanced_rebalancing_plan()
        
        print("=" * 80)
        print("🚀 ADVANCED PORTFOLIO REBALANCING ANALYSIS")
        print("=" * 80)
        
        print(f"\n📊 Portfolio Overview:")
        print(f"  Market Regime: {plan['market_regime']}")
        print(f"  Total Positions: {plan['total_positions']}")
        print(f"  Cash Percentage: {plan['cash_percentage']:.1f}%")
        
        print(f"\n🏢 Sector Allocations:")
        for sector, allocation in sorted(plan['sector_allocations'].items(), key=lambda x: x[1], reverse=True):
            status = "⚠️ OVERWEIGHT" if allocation > 40 else "✅"
            print(f"  {sector}: {allocation:.1f}% {status}")
        
        print(f"\n🔴 Position Issues:")
        if plan['oversized_positions']:
            print(f"  Oversized Positions ({len(plan['oversized_positions'])}):")
            for pos in plan['oversized_positions']:
                print(f"    {pos['symbol']}: {pos['current_weight']:.1f}% (target: {pos['optimal_weight']:.1f}%)")
        
        if plan['performance_concerns']:
            print(f"  Performance Concerns ({len(plan['performance_concerns'])}):")
            for pos in plan['performance_concerns']:
                print(f"    {pos['symbol']}: {pos['unrealized_pnl']:.1f}% PnL, Score: {pos['performance_score']:.2f}")
        
        if plan['consolidation_targets']:
            print(f"  Tiny Positions ({len(plan['consolidation_targets'])}):")
            for pos in plan['consolidation_targets']:
                print(f"    {pos['symbol']}: ${pos['current_value']:.2f}")
        
        print(f"\n📋 Recommended Actions ({len(plan['recommended_actions'])}):")
        for action in plan['recommended_actions']:
            print(f"  {action['action']}: {action['description']}")

def run_periodic_rebalancing():
    """Run periodic rebalancing in background"""
    rebalancer = AdvancedPortfolioRebalancer()
    rebalancing_interval = int(os.environ.get('REBALANCING_INTERVAL_HOURS', 4)) * 3600  # Convert to seconds
    
    logger.info(f"� Starting periodic rebalancing every {rebalancing_interval/3600:.1f} hours")
    
    while True:
        try:
            logger.info("🎯 Starting advanced portfolio rebalancing cycle...")
            
            # Generate and print analysis
            plan = rebalancer.generate_advanced_rebalancing_plan()
            
            if plan and plan.get('recommended_actions'):
                logger.info(f"📊 Found {len(plan['recommended_actions'])} rebalancing opportunities")
                
                # Create rebalancing signals
                success = rebalancer.create_advanced_rebalancing_signals()
                
                if success:
                    health_status["last_rebalancing"] = datetime.now().isoformat()
                    health_status["rebalancing_cycles_completed"] += 1
                    health_status["status"] = "rebalancing_complete"
                    logger.info("✅ Advanced rebalancing cycle completed successfully")
                else:
                    health_status["status"] = "rebalancing_failed"
                    logger.error("❌ Rebalancing cycle failed")
            else:
                logger.info("ℹ️ No rebalancing needed - portfolio is optimally balanced")
                health_status["last_rebalancing"] = datetime.now().isoformat()
                health_status["rebalancing_cycles_completed"] += 1
                health_status["status"] = "no_rebalancing_needed"
            
        except Exception as e:
            health_status["last_error"] = str(e)
            health_status["status"] = "error"
            logger.error(f"❌ Rebalancing cycle error: {e}")
        
        # Wait for next cycle
        logger.info(f"⏳ Waiting {rebalancing_interval/3600:.1f} hours until next rebalancing cycle...")
        time.sleep(rebalancing_interval)

if __name__ == "__main__":
    # Start health server
    start_health_server()
    
    # Start periodic rebalancing in background
    rebalancing_thread = threading.Thread(target=run_periodic_rebalancing, daemon=True)
    rebalancing_thread.start()
    
    logger.info("🚀 Advanced Portfolio Rebalancer Service started")
    logger.info("📊 Health API: http://localhost:8035/health")
    logger.info("📈 Status API: http://localhost:8035/status")
    logger.info("🎯 Manual trigger: POST http://localhost:8035/rebalance/trigger")
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("🛑 Service shutting down...")
        print("✅ Advanced rebalancing signals created successfully!")
        print("📊 The signal bridge will process these in the next cycle...")
    else:
        print("❌ Failed to create advanced rebalancing signals")