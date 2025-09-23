#!/usr/bin/env python3
"""
Advanced Risk Management Service
Provides volatility-based position sizing, correlation monitoring, and portfolio heat analysis
Integrates with trading engine to optimize trade sizes dynamically
"""

import os
import sys
import asyncio
import logging
import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Advanced Risk Management Service", version="1.0.0")

class TradeRequest(BaseModel):
    symbol: str
    base_size: float
    current_positions: Dict

class RiskAdjustment(BaseModel):
    original_size: float
    final_size: float
    volatility_factor: float
    correlation_factor: float
    heat_factor: float
    portfolio_heat: float
    reasoning: Dict

class AdvancedRiskManager:
    def __init__(self):
        self.db_config = {
            'host': os.environ.get('DATABASE_HOST', 'host.docker.internal'),
            'user': os.environ.get('DATABASE_USER', 'news_collector'),
            'password': os.environ.get('DATABASE_PASSWORD', '99Rules!'),
            'database': os.environ.get('DATABASE_NAME', 'crypto_prices'),
            'port': int(os.environ.get('DATABASE_PORT', 3306))
        }
        
        # Risk parameters
        self.max_portfolio_heat = float(os.environ.get('MAX_PORTFOLIO_HEAT', '0.15'))  # Max 15% portfolio at risk
        self.volatility_lookback = int(os.environ.get('VOLATILITY_LOOKBACK', '14'))  # Days for volatility calculation
        self.correlation_threshold = float(os.environ.get('CORRELATION_THRESHOLD', '0.7'))  # Max correlation before reducing positions
        
    def calculate_volatility_adjusted_position_size(self, symbol: str, base_size: float) -> Tuple[float, float]:
        """Calculate position size adjusted for asset volatility"""
        try:
            # Get historical price data for volatility calculation
            conn = mysql.connector.connect(**self.db_config)
            query = """
            SELECT current_price, timestamp_iso
            FROM ml_features_materialized 
            WHERE symbol = %s 
            AND timestamp_iso >= DATE_SUB(NOW(), INTERVAL %s DAY)
            ORDER BY timestamp_iso ASC
            """
            
            df = pd.read_sql(query, conn, params=[symbol, self.volatility_lookback])
            conn.close()
            
            if len(df) < 5:
                logger.warning(f"Insufficient data for volatility calculation: {symbol}")
                return base_size, 1.0
            
            # Calculate daily returns and volatility
            df['returns'] = df['current_price'].pct_change().dropna()
            volatility = df['returns'].std() * np.sqrt(24)  # Annualized volatility
            
            # Volatility-based adjustment
            # High volatility = smaller position, Low volatility = larger position
            volatility_multiplier = 1.0 / (1.0 + volatility * 8)  # More conservative adjustment
            volatility_multiplier = max(0.4, min(2.0, volatility_multiplier))  # Cap between 40%-200%
            
            adjusted_size = base_size * volatility_multiplier
            
            logger.debug(f"📊 {symbol} volatility: {volatility:.3f}, multiplier: {volatility_multiplier:.3f}")
            return adjusted_size, volatility_multiplier
            
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return base_size, 1.0
    
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
        """Get 14-day volatility for an asset"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            query = """
            SELECT current_price 
            FROM ml_features_materialized 
            WHERE symbol = %s 
            AND timestamp_iso >= DATE_SUB(NOW(), INTERVAL 14 DAY)
            ORDER BY timestamp_iso ASC
            """
            
            df = pd.read_sql(query, conn, params=[symbol])
            conn.close()
            
            if len(df) < 5:
                return 0.3  # Default volatility
            
            returns = df['current_price'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(24)  # Annualized
            
            return min(2.0, max(0.1, volatility))  # Cap between 10%-200%
            
        except Exception as e:
            logger.error(f"Error getting volatility for {symbol}: {e}")
            return 0.3
    
    def calculate_correlation_adjustment(self, symbol: str, current_positions: Dict) -> float:
        """Reduce position size if too correlated with existing positions"""
        try:
            if not current_positions:
                return 1.0
            
            conn = mysql.connector.connect(**self.db_config)
            
            # Get correlation with existing positions
            correlations = []
            for existing_symbol in current_positions.keys():
                if existing_symbol == symbol:
                    continue
                
                correlation = self.get_asset_correlation(symbol, existing_symbol, conn)
                if correlation > self.correlation_threshold:
                    correlations.append(correlation)
            
            conn.close()
            
            if not correlations:
                return 1.0
            
            # Reduce position size based on highest correlation
            max_correlation = max(correlations)
            correlation_adjustment = 1.0 - (max_correlation - self.correlation_threshold) * 1.5
            correlation_adjustment = max(0.3, correlation_adjustment)  # Minimum 30% position
            
            logger.debug(f"🔗 {symbol} max correlation: {max_correlation:.3f}, adjustment: {correlation_adjustment:.3f}")
            return correlation_adjustment
            
        except Exception as e:
            logger.error(f"Error calculating correlation adjustment for {symbol}: {e}")
            return 1.0
    
    def get_asset_correlation(self, symbol1: str, symbol2: str, conn) -> float:
        """Calculate 30-day correlation between two assets"""
        try:
            query = """
            SELECT 
                s1.timestamp_iso,
                s1.current_price as price1,
                s2.current_price as price2
            FROM ml_features_materialized s1
            JOIN ml_features_materialized s2 ON s1.timestamp_iso = s2.timestamp_iso
            WHERE s1.symbol = %s AND s2.symbol = %s
            AND s1.timestamp_iso >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            ORDER BY s1.timestamp_iso ASC
            """
            
            df = pd.read_sql(query, conn, params=[symbol1, symbol2])
            
            if len(df) < 10:
                return 0.0
            
            returns1 = df['price1'].pct_change().dropna()
            returns2 = df['price2'].pct_change().dropna()
            
            correlation = returns1.corr(returns2)
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating correlation between {symbol1} and {symbol2}: {e}")
            return 0.0
    
    def optimize_position_size(self, symbol: str, base_size: float, current_positions: Dict) -> Dict:
        """Main function to optimize position size with all risk factors"""
        try:
            # 1. Volatility adjustment
            volatility_adjusted_size, volatility_factor = self.calculate_volatility_adjusted_position_size(symbol, base_size)
            
            # 2. Correlation adjustment
            correlation_adjustment = self.calculate_correlation_adjustment(symbol, current_positions)
            correlation_adjusted_size = volatility_adjusted_size * correlation_adjustment
            
            # 3. Portfolio heat check
            portfolio_heat = self.calculate_portfolio_heat(current_positions)
            heat_adjustment = 1.0
            if portfolio_heat > self.max_portfolio_heat:
                heat_adjustment = max(0.3, 0.8 - (portfolio_heat - self.max_portfolio_heat) * 2)  # Reduce positions
                logger.warning(f"🔥 Portfolio heat too high: {portfolio_heat:.3f}, reducing position sizes by {(1-heat_adjustment)*100:.0f}%")
            
            final_size = correlation_adjusted_size * heat_adjustment
            
            # 4. Ensure minimum and maximum bounds
            final_size = max(25.0, min(500.0, final_size))  # Between $25-$500
            
            return {
                'original_size': base_size,
                'volatility_adjusted_size': volatility_adjusted_size,
                'correlation_adjustment': correlation_adjustment,
                'heat_adjustment': heat_adjustment,
                'final_size': final_size,
                'portfolio_heat': portfolio_heat,
                'reasoning': {
                    'volatility_factor': volatility_factor,
                    'correlation_factor': correlation_adjustment,
                    'heat_factor': heat_adjustment,
                    'total_adjustment': final_size / base_size
                }
            }
            
        except Exception as e:
            logger.error(f"Error optimizing position size for {symbol}: {e}")
            return {
                'original_size': base_size,
                'final_size': base_size,
                'portfolio_heat': 0.0,
                'reasoning': {'error': str(e)}
            }

# Global risk manager instance
risk_manager = AdvancedRiskManager()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "advanced-risk-management", "timestamp": datetime.now().isoformat()}

@app.post("/optimize_position_size", response_model=RiskAdjustment)
async def optimize_position_size_endpoint(request: TradeRequest):
    """Optimize position size based on risk factors"""
    try:
        result = risk_manager.optimize_position_size(
            request.symbol,
            request.base_size,
            request.current_positions
        )
        
        return RiskAdjustment(
            original_size=result['original_size'],
            final_size=result['final_size'],
            volatility_factor=result['reasoning'].get('volatility_factor', 1.0),
            correlation_factor=result['reasoning'].get('correlation_factor', 1.0),
            heat_factor=result['reasoning'].get('heat_factor', 1.0),
            portfolio_heat=result['portfolio_heat'],
            reasoning=result['reasoning']
        )
        
    except Exception as e:
        logger.error(f"Error in position size optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio_heat")
async def get_portfolio_heat(positions: Dict = None):
    """Get current portfolio heat level"""
    try:
        if positions is None:
            positions = {}
        
        heat = risk_manager.calculate_portfolio_heat(positions)
        return {
            'portfolio_heat': heat,
            'max_heat_threshold': risk_manager.max_portfolio_heat,
            'heat_level': 'HIGH' if heat > risk_manager.max_portfolio_heat else 'NORMAL',
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error calculating portfolio heat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/asset_volatility/{symbol}")
async def get_asset_volatility_endpoint(symbol: str):
    """Get volatility for a specific asset"""
    try:
        volatility = risk_manager.get_asset_volatility(symbol)
        return {
            'symbol': symbol,
            'volatility': volatility,
            'lookback_days': risk_manager.volatility_lookback,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting volatility for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/risk_parameters")
async def get_risk_parameters():
    """Get current risk management parameters"""
    return {
        'max_portfolio_heat': risk_manager.max_portfolio_heat,
        'volatility_lookback': risk_manager.volatility_lookback,
        'correlation_threshold': risk_manager.correlation_threshold,
        'service': 'advanced-risk-management',
        'timestamp': datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "advanced_risk_service:app",
        host="0.0.0.0",
        port=8027,
        log_level="info"
    )
