#!/usr/bin/env python3
import os
import logging
import time
import json
import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title='Risk Management Service', version='1.0.0')

# Configuration
SERVICE_PORT = int(os.getenv('SERVICE_PORT', '8027'))
MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '1000.0'))  # Max USD per position
MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', '500.0'))  # Max daily loss in USD
MAX_PORTFOLIO_RISK = float(os.getenv('MAX_PORTFOLIO_RISK', '0.05'))  # 5% max portfolio risk

class RiskAssessment(BaseModel):
    recommendation_id: int
    symbol: str
    signal_type: str
    amount_usd: float
    risk_score: float
    risk_level: str
    approved: bool
    reason: str
    suggested_amount: Optional[float] = None

def get_db_connection():
    """Get database connection"""
    return mysql.connector.connect(
        host=os.getenv('DB_HOST', '172.22.32.1'),
        user=os.getenv('DB_USER', 'news_collector'),
        password=os.getenv('DB_PASSWORD'),
        database='crypto_prices'
    )

def calculate_position_risk(symbol: str, amount_usd: float) -> Dict[str, Any]:
    """Calculate position-specific risk metrics using advanced analytics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get recent price data for volatility calculation
        cursor.execute('''
            SELECT price, volume_24h, price_change_24h, market_cap
            FROM crypto_prices 
            WHERE symbol = %s 
            ORDER BY timestamp DESC 
            LIMIT 24
        ''', (symbol,))
        
        price_data = cursor.fetchall()
        conn.close()
        
        if not price_data:
            return {
                'volatility': 0.0,
                'volume_score': 0.0,
                'price_stability': 0.0,
                'risk_score': 0.5,
                'var_95': 0.0,
                'correlation_risk': 0.0
            }
        
        # Convert to DataFrame for advanced analysis
        df = pd.DataFrame(price_data)
        
        # Calculate volatility (standard deviation of price changes)
        if len(df) > 1:
            price_changes = df['price'].pct_change().dropna()
            volatility = price_changes.std() if not price_changes.empty else 0.0
            
            # Calculate Value at Risk (VaR) at 95% confidence
            var_95 = np.percentile(price_changes, 5) if not price_changes.empty else 0.0
        else:
            volatility = 0.0
            var_95 = 0.0
        
        # Calculate volume score (normalized with trend analysis)
        current_volume = df['volume_24h'].iloc[0] if not df.empty else 0
        avg_volume = df['volume_24h'].mean() if not df.empty else 0
        volume_trend = df['volume_24h'].pct_change().mean() if len(df) > 1 else 0
        
        # Enhanced volume score considering trend
        volume_score = min(current_volume / avg_volume if avg_volume > 0 else 0, 2.0) / 2.0
        volume_score = volume_score * (1 + volume_trend)  # Adjust for trend
        
        # Calculate price stability with momentum analysis
        recent_change = abs(df['price_change_24h'].iloc[0]) if not df.empty else 0
        price_stability = max(0, 1 - (recent_change / 100))  # Normalize to 0-1
        
        # Calculate correlation risk (simplified - would need multiple symbols for full correlation)
        correlation_risk = 0.0  # Placeholder for correlation analysis
        
        # Advanced risk score calculation
        risk_score = (
            volatility * 0.3 +           # Price volatility
            (1 - volume_score) * 0.25 +  # Volume risk
            (1 - price_stability) * 0.25 +  # Price stability
            abs(var_95) * 0.1 +          # Value at Risk
            correlation_risk * 0.1       # Correlation risk
        )
        risk_score = min(max(risk_score, 0), 1)  # Clamp to 0-1
        
        return {
            'volatility': float(volatility),
            'volume_score': float(volume_score),
            'price_stability': float(price_stability),
            'risk_score': float(risk_score),
            'var_95': float(var_95),
            'correlation_risk': float(correlation_risk)
        }
        
    except Exception as e:
        logger.error(f"Error calculating position risk for {symbol}: {e}")
        return {
            'volatility': 0.5,
            'volume_score': 0.5,
            'price_stability': 0.5,
            'risk_score': 0.5,
            'var_95': 0.0,
            'correlation_risk': 0.0
        }

def check_portfolio_limits(symbol: str, amount_usd: float) -> Dict[str, Any]:
    """Check portfolio-level risk limits with advanced analytics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get current portfolio value (sum of all positions)
        cursor.execute('''
            SELECT SUM(amount_usd) as total_portfolio_value
            FROM trade_recommendations 
            WHERE execution_status = 'EXECUTED' 
            AND created_at >= CURDATE()
        ''')
        
        portfolio_result = cursor.fetchone()
        total_portfolio = portfolio_result['total_portfolio_value'] if portfolio_result['total_portfolio_value'] else 0
        
        # Get current position size for this symbol
        cursor.execute('''
            SELECT SUM(amount_usd) as current_position
            FROM trade_recommendations 
            WHERE symbol = %s 
            AND execution_status = 'EXECUTED' 
            AND created_at >= CURDATE()
        ''', (symbol,))
        
        position_result = cursor.fetchone()
        current_position = position_result['current_position'] if position_result['current_position'] else 0
        
        # Get daily P&L with detailed breakdown
        cursor.execute('''
            SELECT 
                SUM(CASE WHEN signal_type = 'BUY' THEN -amount_usd ELSE 0 END) as total_buys,
                SUM(CASE WHEN signal_type = 'SELL' THEN amount_usd ELSE 0 END) as total_sells,
                COUNT(*) as total_trades
            FROM trade_recommendations 
            WHERE execution_status = 'EXECUTED' 
            AND created_at >= CURDATE()
        ''')
        
        pnl_result = cursor.fetchone()
        daily_pnl = (pnl_result['total_sells'] or 0) - (pnl_result['total_buys'] or 0)
        total_trades = pnl_result['total_trades'] or 0
        
        # Get portfolio diversification metrics
        cursor.execute('''
            SELECT symbol, SUM(amount_usd) as position_size
            FROM trade_recommendations 
            WHERE execution_status = 'EXECUTED' 
            AND created_at >= CURDATE()
            GROUP BY symbol
        ''')
        
        positions = cursor.fetchall()
        conn.close()
        
        # Calculate diversification metrics
        if positions:
            position_df = pd.DataFrame(positions)
            herfindahl_index = (position_df['position_size'] ** 2).sum() / (position_df['position_size'].sum() ** 2)
            diversification_score = 1 - herfindahl_index  # Higher is more diversified
        else:
            diversification_score = 1.0  # No positions = fully diversified
        
        # Check limits
        new_position_size = current_position + amount_usd
        portfolio_risk = new_position_size / max(total_portfolio, 1)
        
        # Advanced risk checks
        exceeds_position_limit = new_position_size > MAX_POSITION_SIZE
        exceeds_daily_loss = daily_pnl < -MAX_DAILY_LOSS
        exceeds_portfolio_risk = portfolio_risk > MAX_PORTFOLIO_RISK
        exceeds_diversification_limit = diversification_score < 0.3  # Require at least 30% diversification
        
        return {
            'total_portfolio': total_portfolio,
            'current_position': current_position,
            'new_position_size': new_position_size,
            'daily_pnl': daily_pnl,
            'portfolio_risk': portfolio_risk,
            'diversification_score': diversification_score,
            'total_trades': total_trades,
            'exceeds_position_limit': exceeds_position_limit,
            'exceeds_daily_loss': exceeds_daily_loss,
            'exceeds_portfolio_risk': exceeds_portfolio_risk,
            'exceeds_diversification_limit': exceeds_diversification_limit
        }
        
    except Exception as e:
        logger.error(f"Error checking portfolio limits: {e}")
        return {
            'total_portfolio': 0,
            'current_position': 0,
            'new_position_size': amount_usd,
            'daily_pnl': 0,
            'portfolio_risk': 0,
            'diversification_score': 1.0,
            'total_trades': 0,
            'exceeds_position_limit': False,
            'exceeds_daily_loss': False,
            'exceeds_portfolio_risk': False,
            'exceeds_diversification_limit': False
        }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "risk-management"}

@app.post("/assess/{recommendation_id}")
async def assess_risk(recommendation_id: int):
    """Assess risk for a trade recommendation with advanced analytics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get the recommendation
        cursor.execute('''
            SELECT id, symbol, signal_type, amount_usd, confidence
            FROM trade_recommendations 
            WHERE id = %s
        ''', (recommendation_id,))
        
        recommendation = cursor.fetchone()
        if not recommendation:
            return {"status": "error", "message": "Recommendation not found"}
        
        symbol = recommendation['symbol']
        signal_type = recommendation['signal_type']
        amount_usd = recommendation['amount_usd']
        confidence = recommendation['confidence']
        
        logger.info(f"Assessing risk for recommendation {recommendation_id}: {symbol} {signal_type} ${amount_usd}")
        
        # Calculate advanced position risk
        position_risk = calculate_position_risk(symbol, amount_usd)
        
        # Check portfolio limits with advanced analytics
        portfolio_limits = check_portfolio_limits(symbol, amount_usd)
        
        # Determine overall risk level with confidence adjustment
        risk_score = position_risk['risk_score']
        
        # Advanced confidence adjustment based on signal strength
        confidence_adjustment = (confidence - 0.5) * 0.3  # -0.15 to +0.15
        risk_score = max(0, min(1, risk_score - confidence_adjustment))
        
        # Determine risk level with more granular classification
        if risk_score < 0.2:
            risk_level = "VERY_LOW"
        elif risk_score < 0.4:
            risk_level = "LOW"
        elif risk_score < 0.6:
            risk_level = "MEDIUM"
        elif risk_score < 0.8:
            risk_level = "HIGH"
        else:
            risk_level = "VERY_HIGH"
        
        # Advanced approval logic
        approved = True
        reason = "Risk assessment passed"
        suggested_amount = amount_usd
        
        # Check various risk limits with detailed reasoning
        if portfolio_limits['exceeds_position_limit']:
            approved = False
            reason = f"Position size ${portfolio_limits['new_position_size']:.2f} exceeds limit ${MAX_POSITION_SIZE}"
            suggested_amount = max(0, MAX_POSITION_SIZE - portfolio_limits['current_position'])
        elif portfolio_limits['exceeds_daily_loss']:
            approved = False
            reason = f"Daily loss ${abs(portfolio_limits['daily_pnl']):.2f} exceeds limit ${MAX_DAILY_LOSS}"
        elif portfolio_limits['exceeds_portfolio_risk']:
            approved = False
            reason = f"Portfolio risk {portfolio_limits['portfolio_risk']:.2%} exceeds limit {MAX_PORTFOLIO_RISK:.2%}"
        elif portfolio_limits['exceeds_diversification_limit']:
            approved = False
            reason = f"Portfolio diversification score {portfolio_limits['diversification_score']:.2%} below minimum 30%"
        elif risk_level in ["HIGH", "VERY_HIGH"] and amount_usd > MAX_POSITION_SIZE * 0.5:
            approved = False
            reason = f"{risk_level} risk trade with large amount ${amount_usd:.2f}"
            suggested_amount = MAX_POSITION_SIZE * 0.2  # Reduce to 20% of max for high risk
        elif position_risk['var_95'] < -0.1:  # VaR indicates >10% potential loss
            approved = False
            reason = f"Value at Risk indicates potential loss >10% (VaR: {position_risk['var_95']:.2%})"
            suggested_amount = amount_usd * 0.5  # Reduce position by 50%
        
        # Create comprehensive risk assessment
        assessment = RiskAssessment(
            recommendation_id=recommendation_id,
            symbol=symbol,
            signal_type=signal_type,
            amount_usd=amount_usd,
            risk_score=risk_score,
            risk_level=risk_level,
            approved=approved,
            reason=reason,
            suggested_amount=suggested_amount if not approved else None
        )
        
        # Update database with comprehensive risk assessment
        cursor.execute('''
            UPDATE trade_recommendations 
            SET risk_score = %s,
                risk_level = %s,
                risk_approved = %s,
                risk_reason = %s,
                risk_assessment_timestamp = NOW()
            WHERE id = %s
        ''', (
            risk_score,
            risk_level,
            approved,
            reason,
            recommendation_id
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Advanced risk assessment completed for recommendation {recommendation_id}: {risk_level} risk, {'APPROVED' if approved else 'REJECTED'}")
        
        return {
            "status": "success",
            "recommendation_id": recommendation_id,
            "assessment": assessment.dict(),
            "position_risk": position_risk,
            "portfolio_limits": portfolio_limits
        }
        
    except Exception as e:
        logger.error(f"Error assessing risk for recommendation {recommendation_id}: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/status")
async def get_status():
    """Get service status and comprehensive risk metrics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get comprehensive risk statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_assessments,
                SUM(CASE WHEN risk_approved = 1 THEN 1 ELSE 0 END) as approved_count,
                AVG(risk_score) as avg_risk_score,
                COUNT(CASE WHEN risk_level = 'VERY_LOW' THEN 1 END) as very_low_count,
                COUNT(CASE WHEN risk_level = 'LOW' THEN 1 END) as low_count,
                COUNT(CASE WHEN risk_level = 'MEDIUM' THEN 1 END) as medium_count,
                COUNT(CASE WHEN risk_level = 'HIGH' THEN 1 END) as high_count,
                COUNT(CASE WHEN risk_level = 'VERY_HIGH' THEN 1 END) as very_high_count
            FROM trade_recommendations 
            WHERE risk_assessment_timestamp >= NOW() - INTERVAL 1 HOUR
        ''')
        
        stats = cursor.fetchone()
        conn.close()
        
        return {
            "status": "healthy",
            "service": "risk-management",
            "version": "1.0.0-advanced",
            "limits": {
                "max_position_size": MAX_POSITION_SIZE,
                "max_daily_loss": MAX_DAILY_LOSS,
                "max_portfolio_risk": MAX_PORTFOLIO_RISK
            },
            "statistics": {
                "total_assessments": stats[0] if stats else 0,
                "approved_count": stats[1] if stats else 0,
                "avg_risk_score": float(stats[2]) if stats and stats[2] else 0.0,
                "risk_distribution": {
                    "very_low": stats[3] if stats else 0,
                    "low": stats[4] if stats else 0,
                    "medium": stats[5] if stats else 0,
                    "high": stats[6] if stats else 0,
                    "very_high": stats[7] if stats else 0
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == '__main__':
    logger.info(f'Starting Advanced Risk Management Service on port {SERVICE_PORT}')
    uvicorn.run(app, host='0.0.0.0', port=SERVICE_PORT)

