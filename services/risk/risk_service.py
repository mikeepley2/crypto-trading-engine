#!/usr/bin/env python3
"""
Advanced Risk Management Service
Port: 8027
Provides comprehensive risk management capabilities including:
- Pre-trade risk checks
- Position sizing optimization
- Portfolio risk analysis
- Volatility-based adjustments
- Correlation analysis
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Depends, Response
from pydantic import BaseModel
import mysql.connector
from advanced_risk_manager import AdvancedRiskManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from ..shared.config import get_db_configs, get_trading_limits, feature_flags
    from ..shared.metrics import inc, snapshot, to_prometheus
    from ..shared.auth import api_key_required
except ImportError:  # direct run fallback
    from backend.services.trading.shared.config import get_db_configs, get_trading_limits, feature_flags  # type: ignore
    from backend.services.trading.shared.metrics import inc, snapshot, to_prometheus  # type: ignore
    from backend.services.trading.shared.auth import api_key_required  # type: ignore

app = FastAPI(title="Advanced Risk Management Service", version="2.0.0")
DB_CFGS = get_db_configs()
LIMITS = get_trading_limits()
FLAGS = feature_flags()

# Initialize advanced risk manager
risk_manager = AdvancedRiskManager()

# Pydantic models
class TradeCheck(BaseModel):
    symbol: str
    side: str
    amount: float  # notional USD amount
    portfolio_value: float
    existing_positions: int

class PositionSizeRequest(BaseModel):
    symbol: str
    base_size: float
    current_positions: Optional[Dict] = None
    market_regime: str = 'sideways'

class RiskAnalysisResponse(BaseModel):
    optimal_size: float
    adjustments: Dict
    risk_level: str
    warnings: List[str]

class PortfolioRiskResponse(BaseModel):
    portfolio_heat: float
    concentration_risk: float
    correlation_risk: float
    risk_score: float
    risk_level: str
    recommendations: List[str]
    total_positions: int
    total_value: float

# Risk state tracking
RISK_STATE = {
    "trades_today": 0,
    "last_reset_day": None
}

MAX_TRADES_PER_DAY = LIMITS.get("daily_trade_limit", 10)
MAX_ALLOC_PCT = LIMITS.get("max_portfolio_allocation", 80.0)
MAX_POSITION_SIZE_PCT = LIMITS.get("max_position_size_percent", 10.0)

@app.get("/health")
def health():
    try:
        # Test database connection
        cfg = DB_CFGS.get('prices', {
            'host': 'host.docker.internal',
            'user': 'news_collector',
            'password': '99Rules!',
            'database': 'crypto_prices'
        })
        connection = mysql.connector.connect(**cfg)
        connection.close()
        
        return {
            "status": "healthy",
            "service": "advanced-risk-management",
            "database": "connected",
            "metrics": snapshot(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "advanced-risk-management",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/metrics")
def metrics():
    return Response(content=to_prometheus(), media_type="text/plain; version=0.0.4")

@app.get("/risk/limits")
def limits():
    return {
        "daily_trade_limit": MAX_TRADES_PER_DAY,
        "max_portfolio_allocation": MAX_ALLOC_PCT,
        "max_position_size_percent": MAX_POSITION_SIZE_PCT,
        "max_portfolio_heat": risk_manager.MAX_PORTFOLIO_HEAT,
        "correlation_threshold": risk_manager.CORRELATION_THRESHOLD,
        "volatility_lookback_days": risk_manager.VOLATILITY_LOOKBACK,
        "timestamp": datetime.now().isoformat()
    }

def _get_portfolio_snapshot() -> Optional[Dict[str, Any]]:
    try:
        cfg = DB_CFGS['transactions']
        cnx = mysql.connector.connect(**cfg)
        cursor = cnx.cursor(dictionary=True)
        cursor.execute("SELECT * FROM mock_portfolio ORDER BY updated_at DESC LIMIT 1")
        portfolio = cursor.fetchone()
        cursor.execute("SELECT symbol, quantity, total_invested FROM mock_holdings WHERE quantity > 0")
        holdings = cursor.fetchall()
        cursor.close(); cnx.close()
        if not portfolio:
            return None
        invested = sum(float(h['total_invested']) for h in holdings)
        return {
            "cash_balance": float(portfolio.get('cash_balance', 0.0)),
            "invested_amount": invested,
            "total_value": float(portfolio.get('total_value', invested)),
            "holdings": holdings
        }
    except Exception:
        return None

@app.post("/risk/check_trade")
def check_trade(req: TradeCheck):
    """Basic trade risk check (legacy endpoint)"""
    if req.amount <= 0:
        return {"allowed": False, "reason": "Amount must be positive"}

    snap = _get_portfolio_snapshot()
    if snap:
        total_value = snap['total_value'] if snap['total_value'] > 0 else req.portfolio_value
        invested_amount = snap['invested_amount']
    else:
        total_value = req.portfolio_value
        invested_amount = req.portfolio_value * 0.0

    # Position size percent of portfolio
    pct = (req.amount / total_value * 100) if total_value > 0 else 0
    if pct > MAX_POSITION_SIZE_PCT:
        return {"allowed": False, "reason": f"Position size {pct:.2f}% exceeds max {MAX_POSITION_SIZE_PCT}%"}

    # Portfolio allocation check
    if req.side == "BUY":
        new_invested = invested_amount + req.amount
        alloc_pct = (new_invested / total_value * 100) if total_value > 0 else 0
        if alloc_pct > MAX_ALLOC_PCT:
            return {"allowed": False, "reason": f"Allocation {alloc_pct:.2f}% exceeds max {MAX_ALLOC_PCT}%"}

    return {"allowed": True, "reason": "Risk checks passed"}

@app.post("/position-size/calculate", response_model=RiskAnalysisResponse)
async def calculate_position_size(request: PositionSizeRequest):
    """Calculate optimal position size considering advanced risk factors"""
    try:
        result = risk_manager.calculate_optimal_position_size(
            symbol=request.symbol,
            base_size=request.base_size,
            current_positions=request.current_positions or {},
            market_regime=request.market_regime
        )
        
        return RiskAnalysisResponse(
            optimal_size=result['optimal_size'],
            adjustments=result['adjustments'],
            risk_level=result['risk_level'],
            warnings=result['warnings']
        )
        
    except Exception as e:
        logger.error(f"❌ Error calculating position size for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/volatility/{symbol}")
async def get_asset_volatility(symbol: str):
    """Get volatility metrics for an asset"""
    try:
        volatility = risk_manager.get_asset_volatility(symbol)
        volatility_adjustment = risk_manager.get_volatility_adjustment(symbol)
        
        return {
            "symbol": symbol,
            "volatility": volatility,
            "volatility_adjustment": volatility_adjustment,
            "risk_level": "high" if volatility > 1.0 else "medium" if volatility > 0.5 else "low",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting volatility for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/correlation/{symbol1}/{symbol2}")
async def get_asset_correlation(symbol1: str, symbol2: str):
    """Get correlation between two assets"""
    try:
        cfg = DB_CFGS.get('prices', risk_manager.db_config)
        connection = mysql.connector.connect(**cfg)
        correlation = risk_manager.calculate_asset_correlation(symbol1, symbol2, connection)
        connection.close()
        
        return {
            "symbol1": symbol1,
            "symbol2": symbol2,
            "correlation": correlation,
            "correlation_level": "high" if correlation > 0.7 else "medium" if correlation > 0.4 else "low",
            "diversification_benefit": "low" if correlation > 0.7 else "medium" if correlation > 0.4 else "high",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting correlation between {symbol1} and {symbol2}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/portfolio/risk-analysis", response_model=PortfolioRiskResponse)
async def analyze_portfolio_risk(positions: Dict):
    """Analyze comprehensive portfolio risk metrics"""
    try:
        risk_metrics = risk_manager.get_portfolio_risk_metrics(positions)
        
        return PortfolioRiskResponse(
            portfolio_heat=risk_metrics['portfolio_heat'],
            concentration_risk=risk_metrics['concentration_risk'],
            correlation_risk=risk_metrics['correlation_risk'],
            risk_score=risk_metrics['risk_score'],
            risk_level=risk_metrics['risk_level'],
            recommendations=risk_metrics['recommendations'],
            total_positions=risk_metrics['total_positions'],
            total_value=risk_metrics['total_value']
        )
        
    except Exception as e:
        logger.error(f"❌ Error analyzing portfolio risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/heat")
async def get_portfolio_heat():
    """Get current portfolio heat from actual positions"""
    try:
        # Get current positions from portfolio
        snap = _get_portfolio_snapshot()
        if not snap:
            return {
                "portfolio_heat": 0.0,
                "max_heat_threshold": risk_manager.MAX_PORTFOLIO_HEAT,
                "heat_level": "low",
                "positions_analyzed": 0,
                "error": "No portfolio data available",
                "timestamp": datetime.now().isoformat()
            }
        
        # Convert to format expected by risk manager
        positions = {}
        for holding in snap.get('holdings', []):
            positions[holding['symbol']] = {
                'value_usd': float(holding['total_invested'])
            }
        
        portfolio_heat = risk_manager.calculate_portfolio_heat(positions)
        
        return {
            "portfolio_heat": portfolio_heat,
            "max_heat_threshold": risk_manager.MAX_PORTFOLIO_HEAT,
            "heat_level": "critical" if portfolio_heat > risk_manager.MAX_PORTFOLIO_HEAT else 
                         "high" if portfolio_heat > risk_manager.MAX_PORTFOLIO_HEAT * 0.8 else
                         "moderate" if portfolio_heat > risk_manager.MAX_PORTFOLIO_HEAT * 0.5 else "low",
            "positions_analyzed": len(positions),
            "total_value": snap.get('total_value', 0),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting portfolio heat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Advanced Risk Management Service on port 8027")
    uvicorn.run(app, host="0.0.0.0", port=8027)
