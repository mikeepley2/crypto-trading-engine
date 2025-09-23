#!/usr/bin/env python3
"""
Portfolio Management Service
Handles all portfolio-related operations including positions, performance, and analytics
Port: 8026
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import mysql.connector
from mysql.connector import Error as MySQLError
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Response
from pydantic import BaseModel
import uvicorn
from advanced_rebalancer import AdvancedPortfolioRebalancer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('portfolio_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'news_collector',
    'password': '99Rules!',
    'database': 'crypto_transactions',
    'autocommit': True,
    'charset': 'utf8mb4'
}

# Pydantic models
class PortfolioPosition(BaseModel):
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    allocation_percent: float
    last_updated: datetime = None

class PortfolioSummary(BaseModel):
    total_value: float
    cash_balance: float
    invested_amount: float
    unrealized_pnl: float
    realized_pnl: float
    num_positions: int
    positions: List[PortfolioPosition]
    performance_metrics: Dict
    last_updated: datetime

class PortfolioPerformance(BaseModel):
    total_return: float
    total_return_percent: float
    daily_return: float
    daily_return_percent: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int

class PortfolioUpdate(BaseModel):
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    net_amount: float
    fees: float = 0.0

# Portfolio Management Service
class PortfolioManager:
    def __init__(self):
        self.db_connection = None
        self.connect_db()
        
    def connect_db(self):
        """Connect to MySQL database"""
        try:
            self.db_connection = mysql.connector.connect(**DB_CONFIG)
            logger.info("✅ Connected to crypto_transactions database")
        except MySQLError as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from external API"""
        try:
            # Clean symbol for API calls
            clean_symbol = symbol.replace('-USD', '').replace('-USDT', '')
            
            # Try Coinbase first
            async with httpx.AsyncClient() as client:
                response = await client.get(f"https://api.coinbase.com/v2/prices/{clean_symbol}-USD/spot", timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    return float(data['data']['amount'])
                    
                # Fallback to Binance
                response = await client.get(f"https://api.binance.com/api/v3/ticker/price?symbol={clean_symbol}USDT", timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    return float(data['price'])
                    
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch price for {symbol}: {e}")
            
        return None

    def get_portfolio_positions(self) -> List[PortfolioPosition]:
        """Get all current portfolio positions"""
        try:
            cursor = self.db_connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM mock_holdings WHERE quantity > 0")
            holdings = cursor.fetchall()
            cursor.close()
            
            positions = []
            for holding in holdings:
                position = PortfolioPosition(
                    symbol=holding['symbol'],
                    quantity=float(holding['quantity']),
                    avg_entry_price=float(holding['avg_entry_price']),
                    current_price=float(holding.get('last_price', 0)),
                    market_value=float(holding.get('position_value', 0)),
                    unrealized_pnl=float(holding.get('unrealized_pnl', 0)),
                    allocation_percent=0.0,  # Will be calculated later
                    last_updated=holding.get('updated_at', datetime.now())
                )
                positions.append(position)
                
            return positions
            
        except Exception as e:
            logger.error(f"❌ Error getting portfolio positions: {e}")
            return []

    async def update_portfolio_positions(self) -> List[PortfolioPosition]:
        """Update all portfolio positions with current prices"""
        try:
            cursor = self.db_connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM mock_holdings WHERE quantity > 0")
            holdings = cursor.fetchall()
            
            positions = []
            total_portfolio_value = 0
            
            # Get cash balance
            cursor.execute("SELECT cash_balance FROM mock_portfolio ORDER BY updated_at DESC LIMIT 1")
            portfolio_row = cursor.fetchone()
            cash_balance = float(portfolio_row['cash_balance']) if portfolio_row else 100000.0
            total_portfolio_value += cash_balance
            
            # Update each position
            for holding in holdings:
                symbol = holding['symbol']
                quantity = float(holding['quantity'])
                avg_entry_price = float(holding['avg_entry_price'])
                total_invested = float(holding['total_invested'])
                
                # Get current price
                current_price = await self.get_current_price(symbol)
                if not current_price:
                    logger.warning(f"⚠️ Could not get price for {symbol}, skipping update")
                    continue
                
                # Calculate values
                market_value = quantity * current_price
                unrealized_pnl = market_value - total_invested
                total_portfolio_value += market_value
                
                # Update database
                cursor.execute("""
                    UPDATE mock_holdings 
                    SET last_price = %s, position_value = %s, unrealized_pnl = %s, updated_at = NOW()
                    WHERE symbol = %s
                """, (current_price, market_value, unrealized_pnl, symbol))
                
                position = PortfolioPosition(
                    symbol=symbol,
                    quantity=quantity,
                    avg_entry_price=avg_entry_price,
                    current_price=current_price,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    allocation_percent=0.0,  # Will be calculated after we know total value
                    last_updated=datetime.now()
                )
                positions.append(position)
            
            # Calculate allocation percentages
            for position in positions:
                if total_portfolio_value > 0:
                    position.allocation_percent = (position.market_value / total_portfolio_value) * 100
            
            cursor.close()
            logger.info(f"✅ Updated {len(positions)} portfolio positions")
            return positions
            
        except Exception as e:
            logger.error(f"❌ Error updating portfolio positions: {e}")
            return []

    def get_portfolio_summary(self) -> Dict:
        """Get complete portfolio summary"""
        try:
            cursor = self.db_connection.cursor(dictionary=True)
            
            # Get portfolio summary
            cursor.execute("SELECT * FROM mock_portfolio ORDER BY updated_at DESC LIMIT 1")
            portfolio = cursor.fetchone()
            
            if not portfolio:
                # Initialize portfolio if not exists
                cursor.execute("""
                    INSERT INTO mock_portfolio (total_value, cash_balance, invested_amount, unrealized_pnl, realized_pnl)
                    VALUES (100000.0, 100000.0, 0.0, 0.0, 0.0)
                """)
                cursor.execute("SELECT * FROM mock_portfolio ORDER BY updated_at DESC LIMIT 1")
                portfolio = cursor.fetchone()
            
            # Get current positions
            positions = self.get_portfolio_positions()
            
            # Calculate totals
            cash_balance = float(portfolio['cash_balance'])
            invested_amount = sum(pos.market_value for pos in positions)
            total_value = cash_balance + invested_amount
            unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
            realized_pnl = float(portfolio.get('realized_pnl', 0))
            
            # Update allocation percentages
            for position in positions:
                if total_value > 0:
                    position.allocation_percent = (position.market_value / total_value) * 100
            
            cursor.close()
            
            return {
                'total_value': total_value,
                'cash_balance': cash_balance,
                'invested_amount': invested_amount,
                'unrealized_pnl': unrealized_pnl,
                'realized_pnl': realized_pnl,
                'num_positions': len(positions),
                'positions': positions,
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting portfolio summary: {e}")
            return None

    def update_holdings(self, symbol: str, side: str, quantity: float, price: float, net_amount: float, fees: float = 0.0):
        """Update holdings after trade execution"""
        logger.info(f"🔔 Updating holdings: {symbol} {side} qty={quantity:.8f} price=${price:.2f} amount=${net_amount:.2f}")
        
        try:
            cursor = self.db_connection.cursor()
            
            if side.upper() == 'BUY':
                # Check if holding exists
                cursor.execute("SELECT quantity, total_invested FROM mock_holdings WHERE symbol = %s", (symbol,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing holding
                    old_qty = float(existing[0])
                    old_invested = float(existing[1])
                    new_qty = old_qty + quantity
                    new_invested = old_invested + net_amount
                    new_avg_price = new_invested / new_qty if new_qty > 0 else price
                    
                    cursor.execute("""
                        UPDATE mock_holdings 
                        SET quantity = %s, avg_entry_price = %s, total_invested = %s, updated_at = NOW()
                        WHERE symbol = %s
                    """, (new_qty, new_avg_price, new_invested, symbol))
                    logger.info(f"✅ Updated existing holding: {symbol} new_qty={new_qty:.8f}")
                    
                else:
                    # Insert new holding
                    cursor.execute("""
                        INSERT INTO mock_holdings 
                        (symbol, quantity, avg_entry_price, total_invested, realized_pnl, unrealized_pnl, total_pnl, last_price, position_value)
                        VALUES (%s, %s, %s, %s, 0.0, 0.0, 0.0, 0.0, 0.0)
                    """, (symbol, quantity, price, net_amount))
                    logger.info(f"✅ Inserted new holding: {symbol} qty={quantity:.8f} price=${price:.2f}")
                    
                # Update cash balance (subtract purchase amount)
                cursor.execute("""
                    UPDATE mock_portfolio 
                    SET cash_balance = cash_balance - %s, updated_at = NOW()
                    ORDER BY updated_at DESC LIMIT 1
                """, (net_amount,))
                
            elif side.upper() == 'SELL':
                # Check if holding exists
                cursor.execute("SELECT quantity, total_invested FROM mock_holdings WHERE symbol = %s", (symbol,))
                existing = cursor.fetchone()
                
                if existing:
                    old_qty = float(existing[0])
                    old_invested = float(existing[1])
                    
                    if old_qty >= quantity:
                        # Calculate realized PnL
                        avg_cost_basis = old_invested / old_qty if old_qty > 0 else 0
                        cost_of_sold = avg_cost_basis * quantity
                        realized_pnl = net_amount - cost_of_sold
                        
                        new_qty = old_qty - quantity
                        new_invested = old_invested - cost_of_sold
                        
                        if new_qty > 0:
                            cursor.execute("""
                                UPDATE mock_holdings 
                                SET quantity = %s, total_invested = %s, realized_pnl = realized_pnl + %s, updated_at = NOW()
                                WHERE symbol = %s
                            """, (new_qty, new_invested, realized_pnl, symbol))
                        else:
                            # Close position completely
                            cursor.execute("""
                                UPDATE mock_holdings 
                                SET quantity = 0, total_invested = 0, realized_pnl = realized_pnl + %s, updated_at = NOW()
                                WHERE symbol = %s
                            """, (realized_pnl, symbol))
                        
                        logger.info(f"✅ Updated sell holding: {symbol} new_qty={new_qty:.8f} realized_pnl=${realized_pnl:.2f}")
                        
                        # Update cash balance (add sale proceeds)
                        cursor.execute("""
                            UPDATE mock_portfolio 
                            SET cash_balance = cash_balance + %s, realized_pnl = realized_pnl + %s, updated_at = NOW()
                            ORDER BY updated_at DESC LIMIT 1
                        """, (net_amount, realized_pnl))
                    else:
                        logger.error(f"❌ Insufficient quantity for {symbol}: have {old_qty}, trying to sell {quantity}")
                else:
                    logger.error(f"❌ No holding found for {symbol} to sell")
                    
            cursor.close()
            logger.info(f"✅ Holdings update completed for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Error updating holdings: {e}")

    def calculate_performance_metrics(self, days: int = 30) -> Dict:
        """Calculate portfolio performance metrics"""
        try:
            cursor = self.db_connection.cursor(dictionary=True)
            
            # Get recent performance history
            cursor.execute("""
                SELECT * FROM mock_performance_history 
                WHERE date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
                ORDER BY date DESC
            """, (days,))
            performance_data = cursor.fetchall()
            
            # Get trade statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as total_profits,
                    SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) as total_losses,
                    AVG(pnl) as avg_pnl
                FROM mock_trades 
                WHERE created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
            """, (days,))
            trade_stats = cursor.fetchone()
            
            cursor.close()
            
            # Calculate metrics
            metrics = {
                'total_return': 0.0,
                'total_return_percent': 0.0,
                'daily_return': 0.0,
                'daily_return_percent': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'num_trades': trade_stats['total_trades'] if trade_stats else 0
            }
            
            if performance_data and len(performance_data) > 1:
                latest = performance_data[0]
                current_value = float(latest['total_value'])
                
                # Reset baseline: Use current portfolio value as new starting point
                baseline_value = 2571.86  # Current portfolio value becomes new baseline
                
                metrics['total_return'] = current_value - baseline_value
                metrics['total_return_percent'] = (metrics['total_return'] / baseline_value * 100) if baseline_value > 0 else 0
                
                if len(performance_data) >= 2:
                    yesterday = performance_data[1]
                    yesterday_value = float(yesterday['total_value'])
                    metrics['daily_return'] = current_value - yesterday_value
                    metrics['daily_return_percent'] = (metrics['daily_return'] / yesterday_value * 100) if yesterday_value > 0 else 0
            
            if trade_stats and trade_stats['total_trades'] > 0:
                metrics['win_rate'] = (trade_stats['winning_trades'] / trade_stats['total_trades']) * 100
                
                if trade_stats['total_losses'] > 0:
                    metrics['profit_factor'] = trade_stats['total_profits'] / trade_stats['total_losses']
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating performance metrics: {e}")
            return {}

# FastAPI app
try:
    from ..shared.metrics import snapshot, to_prometheus
    from ..shared.auth import api_key_required
except ImportError:
    from backend.services.trading.shared.metrics import snapshot, to_prometheus  # type: ignore
    from backend.services.trading.shared.auth import api_key_required  # type: ignore

app = FastAPI(title="Portfolio Management Service", version="1.0.0")

# Global portfolio manager instance
portfolio_manager = PortfolioManager()

@app.get("/health")
async def health_check(dep=Depends(api_key_required)):
    """Health check endpoint"""
    return {"status": "healthy", "service": "portfolio_manager", "metrics": snapshot()}

@app.get("/metrics")
def metrics(dep=Depends(api_key_required)):
    return Response(content=to_prometheus(), media_type="text/plain; version=0.0.4")

@app.get("/portfolio", response_model=Dict)
async def get_portfolio(dep=Depends(api_key_required)):
    """Get complete portfolio summary"""
    try:
        portfolio_data = portfolio_manager.get_portfolio_summary()
        if portfolio_data:
            return {
                "status": "success",
                "portfolio": portfolio_data
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to retrieve portfolio data")
            
    except Exception as e:
        logger.error(f"❌ Error in get_portfolio endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/positions")
async def get_positions(dep=Depends(api_key_required)):
    """Get all current positions"""
    try:
        positions = await portfolio_manager.update_portfolio_positions()
        return {
            "status": "success",
            "positions": [pos.dict() for pos in positions],
            "count": len(positions)
        }
        
    except Exception as e:
        logger.error(f"❌ Error in get_positions endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance")
async def get_performance(days: int = 30, dep=Depends(api_key_required)):
    """Get portfolio performance metrics"""
    try:
        metrics = portfolio_manager.calculate_performance_metrics(days)
        return {
            "status": "success",
            "performance": metrics,
            "period_days": days
        }
        
    except Exception as e:
        logger.error(f"❌ Error in get_performance endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/holdings/update")
async def update_holdings(update: PortfolioUpdate, dep=Depends(api_key_required)):
    """Update holdings after trade execution"""
    try:
        portfolio_manager.update_holdings(
            symbol=update.symbol,
            side=update.side,
            quantity=update.quantity,
            price=update.price,
            net_amount=update.net_amount,
            fees=update.fees
        )
        
        return {
            "status": "success",
            "message": f"Holdings updated for {update.symbol}"
        }
        
    except Exception as e:
        logger.error(f"❌ Error updating holdings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/portfolio/refresh")
async def refresh_portfolio(dep=Depends(api_key_required)):
    """Refresh all portfolio positions with current prices"""
    try:
        positions = await portfolio_manager.update_portfolio_positions()
        portfolio_data = portfolio_manager.get_portfolio_summary()
        
        return {
            "status": "success",
            "message": f"Portfolio refreshed with {len(positions)} positions",
            "portfolio": portfolio_data
        }
        
    except Exception as e:
        logger.error(f"❌ Error refreshing portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Advanced Rebalancing Endpoints
@app.get("/rebalance/analysis")
async def get_rebalancing_analysis():
    """Get comprehensive portfolio rebalancing analysis"""
    try:
        rebalancer = AdvancedPortfolioRebalancer(DB_CONFIG)
        analysis = rebalancer.get_rebalancing_analysis()
        
        return {
            "status": "success",
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting rebalancing analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rebalance/generate-signals")
async def generate_rebalancing_signals():
    """Generate trading signals based on portfolio rebalancing analysis"""
    try:
        rebalancer = AdvancedPortfolioRebalancer(DB_CONFIG)
        success = rebalancer.create_rebalancing_signals()
        
        if success:
            return {
                "status": "success",
                "message": "Rebalancing signals generated successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "message": "Failed to generate rebalancing signals"
            }
        
    except Exception as e:
        logger.error(f"❌ Error generating rebalancing signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rebalance/recommendations")
async def get_rebalancing_recommendations():
    """Get current rebalancing recommendations without generating signals"""
    try:
        rebalancer = AdvancedPortfolioRebalancer(DB_CONFIG)
        positions = rebalancer.get_portfolio_positions()
        recommendations = rebalancer.generate_rebalancing_recommendations(positions)
        
        return {
            "status": "success",
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "high_priority": len([r for r in recommendations if r['priority'] == 'high']),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting rebalancing recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/risk/concentration")
async def get_concentration_analysis():
    """Get portfolio concentration risk analysis"""
    try:
        rebalancer = AdvancedPortfolioRebalancer(DB_CONFIG)
        positions = rebalancer.get_portfolio_positions()
        concentration_analysis = rebalancer.analyze_concentration_risk(positions)
        
        return {
            "status": "success",
            "concentration_analysis": concentration_analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting concentration analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("🚀 Starting Portfolio Management Service on port 8026")
    uvicorn.run(app, host="0.0.0.0", port=8026)
