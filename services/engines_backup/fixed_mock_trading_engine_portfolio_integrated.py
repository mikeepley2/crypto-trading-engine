#!/usr/bin/env python3
"""
Fixed Mock Trading Engine - Integrated with Portfolio Management Service
Uses portfolio service for all portfolio operations instead of direct database access
"""

import logging
import time
import requests
import httpx
import asyncio
import mysql.connector
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration - SIMPLIFIED
MYSQL_CONFIG = {
    'host': 'host.docker.internal',
    'user': 'news_collector', 
    'password': '99Rules!',
    'port': 3306,
    'database': 'crypto_transactions',
    'autocommit': True  # Auto-commit each statement
}

# Portfolio service URLs
PORTFOLIO_SERVICE_URL = "http://portfolio-service:8026"  # Docker service name
PORTFOLIO_SERVICE_URL_LOCAL = "http://localhost:8026"    # For local testing

# Trading configuration
TRADING_CONFIG = {
    'trading_fee_percent': 0.5,     # 0.5% trading fee
    'min_trade_amount': 10.0,       # Minimum $10 trade
    'max_trade_amount': 10000.0,    # Maximum $10k trade
    'default_position_size_percent': 2.0,  # 2% of portfolio per trade
    'max_position_size_percent': 10.0,     # Max 10% in single position
    'daily_trade_limit': 10,        # Max 10 trades per day
    'max_portfolio_allocation': 80.0,      # Max 80% of portfolio invested
}

# Pydantic models
class TradeRequest(BaseModel):
    symbol: str
    side: str  # 'BUY' or 'SELL'
    amount: float
    price: Optional[float] = 0  # 0 for market order
    confidence: Optional[float] = None

class PortfolioPosition(BaseModel):
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    allocation_percent: float

class MockTradingEngine:
    def __init__(self):
        self.db_connection = None
        self.connect_database()
        
    def connect_database(self):
        """Connect to MySQL database"""
        try:
            self.db_connection = mysql.connector.connect(**MYSQL_CONFIG)
            logger.info("✅ Connected to crypto_transactions database")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise

    async def call_portfolio_service(self, endpoint: str, method: str = "GET", data: Dict = None) -> Optional[Dict]:
        """Call portfolio service endpoint"""
        try:
            # Try Docker service name first, fallback to localhost
            for base_url in [PORTFOLIO_SERVICE_URL, PORTFOLIO_SERVICE_URL_LOCAL]:
                try:
                    async with httpx.AsyncClient() as client:
                        if method == "GET":
                            response = await client.get(f"{base_url}{endpoint}", timeout=10.0)
                        elif method == "POST":
                            response = await client.post(f"{base_url}{endpoint}", json=data, timeout=10.0)
                        
                        if response.status_code == 200:
                            return response.json()
                        
                except Exception as e:
                    logger.warning(f"⚠️ Portfolio service call failed for {base_url}: {e}")
                    continue
                    
            logger.error(f"❌ All portfolio service endpoints failed for {endpoint}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Portfolio service call error: {e}")
            return None

    async def get_portfolio_info(self) -> Dict:
        """Get portfolio information from portfolio service"""
        try:
            result = await self.call_portfolio_service("/portfolio")
            if result and result.get("status") == "success":
                return result.get("portfolio", {})
            else:
                logger.error("❌ Failed to get portfolio from service")
                return self._fallback_portfolio_info()
                
        except Exception as e:
            logger.error(f"❌ Error getting portfolio info: {e}")
            return self._fallback_portfolio_info()

    def _fallback_portfolio_info(self) -> Dict:
        """Fallback portfolio info from direct database query"""
        try:
            cursor = self.db_connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM mock_portfolio ORDER BY updated_at DESC LIMIT 1")
            portfolio = cursor.fetchone()
            cursor.close()
            
            if portfolio:
                return {
                    'total_value': float(portfolio['total_value']),
                    'cash_balance': float(portfolio['cash_balance']),
                    'invested_amount': float(portfolio.get('invested_amount', 0)),
                    'unrealized_pnl': float(portfolio.get('unrealized_pnl', 0)),
                    'realized_pnl': float(portfolio.get('realized_pnl', 0)),
                    'positions': [],
                    'num_positions': 0
                }
            else:
                return {
                    'total_value': 100000.0,
                    'cash_balance': 100000.0,
                    'invested_amount': 0.0,
                    'unrealized_pnl': 0.0,
                    'realized_pnl': 0.0,
                    'positions': [],
                    'num_positions': 0
                }
                
        except Exception as e:
            logger.error(f"❌ Fallback portfolio info failed: {e}")
            return {'total_value': 0, 'cash_balance': 0, 'positions': [], 'num_positions': 0}

    async def update_holdings(self, symbol: str, side: str, quantity: float, price: float, net_amount: float):
        """Update holdings via portfolio service"""
        logger.info(f"🔔 Updating holdings via portfolio service: {symbol} {side} qty={quantity:.8f} price=${price:.2f}")
        
        try:
            update_data = {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "net_amount": net_amount,
                "fees": 0.0
            }
            
            result = await self.call_portfolio_service("/holdings/update", method="POST", data=update_data)
            if result and result.get("status") == "success":
                logger.info(f"✅ Holdings updated via portfolio service: {symbol}")
                return True
            else:
                logger.error(f"❌ Portfolio service holdings update failed: {result}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error updating holdings via portfolio service: {e}")
            return False

    def can_execute_trade(self, trade_request: TradeRequest) -> Tuple[bool, str]:
        """Check if trade can be executed"""
        try:
            # Basic validation
            if not trade_request.symbol or not trade_request.side:
                return False, "Invalid symbol or side"
            
            if trade_request.side not in ['BUY', 'SELL']:
                return False, "Side must be BUY or SELL"
            
            if trade_request.amount <= 0:
                return False, "Amount must be positive"
            
            if trade_request.amount < TRADING_CONFIG['min_trade_amount']:
                return False, f"Amount below minimum ${TRADING_CONFIG['min_trade_amount']}"
            
            if trade_request.amount > TRADING_CONFIG['max_trade_amount']:
                return False, f"Amount above maximum ${TRADING_CONFIG['max_trade_amount']}"
            
            return True, "Trade validation passed"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def get_current_price(self, symbol):
        """Get current market price"""
        try:
            # Clean symbol for API calls
            clean_symbol = symbol.replace('-USD', '').replace('-USDT', '')
            
            # Try multiple price sources
            try:
                # Coinbase Pro
                response = requests.get(f"https://api.coinbase.com/v2/prices/{clean_symbol}-USD/spot", timeout=3)
                if response.status_code == 200:
                    return float(response.json()['data']['amount'])
            except:
                pass
                
            try:
                # Binance
                response = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={clean_symbol}USDT", timeout=3)
                if response.status_code == 200:
                    return float(response.json()['price'])
            except:
                pass
                
        except Exception as e:
            logger.warning(f"Failed to get price for {symbol}: {e}")
            
        # Return mock price if all fails
        return 45000.0 if symbol in ['BTC', 'BTC-USD'] else 3000.0

    def record_trade(self, trade_request: TradeRequest, execution_price: float, quantity: float, fee: float, net_amount: float) -> Optional[int]:
        """Record trade in database"""
        try:
            cursor = self.db_connection.cursor()
            
            insert_query = """
                INSERT INTO mock_trades 
                (symbol, action, quantity, price, amount, fee, net_amount, confidence, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """
            
            values = (
                trade_request.symbol,
                trade_request.side,
                quantity,
                execution_price,
                trade_request.amount,
                fee,
                net_amount,
                trade_request.confidence or 0.0
            )
            
            cursor.execute(insert_query, values)
            trade_id = cursor.lastrowid
            cursor.close()
            
            logger.info(f"✅ Trade recorded with ID: {trade_id}")
            return trade_id
            
        except Exception as e:
            logger.error(f"❌ Failed to record trade: {e}")
            return None

    async def execute_trade(self, trade_request: TradeRequest) -> Dict:
        """Execute a trade using portfolio service"""
        try:
            # Validate trade
            can_execute, reason = self.can_execute_trade(trade_request)
            if not can_execute:
                logger.warning(f"❌ Trade rejected: {reason}")
                return {
                    'success': False,
                    'reason': reason,
                    'trade_id': None
                }
            
            # Get current price
            current_price = self.get_current_price(trade_request.symbol)
            if not current_price:
                return {
                    'success': False,
                    'reason': f'Could not get current price for {trade_request.symbol}',
                    'trade_id': None
                }
            
            # Calculate execution details
            execution_price = float(current_price)
            quantity = trade_request.amount / execution_price
            fee = trade_request.amount * (TRADING_CONFIG['trading_fee_percent'] / 100)
            net_amount = trade_request.amount - fee
            
            # Update holdings via portfolio service
            holdings_updated = await self.update_holdings(trade_request.symbol, trade_request.side, quantity, execution_price, net_amount)
            if not holdings_updated:
                return {
                    'success': False,
                    'reason': 'Failed to update holdings via portfolio service',
                    'trade_id': None
                }
            
            # Record trade
            trade_id = self.record_trade(trade_request, execution_price, quantity, fee, net_amount)
            
            logger.info(f"✅ Trade executed via portfolio service: {trade_request.side} {quantity:.8f} {trade_request.symbol} at ${execution_price:.2f}")
            
            return {
                'success': True,
                'symbol': trade_request.symbol,
                'side': trade_request.side,
                'quantity': quantity,
                'execution_price': execution_price,
                'amount': trade_request.amount,
                'fee': fee,
                'net_amount': net_amount,
                'trade_id': trade_id
            }
            
        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
            return {
                'success': False,
                'reason': str(e),
                'trade_id': None
            }

# FastAPI App
app = FastAPI(title="Fixed Mock Trading Engine", version="1.0.0")
trading_engine = None

@app.on_event("startup")
async def startup_event():
    global trading_engine
    try:
        trading_engine = MockTradingEngine()
        logger.info("🚀 Fixed Mock Trading Engine started successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to start trading engine: {e}")
        raise

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "fixed_mock_trading_engine"}

@app.get("/portfolio")
async def get_portfolio():
    """Get current portfolio information via portfolio service"""
    if not trading_engine:
        raise HTTPException(status_code=500, detail="Trading engine not initialized")
    
    portfolio = await trading_engine.get_portfolio_info()
    if not portfolio:
        raise HTTPException(status_code=500, detail="Failed to get portfolio information")
    
    return {"status": "success", "portfolio": portfolio}

@app.post("/execute_trade")
async def execute_trade_endpoint(trade_request: TradeRequest):
    """Execute a trade using portfolio service"""
    if not trading_engine:
        raise HTTPException(status_code=500, detail="Trading engine not initialized")
    
    result = await trading_engine.execute_trade(trade_request)
    
    if not result['success']:
        raise HTTPException(status_code=400, detail=result['reason'])
    
    return {"status": "success", "trade": result}

@app.post("/process-recommendation/{rec_id}")
async def process_recommendation(rec_id: int):
    """Process a trade recommendation by ID using portfolio service"""
    if not trading_engine:
        raise HTTPException(status_code=500, detail="Trading engine not initialized")
    
    try:
        # Fetch recommendation from database
        cursor = trading_engine.db_connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM trade_recommendations 
            WHERE id = %s AND status = 'pending'
        """, (rec_id,))
        recommendation = cursor.fetchone()
        cursor.close()
        
        if not recommendation:
            raise HTTPException(status_code=404, detail=f"No pending recommendation found with ID {rec_id}")
        
        # Convert recommendation to trade request
        current_price = trading_engine.get_current_price(recommendation['symbol'])
        if not current_price:
            raise HTTPException(status_code=400, detail=f"Could not get current price for {recommendation['symbol']}")
        
        # Calculate trade amount from position size percentage
        portfolio = await trading_engine.get_portfolio_info()
        if not portfolio:
            raise HTTPException(status_code=500, detail="Could not get portfolio information")
        
        position_size_percent = float(recommendation.get('position_size_percent', 2.0))
        trade_amount = float(portfolio['total_value']) * (position_size_percent / 100.0)
        
        # Ensure trade amount is within limits
        trade_amount = max(TRADING_CONFIG['min_trade_amount'], 
                          min(trade_amount, TRADING_CONFIG['max_trade_amount']))
        
        # Create trade request
        trade_request = TradeRequest(
            symbol=recommendation['symbol'],
            side=recommendation['action'],
            amount=trade_amount,
            price=0,  # Market price
            confidence=float(recommendation.get('confidence', 0.0)) if recommendation.get('confidence') else None
        )
        
        # Execute the trade
        result = await trading_engine.execute_trade(trade_request)
        
        if result['success']:
            # Update recommendation status
            cursor = trading_engine.db_connection.cursor()
            cursor.execute("""
                UPDATE trade_recommendations 
                SET status = 'executed', executed_at = NOW()
                WHERE id = %s
            """, (rec_id,))
            cursor.close()
            
            logger.info(f"✅ Successfully processed recommendation {rec_id}: {trade_request.side} {trade_request.symbol}")
            return {
                "status": "success", 
                "recommendation_id": rec_id,
                "trade": result
            }
        else:
            # Update recommendation status to failed
            cursor = trading_engine.db_connection.cursor()
            cursor.execute("""
                UPDATE trade_recommendations 
                SET status = 'failed', executed_at = NOW()
                WHERE id = %s
            """, (rec_id,))
            cursor.close()
            
            raise HTTPException(status_code=400, detail=f"Trade execution failed: {result['reason']}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing recommendation {rec_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error processing recommendation: {str(e)}")

if __name__ == "__main__":
    logger.info("🚀 Starting Fixed Mock Trading Engine on port 8021")
    uvicorn.run(app, host="0.0.0.0", port=8021)
