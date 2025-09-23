#!/usr/bin/env python3
"""
Fixed Mock Trading Engine - Integrated with Portfolio Management Service
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
    'database': 'crypto_analysis',
    'autocommit': True  # Auto-commit each statement
}

# Service URLs
PORTFOLIO_SERVICE_URL = "http://portfolio-service:8026"  # Docker service name
PORTFOLIO_SERVICE_URL_LOCAL = "http://localhost:8026"    # For local testing

# Trading configuration
TRADING_CONFIG = {
    'min_trade_amount': 100.0,
    'max_trade_amount': 10000.0,
    'trading_fee_percent': 0.1,
    'max_daily_trades': 50,
    'min_confidence_threshold': 0.6,
    'max_portfolio_allocation': 80.0,
    'base_position_size_percent': 2.0
}

# Request models
class TradeRequest(BaseModel):
    symbol: str
    side: str  # 'BUY' or 'SELL'
    amount: float
    price: float = 0  # 0 for market price
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
        self.daily_trade_count = 0
        self.connect_database()
        logger.info("🤖 Mock Trading Engine initialized")

    def connect_database(self):
        """Connect to MySQL database"""
        try:
            self.db_connection = mysql.connector.connect(**MYSQL_CONFIG)
            logger.info("✅ Database connected successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False

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
                "fees": abs(net_amount) * 0.001  # 0.1% fee
            }
            
            result = await self.call_portfolio_service("/holdings/update", method="POST", data=update_data)
            if result and result.get("status") == "success":
                logger.info(f"✅ Holdings updated via portfolio service: {symbol}")
                return True
            else:
                logger.error(f"❌ Portfolio service update failed for {symbol}")
                return self._fallback_update_holdings(symbol, side, quantity, price, net_amount)
                
        except Exception as e:
            logger.error(f"❌ Error updating holdings via service: {e}")
            return self._fallback_update_holdings(symbol, side, quantity, price, net_amount)

    def _fallback_update_holdings(self, symbol: str, side: str, quantity: float, price: float, net_amount: float):
        """Fallback holdings update via direct database"""
        logger.info(f"🔄 Using fallback holdings update for {symbol}")
        try:
            cursor = self.db_connection.cursor()
            
            if side.upper() == 'BUY':
                cursor.execute("SELECT quantity, total_invested FROM mock_holdings WHERE symbol = %s", (symbol,))
                existing = cursor.fetchone()
                
                if existing:
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
                else:
                    cursor.execute("""
                        INSERT INTO mock_holdings 
                        (symbol, quantity, avg_entry_price, total_invested, realized_pnl, unrealized_pnl, total_pnl, last_price, position_value)
                        VALUES (%s, %s, %s, %s, 0.0, 0.0, 0.0, 0.0, 0.0)
                    """, (symbol, quantity, price, net_amount))
                
                cursor.execute("""
                    UPDATE mock_portfolio 
                    SET cash_balance = cash_balance - %s, updated_at = NOW()
                    ORDER BY updated_at DESC LIMIT 1
                """, (net_amount,))
                
            cursor.close()
            logger.info(f"✅ Fallback holdings update completed for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Fallback holdings update failed: {e}")
            return False

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol"""
        try:
            # Try Coinbase API
            url = f"https://api.coinbase.com/v2/exchange-rates?currency={symbol.split('-')[0]}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                usd_rate = float(data['data']['rates']['USD'])
                logger.info(f"💰 Real-time price for {symbol}: ${usd_rate:,.2f}")
                return usd_rate
        except Exception as e:
            logger.error(f"❌ Price fetch failed for {symbol}: {e}")
        
        return None

    def can_execute_trade(self, trade_request: TradeRequest) -> Tuple[bool, str]:
        """Check if trade can be executed"""
        # Basic validation
        if trade_request.amount < TRADING_CONFIG['min_trade_amount']:
            return False, f"Trade amount too small: ${trade_request.amount:.2f} < ${TRADING_CONFIG['min_trade_amount']:.2f}"
        
        if trade_request.amount > TRADING_CONFIG['max_trade_amount']:
            return False, f"Trade amount too large: ${trade_request.amount:.2f} > ${TRADING_CONFIG['max_trade_amount']:.2f}"
        
        return True, "Trade can be executed"

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
                logger.error(f"❌ Failed to update holdings for {trade_request.symbol}")
                return {
                    'success': False,
                    'reason': 'Failed to update portfolio holdings',
                    'trade_id': None
                }
            
            # Record trade in database
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO mock_trades (symbol, side, quantity, price, amount, fee, net_amount, status, execution_time, confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'FILLED', NOW(), %s)
            """, (trade_request.symbol, trade_request.side, quantity, execution_price, 
                  trade_request.amount, fee, net_amount, trade_request.confidence))
            trade_id = cursor.lastrowid
            cursor.close()
            
            self.daily_trade_count += 1
            
            logger.info(f"✅ Trade executed: {trade_request.side} {quantity:.8f} {trade_request.symbol} @ ${execution_price:.2f}")
            
            return {
                'success': True,
                'trade_id': trade_id,
                'symbol': trade_request.symbol,
                'side': trade_request.side,
                'quantity': quantity,
                'executed_price': execution_price,
                'amount': trade_request.amount,
                'fee': fee,
                'net_amount': net_amount,
                'reason': 'Trade executed successfully'
            }
            
        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
            return {
                'success': False,
                'reason': str(e),
                'trade_id': None
            }
        
        return None

    def get_portfolio_info(self) -> Dict:
        """Get current portfolio information"""
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
            
            # Get current holdings
            cursor.execute("SELECT * FROM mock_holdings WHERE quantity > 0")
            holdings = cursor.fetchall()
            
            logger.info(f"🔍 Found {len(holdings)} holdings in database")
            
            # Calculate current portfolio value
            total_value = float(portfolio['cash_balance'])
            positions = []
            
            for holding in holdings:
                current_price = self.get_current_price(holding['symbol'])
                if current_price:
                    qty = float(holding['quantity'])
                    entry_price = float(holding['avg_entry_price'])
                    invested = float(holding['total_invested'])
                    market_value = qty * float(current_price)
                    unrealized_pnl = market_value - invested
                    
                    position = PortfolioPosition(
                        symbol=holding['symbol'],
                        quantity=qty,
                        avg_entry_price=entry_price,
                        current_price=float(current_price),
                        market_value=market_value,
                        unrealized_pnl=unrealized_pnl,
                        allocation_percent=(market_value / total_value) * 100 if total_value > 0 else 0
                    )
                    positions.append(position)
                    total_value += market_value
            
            cursor.close()
            
            return {
                'total_value': total_value,
                'cash_balance': float(portfolio['cash_balance']),
                'invested_amount': sum(pos.market_value for pos in positions),
                'total_pnl': sum(pos.unrealized_pnl for pos in positions),
                'positions': positions,
                'num_positions': len(positions)
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting portfolio info: {e}")
            return None

    def update_holdings(self, symbol: str, side: str, quantity: float, price: float, net_amount: float):
        """Update holdings after trade execution - SIMPLIFIED VERSION"""
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
                    
            elif side.upper() == 'SELL':
                # Update existing holding (reduce quantity)
                cursor.execute("""
                    UPDATE mock_holdings 
                    SET quantity = quantity - %s,
                        realized_pnl = realized_pnl + (%s - (avg_entry_price * %s)),
                        updated_at = NOW()
                    WHERE symbol = %s
                """, (quantity, net_amount, quantity, symbol))
                
                # Remove if quantity becomes zero or negative
                cursor.execute("DELETE FROM mock_holdings WHERE symbol = %s AND quantity <= 0", (symbol,))
                logger.info(f"✅ Updated holding for SELL: {symbol}")
            
            # Verify the operation
            cursor.execute("SELECT quantity, avg_entry_price, total_invested FROM mock_holdings WHERE symbol = %s", (symbol,))
            verification = cursor.fetchone()
            if verification:
                logger.info(f"🔍 Verified holding: {symbol} qty={verification[0]} price=${verification[1]:.2f} invested=${verification[2]:.2f}")
            else:
                logger.info(f"🔍 No holding found after {side} operation for {symbol}")
            
            cursor.close()
            logger.info(f"📊 Holdings operation completed for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Error updating holdings for {symbol}: {e}")
            raise

    def can_execute_trade(self, trade_request: TradeRequest) -> Tuple[bool, str]:
        """Check if trade can be executed"""
        # Check daily trade limit
        if self.daily_trade_count >= TRADING_CONFIG['max_daily_trades']:
            return False, f"Daily trade limit reached ({TRADING_CONFIG['max_daily_trades']})"
        
        # Check minimum trade amount
        if trade_request.amount < TRADING_CONFIG['min_trade_amount']:
            return False, f"Trade amount too small: ${trade_request.amount:.2f} < ${TRADING_CONFIG['min_trade_amount']:.2f}"
        
        # Check maximum trade amount
        if trade_request.amount > TRADING_CONFIG['max_trade_amount']:
            return False, f"Trade amount too large: ${trade_request.amount:.2f} > ${TRADING_CONFIG['max_trade_amount']:.2f}"
        
        return True, "Trade can be executed"

    def execute_trade(self, trade_request: TradeRequest) -> Dict:
        """Execute a trade"""
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
            
            # Update holdings
            self.update_holdings(trade_request.symbol, trade_request.side, quantity, execution_price, net_amount)
            
            # Update cash balance
            cursor = self.db_connection.cursor()
            if trade_request.side.upper() == 'BUY':
                cursor.execute("""
                    UPDATE mock_portfolio 
                    SET cash_balance = cash_balance - %s, updated_at = NOW()
                    ORDER BY updated_at DESC LIMIT 1
                """, (trade_request.amount,))
            else:  # SELL
                cursor.execute("""
                    UPDATE mock_portfolio 
                    SET cash_balance = cash_balance + %s, updated_at = NOW()
                    ORDER BY updated_at DESC LIMIT 1
                """, (net_amount,))
            cursor.close()
            
            # Record trade
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO mock_trades (symbol, side, quantity, price, amount, fee, net_amount, status, execution_time, confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'FILLED', NOW(), %s)
            """, (trade_request.symbol, trade_request.side, quantity, execution_price, 
                  trade_request.amount, fee, net_amount, trade_request.confidence))
            trade_id = cursor.lastrowid
            cursor.close()
            
            self.daily_trade_count += 1
            
            logger.info(f"✅ Trade executed: {trade_request.side} {quantity:.8f} {trade_request.symbol} @ ${execution_price:.2f}")
            
            return {
                'success': True,
                'trade_id': trade_id,
                'symbol': trade_request.symbol,
                'side': trade_request.side,
                'quantity': quantity,
                'executed_price': execution_price,
                'amount': trade_request.amount,
                'fee': fee,
                'net_amount': net_amount,
                'reason': 'Trade executed successfully'
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

@app.post("/process-recommendations")
def process_all_pending_recommendations():
    """Process all pending recommendations"""
    if not trading_engine:
        raise HTTPException(status_code=500, detail="Trading engine not initialized")
    
    try:
        # Fetch all pending recommendations
        cursor = trading_engine.db_connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM trade_recommendations 
            WHERE status = 'pending' 
            AND confidence >= %s
            ORDER BY generated_at ASC
        """, (TRADING_CONFIG['min_confidence_threshold'],))
        recommendations = cursor.fetchall()
        cursor.close()
        
        processed = 0
        failed = 0
        results = []
        
        for rec in recommendations:
            try:
                # Use the single recommendation processing logic
                response = process_recommendation(rec['id'])
                processed += 1
                results.append({
                    "rec_id": rec['id'],
                    "status": "success",
                    "trade_id": response.get("trade", {}).get("trade_id")
                })
            except Exception as e:
                failed += 1
                results.append({
                    "rec_id": rec['id'],
                    "status": "failed",
                    "error": str(e)
                })
                logger.error(f"❌ Failed to process recommendation {rec['id']}: {e}")
        
        return {
            "status": "completed",
            "total_recommendations": len(recommendations),
            "processed": processed,
            "failed": failed,
            "results": results
        }
    
    except Exception as e:
        logger.error(f"❌ Error processing recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Fixed Mock Trading Engine on port 8021...")
    uvicorn.run(app, host="0.0.0.0", port=8021)
