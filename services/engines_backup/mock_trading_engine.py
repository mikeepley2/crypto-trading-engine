#!/usr/bin/env python3

import mysql.connector
import requests
import logging
import time
import json
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
import sys
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/mnt/e/git/aitest/backend/services/trading_engine/mock_trading_engine.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Database configuration
MYSQL_CONFIG = {
    'host': 'host.docker.internal',
    'user': 'news_collector', 
    'password': '99Rules!',
    'port': 3306,
    'database': 'crypto_analysis',
    'autocommit': True
}

# Trading configuration
TRADING_CONFIG = {
    'default_position_size_percent': 2.0,  # 2% of portfolio per trade
    'max_position_size_percent': 10.0,     # Max 10% in single position
    'min_trade_amount': 100.0,             # Minimum $100 trade
    'max_portfolio_allocation': 80.0,      # Max 80% of portfolio invested
    'trading_fee_percent': 0.1,            # 0.1% trading fee
    'slippage_percent': 0.05,              # 0.05% slippage
    'min_confidence_threshold': 0.7,       # Minimum confidence to execute
    'max_daily_trades': 20,                # Risk management
    'rebalance_threshold': 5.0,            # Rebalance if allocation off by 5%
}

@dataclass
class TradeRequest:
    """Trade request data structure"""
    symbol: str
    side: str  # BUY or SELL
    amount: float
    price: float
    confidence: float = None
    regime: str = None
    recommendation_id: int = None
    notes: str = None

@dataclass
class PortfolioPosition:
    """Portfolio position data structure"""
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    allocation_percent: float

class MockTradingEngine:
    """Mock trading execution engine that processes trade recommendations"""
    
    def __init__(self):
        self.db_connection = None
        self.real_time_provider = None
        
        # Performance tracking
        self.performance = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_volume': 0.0,
            'total_fees': 0.0,
            'start_time': datetime.now()
        }
        
        # Trading state
        self.daily_trade_count = 0
        self.last_portfolio_update = None
        
        logger.info("🤖 Mock Trading Engine initialized")
    
    def connect_database(self):
        """Connect to MySQL database"""
        try:
            self.db_connection = mysql.connector.connect(**MYSQL_CONFIG)
            self.db_connection.autocommit = True
            logger.info("✅ Database connected successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol"""
        try:
            # Try to get from real-time APIs
            price = self._fetch_real_time_price(symbol)
            if price:
                return price
            
            # Fallback to database
            return self._get_db_price(symbol)
            
        except Exception as e:
            logger.error(f"❌ Error getting price for {symbol}: {e}")
            return None
    
    def _fetch_real_time_price(self, symbol: str) -> Optional[float]:
        """Fetch real-time price from external APIs"""
        try:
            # Try Coinbase first
            # Handle symbol format (use as-is if already includes -USD, otherwise add it)
            if symbol.endswith('-USD'):
                coinbase_symbol = symbol
            else:
                coinbase_symbol = f"{symbol}-USD"
            url = f"https://api.exchange.coinbase.com/products/{coinbase_symbol}/ticker"
            
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                price = float(data['price'])
                logger.info(f"💰 Real-time price for {symbol}: ${price:,.2f}")
                return price
                
        except Exception as e:
            logger.debug(f"Real-time price fetch failed for {symbol}: {e}")
        
        return None
    
    def _get_db_price(self, symbol: str) -> Optional[float]:
        """Get latest price from database"""
        try:
            cursor = self.db_connection.cursor()
            
            # Get from crypto_prices database
            cursor.execute("USE crypto_prices")
            cursor.execute("""
                SELECT current_price 
                FROM price_data 
                WHERE symbol = %s 
                ORDER BY timestamp_iso DESC 
                LIMIT 1
            """, (symbol,))
            
            result = cursor.fetchone()
            cursor.execute("USE crypto_transactions")  # Switch back
            cursor.close()
            
            if result:
                price = float(result[0])
                logger.info(f"📊 DB price for {symbol}: ${price:,.2f}")
                return price
                
        except Exception as e:
            logger.error(f"❌ DB price fetch failed for {symbol}: {e}")
        
        return None
    
    def get_portfolio_info(self) -> Dict:
        """Get current portfolio information"""
        try:
            cursor = self.db_connection.cursor(dictionary=True)
            # Get portfolio summary
            cursor.execute("SELECT * FROM mock_portfolio ORDER BY updated_at DESC LIMIT 1")
            portfolio = cursor.fetchone()
            # Get current holdings
            cursor.execute("SELECT * FROM mock_holdings WHERE quantity > 0")
            holdings = cursor.fetchall()
            logger.info(f"🔍 Raw holdings query result: {holdings}")
            
            # Also check all holdings (including zero quantity)
            cursor.execute("SELECT symbol, quantity, avg_entry_price, total_invested FROM mock_holdings")
            all_holdings = cursor.fetchall()
            logger.info(f"🔍 All holdings in DB: {all_holdings}")
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
    
    def can_execute_trade(self, trade_request: TradeRequest) -> Tuple[bool, str]:
        """Check if trade can be executed based on risk management rules"""
        
        portfolio = self.get_portfolio_info()
        if not portfolio:
            return False, "Cannot get portfolio information"
        
        # Check daily trade limit
        if self.daily_trade_count >= TRADING_CONFIG['max_daily_trades']:
            return False, f"Daily trade limit reached ({TRADING_CONFIG['max_daily_trades']})"
        
        # Check minimum confidence
        if trade_request.confidence and trade_request.confidence < TRADING_CONFIG['min_confidence_threshold']:
            return False, f"Confidence too low: {trade_request.confidence:.2f} < {TRADING_CONFIG['min_confidence_threshold']:.2f}"
        
        # Check minimum trade amount
        if trade_request.amount < TRADING_CONFIG['min_trade_amount']:
            return False, f"Trade amount too small: ${trade_request.amount:.2f} < ${TRADING_CONFIG['min_trade_amount']:.2f}"
        
        if trade_request.side == 'BUY':
            # Check available cash
            required_cash = trade_request.amount * (1 + TRADING_CONFIG['trading_fee_percent'] / 100)
            if required_cash > portfolio['cash_balance']:
                return False, f"Insufficient cash: ${required_cash:.2f} required, ${portfolio['cash_balance']:.2f} available"
            
            # Check max portfolio allocation
            total_after_trade = portfolio['invested_amount'] + trade_request.amount
            allocation_percent = (total_after_trade / portfolio['total_value']) * 100
            if allocation_percent > TRADING_CONFIG['max_portfolio_allocation']:
                return False, f"Portfolio allocation limit: {allocation_percent:.1f}% > {TRADING_CONFIG['max_portfolio_allocation']:.1f}%"
        
        elif trade_request.side == 'SELL':
            # Check if we have the position
            position = next((p for p in portfolio['positions'] if p.symbol == trade_request.symbol), None)
            if not position:
                return False, f"No position in {trade_request.symbol} to sell"
            
            # Check if we have enough quantity
            quantity_to_sell = trade_request.amount / trade_request.price
            if quantity_to_sell > position.quantity:
                return False, f"Insufficient quantity: {quantity_to_sell:.8f} required, {position.quantity:.8f} available"
        
        return True, "Trade can be executed"
    
    def calculate_position_size(self, symbol: str, price: float, confidence: float = None) -> float:
        """Calculate optimal position size based on portfolio and confidence"""
        
        portfolio = self.get_portfolio_info()
        if not portfolio:
            return 0.0
        
        # Base position size as percentage of total portfolio
        base_size_percent = TRADING_CONFIG['default_position_size_percent']
        
        # Adjust based on confidence
        if confidence:
            # Higher confidence = larger position (up to max)
            confidence_multiplier = min(confidence / 0.8, 1.5)  # Cap at 1.5x
            adjusted_size_percent = base_size_percent * confidence_multiplier
        else:
            adjusted_size_percent = base_size_percent
        
        # Cap at maximum position size
        adjusted_size_percent = min(adjusted_size_percent, TRADING_CONFIG['max_position_size_percent'])
        
        # Calculate dollar amount
        target_amount = portfolio['total_value'] * (adjusted_size_percent / 100)
        
        # Ensure minimum trade amount
        target_amount = max(target_amount, TRADING_CONFIG['min_trade_amount'])
        
        # Ensure we don't exceed available cash (with buffer for fees)
        max_amount = portfolio['cash_balance'] * 0.95  # 5% buffer
        target_amount = min(target_amount, max_amount)
        
        logger.info(f"💡 Position size for {symbol}: ${target_amount:,.2f} ({adjusted_size_percent:.1f}% of portfolio)")
        return target_amount
    
    def execute_trade(self, trade_request: TradeRequest) -> Dict:
        """Execute a mock trade"""
        
        try:
            # Validate trade
            can_trade, reason = self.can_execute_trade(trade_request)
            if not can_trade:
                logger.warning(f"⚠️ Trade rejected: {reason}")
                return {
                    'success': False,
                    'reason': reason,
                    'trade_id': None
                }
            
            # Calculate trade details
            quantity = trade_request.amount / trade_request.price
            fee = trade_request.amount * (TRADING_CONFIG['trading_fee_percent'] / 100)
            slippage = trade_request.amount * (TRADING_CONFIG['slippage_percent'] / 100)
            
            # Adjust for slippage
            if trade_request.side == 'BUY':
                actual_price = trade_request.price * (1 + TRADING_CONFIG['slippage_percent'] / 100)
                net_amount = trade_request.amount + fee
            else:
                actual_price = trade_request.price * (1 - TRADING_CONFIG['slippage_percent'] / 100)
                net_amount = trade_request.amount - fee
            
            # Execute the trade
            trade_id = self._record_trade(trade_request, actual_price, quantity, fee, net_amount)
            
            if trade_id:
                logger.info(f"🔔 Calling _update_holdings for {trade_request.symbol}, {trade_request.side}, qty={quantity}, price={actual_price}, net_amount={net_amount}")
                self._update_holdings(trade_request.symbol, trade_request.side, quantity, actual_price, net_amount)
                logger.info(f"🔔 Returned from _update_holdings for {trade_request.symbol}")
                # Update portfolio
                self._update_portfolio()
                
                # Mark recommendation as executed
                if trade_request.recommendation_id:
                    self._mark_recommendation_executed(trade_request.recommendation_id, actual_price)
                
                # Update performance tracking
                self.performance['total_trades'] += 1
                self.performance['successful_trades'] += 1
                self.performance['total_volume'] += trade_request.amount
                self.performance['total_fees'] += fee
                self.daily_trade_count += 1
                
                logger.info(f"✅ Trade executed: {trade_request.side} ${trade_request.amount:,.2f} {trade_request.symbol} @ ${actual_price:,.2f}")
                
                return {
                    'success': True,
                    'trade_id': trade_id,
                    'executed_price': actual_price,
                    'quantity': quantity,
                    'fee': fee,
                    'net_amount': net_amount,
                    'reason': 'Trade executed successfully'
                }
            else:
                return {
                    'success': False,
                    'reason': 'Failed to record trade',
                    'trade_id': None
                }
                
        except Exception as e:
            logger.error(f"❌ Trade execution error: {e}")
            self.performance['failed_trades'] += 1
            return {
                'success': False,
                'reason': f"Execution error: {str(e)}",
                'trade_id': None
            }
    
    def _record_trade(self, trade_request: TradeRequest, actual_price: float, quantity: float, fee: float, net_amount: float) -> Optional[int]:
        """Record trade in database"""
        try:
            cursor = self.db_connection.cursor()
            
            insert_query = """
                INSERT INTO mock_trades 
                (recommendation_id, symbol, side, quantity, price, amount, fee, net_amount, 
                 execution_time, confidence, regime, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(insert_query, (
                trade_request.recommendation_id,
                trade_request.symbol,
                trade_request.side,
                quantity,
                actual_price,
                trade_request.amount,
                fee,
                net_amount,
                datetime.now(),
                trade_request.confidence,
                trade_request.regime,
                trade_request.notes
            ))
            
            trade_id = cursor.lastrowid
            # Note: Using autocommit mode, no manual commit needed
            cursor.close()
            
            return trade_id
            
        except Exception as e:
            logger.error(f"❌ Error recording trade: {e}")
            return None
    
    def _update_holdings(self, symbol: str, side: str, quantity: float, price: float, net_amount: float):
        # Log current database name
        try:
            cursor_db = self.db_connection.cursor()
            cursor_db.execute("SELECT DATABASE();")
            db_name = cursor_db.fetchone()[0]
            logger.info(f"🔎 Connected to database: {db_name}")
            cursor_db.close()
        except Exception as e:
            logger.error(f"❌ Error checking DB name: {e}")
        """Update holdings after trade execution"""
        import decimal
        logger.info(f"🔔 Entered _update_holdings for {symbol}, {side}, qty={quantity}, price={price}, net_amount={net_amount}")
        try:
            cursor = self.db_connection.cursor()
            # Ensure all values are Decimal for MySQL
            def round8(val):
                return decimal.Decimal(str(val)).quantize(decimal.Decimal('0.00000001'))
            d_quantity = round8(quantity)
            d_price = round8(price)
            d_net_amount = round8(net_amount)
            logger.info(f"🔍 Updating holdings: {symbol}, {side}, qty={d_quantity}, price={d_price}, net_amount={d_net_amount}")

            if side == 'BUY':
                # Check if holding exists
                cursor.execute("SELECT id, quantity, total_invested FROM mock_holdings WHERE symbol = %s", (symbol,))
                result = cursor.fetchone()
                if result is None:
                    # Insert new holding
                    try:
                        insert_query = ("INSERT INTO mock_holdings (symbol, quantity, avg_entry_price, total_invested, realized_pnl, unrealized_pnl, total_pnl, last_price, position_value) "
                                      "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)")
                        insert_values = (symbol, d_quantity, d_price, d_net_amount, 0, 0, 0, 0, 0)
                        logger.info(f"🔍 About to execute INSERT: {insert_query}")
                        logger.info(f"🔍 INSERT values: {insert_values}")
                        
                        cursor.execute(insert_query, insert_values)
                        rows_affected = cursor.rowcount
                        logger.info(f"🔍 INSERT executed - rows affected: {rows_affected}")
                        
                        if cursor.lastrowid:
                            logger.info(f"🔍 New record ID: {cursor.lastrowid}")
                        
                        # Immediately select and log holdings
                        cursor.execute("SELECT id, symbol, quantity, avg_entry_price, total_invested FROM mock_holdings WHERE symbol = %s", (symbol,))
                        new_holding = cursor.fetchone()
                        logger.info(f"🔎 New holding after insert: {new_holding}")
                        
                        # Check all holdings
                        cursor.execute("SELECT id, symbol, quantity, avg_entry_price, total_invested FROM mock_holdings")
                        all_holdings = cursor.fetchall()
                        logger.info(f"🔎 All holdings after insert: {all_holdings}")
                        
                    except Exception as e:
                        logger.error(f"❌ SQL Error in INSERT: {e}")
                        logger.error(f"❌ Error type: {type(e)}")
                        if hasattr(e, 'errno'):
                            logger.error(f"❌ MySQL Error number: {e.errno}")
                        if hasattr(e, 'sqlstate'):
                            logger.error(f"❌ SQL State: {e.sqlstate}")
                        raise
                else:
                    # Update existing holding
                    old_quantity = round8(result[1])
                    old_invested = round8(result[2])
                    new_quantity = round8(old_quantity + d_quantity)
                    new_invested = round8(old_invested + d_net_amount)
                    new_avg_price = round8(new_invested / new_quantity) if new_quantity > 0 else d_price
                    try:
                        cursor.execute("UPDATE mock_holdings SET quantity = %s, avg_entry_price = %s, total_invested = %s WHERE symbol = %s", (new_quantity, new_avg_price, new_invested, symbol))
                        logger.info(f"🔍 UPDATE executed for existing holding {symbol}")
                    except Exception as e:
                        logger.error(f"❌ SQL Error in update: {e}")

            else:  # SELL
                # Update existing holding
                cursor.execute("""
                    UPDATE mock_holdings 
                    SET quantity = quantity - %s,
                        realized_pnl = realized_pnl + (%s - (avg_entry_price * %s))
                    WHERE symbol = %s
                """, (d_quantity, d_net_amount, d_quantity, symbol))
                # Remove holding if quantity becomes zero or negative
                cursor.execute("DELETE FROM mock_holdings WHERE symbol = %s AND quantity <= 0", (symbol,))

            try:
                # Note: Using autocommit mode, no manual commit needed
                logger.info(f"🔍 Transaction auto-committed")
            except Exception as e:
                logger.error(f"❌ Auto-commit error: {e}")
            cursor.close()
            logger.info(f"📊 Holdings updated for {symbol}")

        except Exception as e:
            logger.error(f"❌ Error updating holdings: {e}")
            logger.error(f"❌ SQL Error details: {str(e)}")
            if hasattr(e, 'errno'):
                logger.error(f"❌ Error number: {e.errno}")
            # Note: Using autocommit mode, no rollback possible
            # Each statement is automatically committed
    
    def _update_portfolio(self):
        """Update portfolio summary"""
        try:
            portfolio = self.get_portfolio_info()
            if not portfolio:
                return
            
            cursor = self.db_connection.cursor()
            
            # Update portfolio summary
            cursor.execute("""
                UPDATE mock_portfolio SET
                    total_value = %s,
                    invested_amount = %s,
                    total_pnl = %s,
                    total_pnl_percent = (total_pnl / 100000.0) * 100,
                    updated_at = NOW()
                ORDER BY id DESC
                LIMIT 1
            """, (portfolio['total_value'], portfolio['invested_amount'], portfolio['total_pnl']))
            
            cursor.close()
            logger.info(f"💰 Portfolio updated: ${portfolio['total_value']:,.2f} total value")
            
        except Exception as e:
            logger.error(f"❌ Error updating portfolio: {e}")
    
    def _mark_recommendation_executed(self, recommendation_id: int, execution_price: float):
        """Mark trade recommendation as executed"""
        try:
            cursor = self.db_connection.cursor()
            
            cursor.execute("""
                UPDATE trade_recommendations SET
                    execution_status = 'EXECUTED',
                    status = 'executed',
                    executed_at = NOW(),
                    execution_price = %s,
                    execution_notes = 'Executed by mock trading engine'
                WHERE id = %s
            """, (execution_price, recommendation_id))
            
            cursor.close()
            logger.info(f"✅ Recommendation {recommendation_id} marked as executed")
            
        except Exception as e:
            logger.error(f"❌ Error marking recommendation executed: {e}")

    def _mark_recommendation_failed(self, recommendation_id: int, reason: str):
        """Mark recommendation as failed/rejected"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                UPDATE trade_recommendations SET
                    execution_status = 'REJECTED',
                    status = 'failed',
                    execution_notes = %s
                WHERE id = %s
            """, (reason[:250], recommendation_id))
            cursor.close()
            logger.info(f"⚠️ Recommendation {recommendation_id} marked as failed: {reason}")
        except Exception as e:
            logger.error(f"❌ Error marking recommendation failed: {e}")
    
    def process_pending_recommendations(self) -> List[Dict]:
        """Process all pending trade recommendations"""
        
        try:
            cursor = self.db_connection.cursor(dictionary=True)
            
            # Get pending recommendations with high confidence
            cursor.execute("""
                SELECT * FROM trade_recommendations 
                WHERE execution_status = 'PENDING' 
                AND action IN ('BUY', 'SELL')
                AND confidence >= %s
                ORDER BY generated_at ASC
                LIMIT 10
            """, (TRADING_CONFIG['min_confidence_threshold'],))
            
            recommendations = cursor.fetchall()
            cursor.close()
            
            results = []
            
            for rec in recommendations:
                # Get current price
                current_price = self.get_current_price(rec['symbol'])
                if not current_price:
                    logger.warning(f"⚠️ No price available for {rec['symbol']}, skipping")
                    continue
                
                # Calculate position size
                if rec['action'] == 'BUY':
                    amount = self.calculate_position_size(
                        rec['symbol'], 
                        current_price, 
                        float(rec['confidence']) if rec['confidence'] else None
                    )
                else:  # SELL
                    # For sell, use existing position
                    portfolio = self.get_portfolio_info()
                    position = next((p for p in portfolio['positions'] if p.symbol == rec['symbol']), None)
                    if not position:
                        logger.warning(f"⚠️ No position in {rec['symbol']} to sell")
                        continue
                    amount = position.market_value
                
                # Create trade request
                trade_request = TradeRequest(
                    symbol=rec['symbol'],
                    side=rec['action'],
                    amount=amount,
                    price=current_price,
                    confidence=float(rec['confidence']) if rec['confidence'] else None,
                    regime=rec.get('regime'),
                    recommendation_id=rec['id'],
                    notes=f"Auto-executed from recommendation {rec['id']}"
                )
                
                # Execute trade
                result = self.execute_trade(trade_request)
                result['recommendation_id'] = rec['id']
                result['symbol'] = rec['symbol']
                result['action'] = rec['action']
                
                results.append(result)
                
                # Small delay between trades
                time.sleep(1)
            
            logger.info(f"📊 Processed {len(results)} recommendations, {sum(1 for r in results if r['success'])} executed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error processing recommendations: {e}")
            return []

# FastAPI App for Mock Trading Engine
app = FastAPI(title="Mock Trading Engine", version="1.0.0")

# Global trading engine instance
trading_engine = MockTradingEngine()

@app.on_event("startup")
async def startup_event():
    """Initialize trading engine on startup"""
    if not trading_engine.connect_database():
        logger.error("❌ Failed to connect to database on startup")
        raise Exception("Database connection failed")
    logger.info("🚀 Mock Trading Engine API started")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "mock_trading_engine"}

@app.get("/status")
async def status():
    """Status endpoint with detailed service information"""
    portfolio = trading_engine.get_portfolio_info()
    return {
        "status": "operational",
        "service": "mock_trading_engine",
        "version": "1.0.0",
        "performance": trading_engine.performance,
        "portfolio_summary": {
            "total_value": portfolio['total_value'] if portfolio else 0,
            "num_positions": portfolio['num_positions'] if portfolio else 0
        } if portfolio else {"error": "portfolio_unavailable"},
        "database_connected": trading_engine.db_connection is not None,
        "daily_trades": trading_engine.daily_trade_count
    }

@app.get("/portfolio")
async def get_portfolio():
    """Get current portfolio status"""
    try:
        portfolio = trading_engine.get_portfolio_info()
        if portfolio:
            logger.info(f"🔎 Portfolio info retrieved successfully: {portfolio}")
            return {
                "status": "success",
                "portfolio": {
                    "total_value": portfolio['total_value'],
                    "cash_balance": portfolio['cash_balance'],
                    "invested_amount": portfolio['invested_amount'],
                    "total_pnl": portfolio['total_pnl'],
                    "num_positions": portfolio['num_positions'],
                    "positions": [
                        {
                            "symbol": pos.symbol,
                            "quantity": pos.quantity,
                            "avg_entry_price": pos.avg_entry_price,
                            "current_price": pos.current_price,
                            "market_value": pos.market_value,
                            "unrealized_pnl": pos.unrealized_pnl,
                            "allocation_percent": pos.allocation_percent
                        }
                        for pos in portfolio['positions']
                    ]
                }
            }
        else:
            logger.error("❌ Portfolio info is None")
            raise HTTPException(status_code=500, detail="Failed to get portfolio information")
    except Exception as e:
        logger.error(f"❌ Error in portfolio endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Cannot get portfolio information: {str(e)}")

class ManualTradeRequest(BaseModel):
    symbol: str
    side: str  # BUY or SELL
    amount: float
    confidence: Optional[float] = None
    notes: Optional[str] = None

@app.post("/execute_trade")
async def execute_manual_trade(trade: ManualTradeRequest):
    """Execute a manual trade"""
    
    # Get current price
    current_price = trading_engine.get_current_price(trade.symbol)
    if not current_price:
        raise HTTPException(status_code=400, detail=f"Cannot get price for {trade.symbol}")
    
    # Create trade request
    trade_request = TradeRequest(
        symbol=trade.symbol,
        side=trade.side,
        amount=trade.amount,
        price=current_price,
        confidence=trade.confidence,
        notes=trade.notes or "Manual trade execution"
    )
    
    # Execute trade
    result = trading_engine.execute_trade(trade_request)
    
    if result['success']:
        return {"status": "success", "trade": result}
    else:
        raise HTTPException(status_code=400, detail=result['reason'])

@app.post("/process_recommendations")
async def process_recommendations(background_tasks: BackgroundTasks):
    """Process pending trade recommendations"""
    
    def process_in_background():
        results = trading_engine.process_pending_recommendations()
        logger.info(f"📊 Background processing complete: {len(results)} recommendations processed")
    
    background_tasks.add_task(process_in_background)
    return {"status": "processing_started", "message": "Processing recommendations in background"}

@app.get("/performance")
async def get_performance():
    """Get trading performance metrics"""
    return {
        "status": "success",
        "performance": trading_engine.performance
    }

@app.get("/trades")
async def get_recent_trades(limit: int = 20):
    """Get recent trades"""
    try:
        cursor = trading_engine.db_connection.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT * FROM mock_trades 
            ORDER BY execution_time DESC 
            LIMIT %s
        """, (limit,))
        
        trades = cursor.fetchall()
        cursor.close()
        return trades
        
    except Exception as e:
        logger.error(f"❌ Error fetching trades: {e}")
        return []

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Mock Trading Engine on port 8021...")
    uvicorn.run(app, host="0.0.0.0", port=8021)
