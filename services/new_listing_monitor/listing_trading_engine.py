#!/usr/bin/env python3
"""
Coinbase Listing Trading Strategy Engine
Implements automated trading for new Coinbase listing opportunities

Features:
- Fast-entry execution for listing announcements
- Intelligent hold-until-plateau exit strategy
- Specialized risk management for listing trades
- Integration with existing trading infrastructure
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import mysql.connector
from mysql.connector import Error
import requests
import numpy as np
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ListingTrade:
    """Represents a new listing trading opportunity"""
    symbol: str
    entry_price: float
    entry_time: datetime
    position_size_usd: float
    listing_source: str
    confidence_score: float
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl_usd: Optional[float] = None
    status: str = "ACTIVE"  # ACTIVE, EXITED, STOPPED_OUT

class CoinbaseListingTrader:
    """Trading engine for Coinbase listing opportunities"""
    
    def __init__(self):
        self.db_config = {
            'host': '192.168.230.163',
            'user': 'news_collector',
            'password': '99Rules!',
            'database': 'crypto_transactions'
        }
        
        # Trading parameters
        self.max_position_size_usd = 250.0  # Max per listing trade
        self.max_portfolio_allocation = 0.15  # Max 15% of portfolio in listing trades
        self.stop_loss_percentage = 0.15  # 15% stop loss
        self.min_hold_time_minutes = 30  # Minimum hold time
        self.max_daily_listings = 3  # Max listing trades per day
        
        # Exit strategy parameters
        self.plateau_detection_window = 20  # Price samples to check
        self.plateau_threshold = 0.02  # 2% price stability threshold
        self.profit_target_1 = 0.50  # First profit target at 50%
        self.profit_target_2 = 0.91  # Second target at 91% (Coinbase Effect average)
        
        # Price tracking for exit strategy
        self.price_history: Dict[str, deque] = {}
        
        self.initialize_database()
    
    def initialize_database(self):
        """Initialize database tables for listing trades"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor()
            
            # Create listing trades table
            create_table_query = """
            CREATE TABLE IF NOT EXISTS listing_trades (
                id INT AUTO_INCREMENT PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                entry_price DECIMAL(20, 8) NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                position_size_usd DECIMAL(15, 2) NOT NULL,
                listing_source VARCHAR(50) NOT NULL,
                confidence_score FLOAT NOT NULL,
                exit_price DECIMAL(20, 8),
                exit_time TIMESTAMP,
                pnl_usd DECIMAL(15, 2),
                status VARCHAR(20) DEFAULT 'ACTIVE',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_symbol (symbol),
                INDEX idx_status (status),
                INDEX idx_entry_time (entry_time)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
            
            cursor.execute(create_table_query)
            connection.commit()
            logger.info("✅ Database table 'listing_trades' ready")
            
        except Error as e:
            logger.error(f"❌ Database initialization error: {e}")
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor()
            
            # Get current portfolio value from portfolio_positions or trades
            cursor.execute("""
                SELECT SUM(value_usd) as total_value 
                FROM portfolio_positions 
                WHERE created_at >= CURDATE()
                ORDER BY created_at DESC 
                LIMIT 1
            """)
            
            result = cursor.fetchone()
            return float(result[0]) if result and result[0] else 2571.86  # Default to current value
            
        except Error as e:
            logger.error(f"❌ Error getting portfolio value: {e}")
            return 2571.86  # Default fallback
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def get_active_listing_trades_value(self) -> float:
        """Get total USD value of active listing trades"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor()
            
            cursor.execute("""
                SELECT SUM(position_size_usd) as total_value 
                FROM listing_trades 
                WHERE status = 'ACTIVE'
            """)
            
            result = cursor.fetchone()
            return float(result[0]) if result and result[0] else 0.0
            
        except Error as e:
            logger.error(f"❌ Error getting active listing trades value: {e}")
            return 0.0
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def get_daily_listing_count(self) -> int:
        """Get number of listing trades today"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) as trade_count 
                FROM listing_trades 
                WHERE DATE(entry_time) = CURDATE()
            """)
            
            result = cursor.fetchone()
            return int(result[0]) if result else 0
            
        except Error as e:
            logger.error(f"❌ Error getting daily listing count: {e}")
            return 0
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def can_place_listing_trade(self, position_size_usd: float) -> Tuple[bool, str]:
        """Check if we can place a new listing trade based on risk limits"""
        
        # Check individual position size
        if position_size_usd > self.max_position_size_usd:
            return False, f"Position size ${position_size_usd} exceeds max ${self.max_position_size_usd}"
        
        # Check daily trade limit
        daily_count = self.get_daily_listing_count()
        if daily_count >= self.max_daily_listings:
            return False, f"Daily listing trade limit reached: {daily_count}/{self.max_daily_listings}"
        
        # Check portfolio allocation limit
        portfolio_value = self.get_portfolio_value()
        active_listing_value = self.get_active_listing_trades_value()
        new_total = active_listing_value + position_size_usd
        allocation_pct = new_total / portfolio_value
        
        if allocation_pct > self.max_portfolio_allocation:
            return False, f"Portfolio allocation would exceed {self.max_portfolio_allocation*100}%: {allocation_pct*100:.1f}%"
        
        return True, "OK"
    
    def calculate_position_size(self, confidence_score: float, listing_type: str) -> float:
        """Calculate optimal position size based on confidence and listing type"""
        base_size = self.max_position_size_usd
        
        # Adjust based on confidence
        confidence_multiplier = confidence_score
        
        # Adjust based on listing type
        type_multipliers = {
            "announced": 1.0,      # Full size for official announcements
            "roadmap_added": 0.7,  # Smaller size for roadmap additions
            "base_added": 0.5      # Smallest size for Base network additions
        }
        
        type_multiplier = type_multipliers.get(listing_type, 0.5)
        
        position_size = base_size * confidence_multiplier * type_multiplier
        
        # Ensure minimum viable position
        return max(position_size, 50.0)
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol from Coinbase API"""
        try:
            url = f"https://api.coinbase.com/v2/exchange-rates?currency={symbol}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'rates' in data['data']:
                    usd_rate = data['data']['rates'].get('USD')
                    if usd_rate:
                        return float(usd_rate)
            
            # Fallback to CoinGecko if Coinbase API fails
            return self.get_price_from_coingecko(symbol)
            
        except Exception as e:
            logger.error(f"❌ Error getting price for {symbol}: {e}")
            return None
    
    def get_price_from_coingecko(self, symbol: str) -> Optional[float]:
        """Fallback price source using CoinGecko"""
        try:
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': symbol.lower(),
                'vs_currencies': 'usd'
            }
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if symbol.lower() in data:
                    return float(data[symbol.lower()]['usd'])
            
        except Exception as e:
            logger.error(f"❌ CoinGecko price fallback failed for {symbol}: {e}")
        
        return None
    
    def execute_listing_buy(self, listing_signal: Dict) -> Optional[ListingTrade]:
        """Execute buy order for new listing opportunity"""
        try:
            symbol = listing_signal["symbol"]
            confidence = listing_signal["confidence"]
            listing_type = listing_signal["listing_type"]
            source = listing_signal["source"]
            
            logger.info(f"🎯 Processing listing opportunity: {symbol}")
            
            # Calculate position size
            position_size = self.calculate_position_size(confidence, listing_type)
            
            # Check if we can place the trade
            can_trade, reason = self.can_place_listing_trade(position_size)
            if not can_trade:
                logger.warning(f"⚠️ Cannot place listing trade for {symbol}: {reason}")
                return None
            
            # Get current price
            current_price = self.get_current_price(symbol)
            if not current_price:
                logger.error(f"❌ Could not get price for {symbol}")
                return None
            
            # Execute the trade (integrate with existing trading engine)
            trade_result = self.place_buy_order(symbol, position_size, current_price)
            
            if trade_result:
                # Create listing trade record
                listing_trade = ListingTrade(
                    symbol=symbol,
                    entry_price=current_price,
                    entry_time=datetime.now(),
                    position_size_usd=position_size,
                    listing_source=source,
                    confidence_score=confidence
                )
                
                # Save to database
                if self.save_listing_trade(listing_trade):
                    # Initialize price tracking
                    self.price_history[symbol] = deque(maxlen=self.plateau_detection_window)
                    self.price_history[symbol].append(current_price)
                    
                    logger.info(f"✅ LISTING TRADE EXECUTED: {symbol}")
                    logger.info(f"   Entry Price: ${current_price:.4f}")
                    logger.info(f"   Position Size: ${position_size:.2f}")
                    logger.info(f"   Expected Gain: 91% (Coinbase Effect)")
                    
                    return listing_trade
            
        except Exception as e:
            logger.error(f"❌ Error executing listing buy for {symbol}: {e}")
        
        return None
    
    def place_buy_order(self, symbol: str, position_size_usd: float, price: float) -> bool:
        """Place buy order through existing trading infrastructure"""
        try:
            # This integrates with the existing trading engine
            # For now, simulate the order execution
            
            # Calculate quantity
            quantity = position_size_usd / price
            
            # Log the simulated order
            logger.info(f"📈 Simulated BUY order: {quantity:.6f} {symbol} @ ${price:.4f}")
            logger.info(f"   Total Value: ${position_size_usd:.2f}")
            
            # In production, this would call the actual trading engine
            # e.g., coinbase_api.place_market_order(f"{symbol}-USD", "buy", str(quantity))
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error placing buy order: {e}")
            return False
    
    def save_listing_trade(self, trade: ListingTrade) -> bool:
        """Save listing trade to database"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor()
            
            insert_query = """
            INSERT INTO listing_trades 
            (symbol, entry_price, entry_time, position_size_usd, listing_source, confidence_score)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            values = (
                trade.symbol,
                trade.entry_price,
                trade.entry_time,
                trade.position_size_usd,
                trade.listing_source,
                trade.confidence_score
            )
            
            cursor.execute(insert_query, values)
            connection.commit()
            
            return True
            
        except Error as e:
            logger.error(f"❌ Error saving listing trade: {e}")
            return False
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def update_price_history(self, symbol: str, current_price: float):
        """Update price history for plateau detection"""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.plateau_detection_window)
        
        self.price_history[symbol].append(current_price)
    
    def is_price_plateau(self, symbol: str) -> bool:
        """Detect if price has plateaued (suitable for exit)"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < self.plateau_detection_window:
            return False
        
        prices = list(self.price_history[symbol])
        
        # Calculate price volatility over window
        price_changes = []
        for i in range(1, len(prices)):
            change_pct = abs(prices[i] - prices[i-1]) / prices[i-1]
            price_changes.append(change_pct)
        
        # Check if recent price changes are below threshold
        recent_volatility = np.mean(price_changes[-10:]) if len(price_changes) >= 10 else 1.0
        
        return recent_volatility < self.plateau_threshold
    
    def should_exit_position(self, trade: ListingTrade, current_price: float) -> Tuple[bool, str]:
        """Determine if position should be exited"""
        
        # Calculate current P&L
        pnl_pct = (current_price - trade.entry_price) / trade.entry_price
        
        # Check stop loss
        if pnl_pct <= -self.stop_loss_percentage:
            return True, f"STOP_LOSS triggered at {pnl_pct:.1%}"
        
        # Check minimum hold time
        hold_time = datetime.now() - trade.entry_time
        if hold_time < timedelta(minutes=self.min_hold_time_minutes):
            return False, f"Minimum hold time not reached"
        
        # Check profit targets with plateau detection
        if pnl_pct >= self.profit_target_2:  # 91% target reached
            if self.is_price_plateau(trade.symbol):
                return True, f"Target 2 reached ({pnl_pct:.1%}) with plateau detected"
        
        elif pnl_pct >= self.profit_target_1:  # 50% target reached
            if self.is_price_plateau(trade.symbol):
                return True, f"Target 1 reached ({pnl_pct:.1%}) with plateau detected"
        
        # Check if price is declining after significant gain
        if pnl_pct > 0.20:  # If we're up more than 20%
            if len(self.price_history[trade.symbol]) >= 5:
                recent_prices = list(self.price_history[trade.symbol])[-5:]
                if all(recent_prices[i] <= recent_prices[i-1] for i in range(1, len(recent_prices))):
                    return True, f"Price declining after {pnl_pct:.1%} gain"
        
        return False, f"Hold position (P&L: {pnl_pct:.1%})"
    
    def execute_listing_sell(self, trade: ListingTrade, current_price: float, exit_reason: str) -> bool:
        """Execute sell order for listing position"""
        try:
            # Calculate quantity to sell
            quantity = trade.position_size_usd / trade.entry_price
            
            # Execute sell order (integrate with existing trading engine)
            sell_result = self.place_sell_order(trade.symbol, quantity, current_price)
            
            if sell_result:
                # Calculate final P&L
                pnl_usd = (current_price - trade.entry_price) * quantity
                pnl_pct = (current_price - trade.entry_price) / trade.entry_price
                
                # Update trade record
                self.update_listing_trade_exit(trade, current_price, pnl_usd)
                
                logger.info(f"✅ LISTING TRADE EXITED: {trade.symbol}")
                logger.info(f"   Exit Price: ${current_price:.4f}")
                logger.info(f"   P&L: ${pnl_usd:.2f} ({pnl_pct:.1%})")
                logger.info(f"   Reason: {exit_reason}")
                
                return True
            
        except Exception as e:
            logger.error(f"❌ Error executing listing sell for {trade.symbol}: {e}")
        
        return False
    
    def place_sell_order(self, symbol: str, quantity: float, price: float) -> bool:
        """Place sell order through existing trading infrastructure"""
        try:
            # Log the simulated sell order
            total_value = quantity * price
            logger.info(f"📉 Simulated SELL order: {quantity:.6f} {symbol} @ ${price:.4f}")
            logger.info(f"   Total Value: ${total_value:.2f}")
            
            # In production, this would call the actual trading engine
            # e.g., coinbase_api.place_market_order(f"{symbol}-USD", "sell", str(quantity))
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error placing sell order: {e}")
            return False
    
    def update_listing_trade_exit(self, trade: ListingTrade, exit_price: float, pnl_usd: float):
        """Update trade record with exit information"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor()
            
            update_query = """
            UPDATE listing_trades 
            SET exit_price = %s, exit_time = %s, pnl_usd = %s, status = 'EXITED'
            WHERE symbol = %s AND status = 'ACTIVE'
            """
            
            values = (exit_price, datetime.now(), pnl_usd, trade.symbol)
            cursor.execute(update_query, values)
            connection.commit()
            
        except Error as e:
            logger.error(f"❌ Error updating listing trade exit: {e}")
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def get_active_listing_trades(self) -> List[Dict]:
        """Get all active listing trades"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor(dictionary=True)
            
            cursor.execute("""
                SELECT * FROM listing_trades 
                WHERE status = 'ACTIVE'
                ORDER BY entry_time DESC
            """)
            
            return cursor.fetchall()
            
        except Error as e:
            logger.error(f"❌ Error getting active listing trades: {e}")
            return []
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def monitor_active_positions(self):
        """Monitor active listing positions for exit opportunities"""
        active_trades = self.get_active_listing_trades()
        
        for trade_data in active_trades:
            try:
                symbol = trade_data["symbol"]
                
                # Get current price
                current_price = self.get_current_price(symbol)
                if not current_price:
                    continue
                
                # Update price history
                self.update_price_history(symbol, current_price)
                
                # Create trade object
                trade = ListingTrade(
                    symbol=trade_data["symbol"],
                    entry_price=float(trade_data["entry_price"]),
                    entry_time=trade_data["entry_time"],
                    position_size_usd=float(trade_data["position_size_usd"]),
                    listing_source=trade_data["listing_source"],
                    confidence_score=trade_data["confidence_score"]
                )
                
                # Check if should exit
                should_exit, reason = self.should_exit_position(trade, current_price)
                
                if should_exit:
                    self.execute_listing_sell(trade, current_price, reason)
                else:
                    # Log current status
                    pnl_pct = (current_price - trade.entry_price) / trade.entry_price
                    logger.info(f"📊 {symbol}: ${current_price:.4f} ({pnl_pct:+.1%}) - {reason}")
                
            except Exception as e:
                logger.error(f"❌ Error monitoring position {symbol}: {e}")
    
    def process_listing_signals(self):
        """Process new listing signals from the monitoring system"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            cursor = connection.cursor(dictionary=True)
            
            # Get unprocessed listing signals
            cursor.execute("""
                SELECT * FROM trading_signals 
                WHERE signal_type = 'LISTING_OPPORTUNITY' 
                AND processed = FALSE 
                ORDER BY created_at ASC
            """)
            
            signals = cursor.fetchall()
            
            for signal in signals:
                try:
                    # Parse signal metadata
                    metadata = json.loads(signal["metadata"]) if signal["metadata"] else {}
                    
                    # Execute listing trade
                    trade = self.execute_listing_buy(metadata)
                    
                    # Mark signal as processed
                    cursor.execute("""
                        UPDATE trading_signals 
                        SET processed = TRUE 
                        WHERE id = %s
                    """, (signal["id"],))
                    connection.commit()
                    
                    if trade:
                        logger.info(f"✅ Processed listing signal for {signal['symbol']}")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing signal {signal['id']}: {e}")
            
        except Error as e:
            logger.error(f"❌ Error processing listing signals: {e}")
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

def main():
    """Main trading loop for listing opportunities"""
    trader = CoinbaseListingTrader()
    
    logger.info("🚀 Coinbase Listing Trading Engine Starting...")
    logger.info("📊 Strategy Parameters:")
    logger.info(f"   • Max Position Size: ${trader.max_position_size_usd}")
    logger.info(f"   • Max Portfolio Allocation: {trader.max_portfolio_allocation*100}%")
    logger.info(f"   • Stop Loss: {trader.stop_loss_percentage*100}%")
    logger.info(f"   • Profit Targets: {trader.profit_target_1*100}%, {trader.profit_target_2*100}%")
    logger.info("🎯 Target: Capture Coinbase Effect (91% average gain)")
    
    while True:
        try:
            # Process new listing signals
            trader.process_listing_signals()
            
            # Monitor active positions
            trader.monitor_active_positions()
            
            # Wait 30 seconds between cycles
            time.sleep(30)
            
        except KeyboardInterrupt:
            logger.info("🛑 Trading engine stopped by user")
            break
        except Exception as e:
            logger.error(f"❌ Error in trading loop: {e}")
            time.sleep(60)  # Wait 1 minute before retrying

if __name__ == "__main__":
    main()
