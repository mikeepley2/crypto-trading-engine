#!/usr/bin/env python3
"""
Simple Trading Integration for Windows compatibility
No emoji characters to avoid Unicode issues
"""

import mysql.connector
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_trade_recommendations():
    """Create trade recommendations without emoji characters"""
    
    # Database connections
    price_db_config = {
        'host': 'localhost',
        'user': 'news_collector',
        'password': '99Rules!',
        'database': 'crypto_prices'
    }
    
    trading_db_config = {
        'host': 'localhost',
        'user': 'news_collector',
        'password': '99Rules!',
        'database': 'crypto_transactions'
    }
    
    try:
        logger.info("STARTING: Trade recommendation creation")
        
        # Connect to price database for signals
        price_conn = mysql.connector.connect(**price_db_config)
        price_cursor = price_conn.cursor()
        
        # Connect to trading database for recommendations
        trading_conn = mysql.connector.connect(**trading_db_config)
        trading_cursor = trading_conn.cursor()
        
        # Get new trading signals
        price_cursor.execute('''
        SELECT symbol, signal_type, confidence, timestamp
        FROM trading_signals
        WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 MINUTE)
        ORDER BY created_at DESC
        LIMIT 20
        ''')
        
        signals = price_cursor.fetchall()
        
        if not signals:
            logger.info("INFO: No new signals for recommendations")
            price_conn.close()
            trading_conn.close()
            return 0
        
        logger.info(f"TRADING: Found {len(signals)} signals for recommendations")
        
        # Get current prices for calculations
        symbols_list = list(set([signal[0] for signal in signals]))
        placeholders = ','.join(['%s'] * len(symbols_list))
        price_cursor.execute(f'''
        SELECT symbol, current_price
        FROM ml_features_materialized
        WHERE symbol IN ({placeholders})
        ORDER BY timestamp_iso DESC
        ''', symbols_list)
        
        price_data = {row[0]: row[1] for row in price_cursor.fetchall()}
        
        recommendations_created = 0
        for signal in signals:
            symbol, signal_type, confidence, timestamp = signal
            
            if symbol not in price_data:
                logger.warning(f"WARNING: No price data for {symbol}")
                continue
                
            current_price = float(price_data[symbol])
            
            # Calculate stop-loss and take-profit
            if signal_type == 'BUY':
                stop_loss_price = current_price * 0.95  # 5% below
                take_profit_price = current_price * 1.10  # 10% above
            else:  # SELL
                stop_loss_price = current_price * 1.05  # 5% above for short
                take_profit_price = current_price * 0.90  # 10% below for short
            
            # Create recommendation
            insert_query = '''
            INSERT INTO trade_recommendations (
                symbol, signal_type, entry_price, stop_loss_price, take_profit_price,
                confidence, quantity, created_at, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            '''
            
            trading_cursor.execute(insert_query, (
                symbol, signal_type, current_price, stop_loss_price, 
                take_profit_price, confidence, 100.0, datetime.now(), 'PENDING'
            ))
            
            recommendations_created += 1
            logger.info(f"RECOMMENDATION: {symbol} {signal_type} price={current_price:.4f}")
        
        trading_conn.commit()
        price_conn.close()
        trading_conn.close()
        
        logger.info(f"SUCCESS: Created {recommendations_created} recommendations")
        return recommendations_created
        
    except Exception as e:
        logger.error(f"ERROR: Trade recommendation creation failed: {e}")
        return 0

if __name__ == "__main__":
    create_trade_recommendations()
