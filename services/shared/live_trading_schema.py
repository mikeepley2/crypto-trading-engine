#!/usr/bin/env python3
"""
Live Trading Database Schema - Separate from Mock Trading
Creates dedicated tables for real money trading operations
"""

import mysql.connector
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
MYSQL_CONFIG = {
    'host': 'host.docker.internal',
    'user': 'news_collector',
    'password': '99Rules!',
    'database': 'crypto_analysis',
    'autocommit': True
}

def create_live_trading_schema():
    """Create all live trading tables with proper separation from mock trading"""
    
    try:
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = connection.cursor()
        
        logger.info("🏗️ Creating live trading database schema...")
        
        # 1. Live Holdings Table - Current positions in live trading
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS live_holdings (
                id INT AUTO_INCREMENT PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                quantity DECIMAL(20, 8) NOT NULL DEFAULT 0.00000000,
                avg_entry_price DECIMAL(15, 8) NOT NULL DEFAULT 0.00000000,
                total_invested DECIMAL(15, 2) NOT NULL DEFAULT 0.00,
                realized_pnl DECIMAL(15, 2) NOT NULL DEFAULT 0.00,
                unrealized_pnl DECIMAL(15, 2) NOT NULL DEFAULT 0.00,
                total_pnl DECIMAL(15, 2) NOT NULL DEFAULT 0.00,
                last_price DECIMAL(15, 8) NOT NULL DEFAULT 0.00000000,
                position_value DECIMAL(15, 2) NOT NULL DEFAULT 0.00,
                exchange VARCHAR(50) NOT NULL DEFAULT 'coinbase_pro',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_symbol (symbol),
                INDEX idx_exchange (exchange),
                UNIQUE KEY unique_symbol_exchange (symbol, exchange)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        logger.info("✅ Created live_holdings table")
        
        # 2. Live Trades Table - Complete trade history for live trading
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS live_trades (
                id INT AUTO_INCREMENT PRIMARY KEY,
                exchange_order_id VARCHAR(100) UNIQUE,
                recommendation_id INT,
                symbol VARCHAR(20) NOT NULL,
                side ENUM('BUY', 'SELL') NOT NULL,
                quantity DECIMAL(20, 8) NOT NULL,
                price DECIMAL(15, 8) NOT NULL,
                amount DECIMAL(15, 2) NOT NULL,
                fee DECIMAL(15, 8) NOT NULL DEFAULT 0.00000000,
                net_amount DECIMAL(15, 2) NOT NULL,
                exchange VARCHAR(50) NOT NULL DEFAULT 'coinbase_pro',
                order_type ENUM('market', 'limit', 'stop', 'stop_limit') DEFAULT 'market',
                status ENUM('pending', 'filled', 'partially_filled', 'cancelled', 'failed') DEFAULT 'pending',
                execution_time TIMESTAMP NOT NULL,
                confidence DECIMAL(5, 4),
                regime VARCHAR(50),
                notes TEXT,
                exchange_data JSON,  -- Store raw exchange response
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_symbol (symbol),
                INDEX idx_execution_time (execution_time),
                INDEX idx_exchange (exchange),
                INDEX idx_status (status),
                INDEX idx_recommendation_id (recommendation_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        logger.info("✅ Created live_trades table")
        
        # 3. Live Portfolio Table - Overall portfolio tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS live_portfolio (
                id INT AUTO_INCREMENT PRIMARY KEY,
                exchange VARCHAR(50) NOT NULL DEFAULT 'coinbase_pro',
                total_value DECIMAL(15, 2) NOT NULL DEFAULT 100000.00,
                cash_balance DECIMAL(15, 2) NOT NULL DEFAULT 100000.00,
                invested_amount DECIMAL(15, 2) NOT NULL DEFAULT 0.00,
                total_pnl DECIMAL(15, 2) NOT NULL DEFAULT 0.00,
                total_pnl_percent DECIMAL(8, 4) NOT NULL DEFAULT 0.0000,
                daily_pnl DECIMAL(15, 2) NOT NULL DEFAULT 0.00,
                daily_pnl_percent DECIMAL(8, 4) NOT NULL DEFAULT 0.0000,
                num_positions INT NOT NULL DEFAULT 0,
                max_drawdown DECIMAL(8, 4) NOT NULL DEFAULT 0.0000,
                max_drawdown_date DATE,
                all_time_high DECIMAL(15, 2) NOT NULL DEFAULT 100000.00,
                all_time_high_date DATE,
                sharpe_ratio DECIMAL(8, 4),
                win_rate DECIMAL(5, 2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_exchange (exchange)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        logger.info("✅ Created live_portfolio table")
        
        # 4. Live Orders Table - Track pending and executed orders
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS live_orders (
                id INT AUTO_INCREMENT PRIMARY KEY,
                exchange_order_id VARCHAR(100) UNIQUE,
                client_order_id VARCHAR(100),
                symbol VARCHAR(20) NOT NULL,
                side ENUM('BUY', 'SELL') NOT NULL,
                order_type ENUM('market', 'limit', 'stop', 'stop_limit') NOT NULL,
                quantity DECIMAL(20, 8) NOT NULL,
                price DECIMAL(15, 8),  -- NULL for market orders
                stop_price DECIMAL(15, 8),  -- For stop orders
                time_in_force ENUM('GTC', 'GTT', 'IOC', 'FOK') DEFAULT 'GTC',
                status ENUM('pending', 'open', 'filled', 'partially_filled', 'cancelled', 'rejected', 'expired') DEFAULT 'pending',
                filled_quantity DECIMAL(20, 8) DEFAULT 0.00000000,
                remaining_quantity DECIMAL(20, 8),
                avg_fill_price DECIMAL(15, 8),
                total_fees DECIMAL(15, 8) DEFAULT 0.00000000,
                exchange VARCHAR(50) NOT NULL DEFAULT 'coinbase_pro',
                exchange_data JSON,  -- Store raw exchange response
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                filled_at TIMESTAMP NULL,
                INDEX idx_symbol (symbol),
                INDEX idx_status (status),
                INDEX idx_exchange (exchange),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        logger.info("✅ Created live_orders table")
        
        # 5. Live Performance History Table - Daily performance tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS live_performance_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                date DATE NOT NULL,
                exchange VARCHAR(50) NOT NULL DEFAULT 'coinbase_pro',
                portfolio_value DECIMAL(15, 2) NOT NULL,
                cash_balance DECIMAL(15, 2) NOT NULL,
                invested_amount DECIMAL(15, 2) NOT NULL,
                daily_pnl DECIMAL(15, 2) NOT NULL DEFAULT 0.00,
                daily_pnl_percent DECIMAL(8, 4) NOT NULL DEFAULT 0.0000,
                cumulative_pnl DECIMAL(15, 2) NOT NULL DEFAULT 0.00,
                cumulative_pnl_percent DECIMAL(8, 4) NOT NULL DEFAULT 0.0000,
                num_trades INT NOT NULL DEFAULT 0,
                num_positions INT NOT NULL DEFAULT 0,
                max_drawdown DECIMAL(8, 4) NOT NULL DEFAULT 0.0000,
                volatility DECIMAL(8, 6),
                sharpe_ratio DECIMAL(8, 4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY unique_date_exchange (date, exchange),
                INDEX idx_date (date),
                INDEX idx_exchange (exchange)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        logger.info("✅ Created live_performance_history table")
        
        # 6. Exchange Connections Table - Track exchange API connections
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exchange_connections (
                id INT AUTO_INCREMENT PRIMARY KEY,
                exchange VARCHAR(50) NOT NULL,
                name VARCHAR(100) NOT NULL,
                api_key_hash VARCHAR(64),  -- Hashed API key for identification
                is_active BOOLEAN DEFAULT true,
                is_sandbox BOOLEAN DEFAULT false,
                last_connection_test TIMESTAMP,
                connection_status ENUM('connected', 'disconnected', 'error', 'unauthorized') DEFAULT 'disconnected',
                last_error TEXT,
                supported_features JSON,  -- What features this exchange supports
                rate_limits JSON,  -- Rate limit information
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY unique_exchange (exchange),
                INDEX idx_is_active (is_active)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        logger.info("✅ Created exchange_connections table")
        
        # Insert initial portfolio entry for each exchange
        cursor.execute("""
            INSERT IGNORE INTO live_portfolio (exchange, total_value, cash_balance) 
            VALUES 
                ('coinbase_pro', 0.00, 0.00),
                ('binance', 0.00, 0.00),
                ('kraken', 0.00, 0.00)
        """)
        logger.info("✅ Initialized live portfolio entries")
        
        # Insert exchange connection entries
        cursor.execute("""
            INSERT IGNORE INTO exchange_connections (exchange, name, is_sandbox, supported_features) 
            VALUES 
                ('coinbase_pro', 'Coinbase Pro', true, JSON_OBJECT('spot', true, 'margin', false, 'futures', false)),
                ('binance', 'Binance', true, JSON_OBJECT('spot', true, 'margin', true, 'futures', true)),
                ('kraken', 'Kraken', true, JSON_OBJECT('spot', true, 'margin', true, 'futures', false))
        """)
        logger.info("✅ Initialized exchange connection entries")
        
        cursor.close()
        connection.close()
        
        logger.info("🎉 Live trading database schema created successfully!")
        logger.info("📊 Tables created:")
        logger.info("   - live_holdings (current positions)")
        logger.info("   - live_trades (trade history)")
        logger.info("   - live_portfolio (portfolio summary)")
        logger.info("   - live_orders (order management)")
        logger.info("   - live_performance_history (daily tracking)")
        logger.info("   - exchange_connections (exchange management)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating live trading schema: {e}")
        return False

def verify_schema():
    """Verify that all live trading tables exist and have the correct structure"""
    try:
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = connection.cursor()
        
        tables = [
            'live_holdings',
            'live_trades', 
            'live_portfolio',
            'live_orders',
            'live_performance_history',
            'exchange_connections'
        ]
        
        logger.info("🔍 Verifying live trading schema...")
        
        for table in tables:
            cursor.execute(f"SHOW TABLES LIKE '{table}'")
            result = cursor.fetchone()
            if result:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                logger.info(f"✅ {table}: EXISTS ({count} records)")
            else:
                logger.error(f"❌ {table}: MISSING")
                return False
        
        cursor.close()
        connection.close()
        
        logger.info("✅ Live trading schema verification complete!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error verifying schema: {e}")
        return False

def drop_live_trading_schema():
    """Drop all live trading tables (USE WITH CAUTION)"""
    try:
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = connection.cursor()
        
        logger.warning("⚠️ DROPPING ALL LIVE TRADING TABLES...")
        
        tables = [
            'live_performance_history',
            'live_orders', 
            'live_trades',
            'live_holdings',
            'live_portfolio',
            'exchange_connections'
        ]
        
        for table in tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
            logger.info(f"🗑️ Dropped {table}")
        
        cursor.close()
        connection.close()
        
        logger.info("✅ All live trading tables dropped!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error dropping tables: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--drop":
        print("⚠️ This will DROP ALL live trading tables!")
        confirm = input("Type 'DROP_LIVE_TABLES' to confirm: ")
        if confirm == "DROP_LIVE_TABLES":
            drop_live_trading_schema()
        else:
            print("Operation cancelled.")
    elif len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_schema()
    else:
        create_live_trading_schema()
        verify_schema()
