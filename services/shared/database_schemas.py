#!/usr/bin/env python3

import mysql.connector
import datetime

# Database configuration for mock trading schema
MYSQL_CONFIG = {
    'host': 'host.docker.internal',
    'user': 'news_collector',
    'password': '99Rules!',
    'port': 3306
}

DB_NAME = 'crypto_transactions'

# Enhanced schema for mock trading
TABLES = {}

# Mock Holdings Table - separate from real holdings
TABLES['mock_holdings'] = """
    CREATE TABLE IF NOT EXISTS mock_holdings (
        id INT AUTO_INCREMENT PRIMARY KEY,
        symbol VARCHAR(20) NOT NULL,
        quantity DECIMAL(20,8) NOT NULL DEFAULT 0.0,
        avg_entry_price DECIMAL(20,8) NOT NULL DEFAULT 0.0,
        total_invested DECIMAL(20,8) NOT NULL DEFAULT 0.0,
        realized_pnl DECIMAL(20,8) DEFAULT 0.0,
        unrealized_pnl DECIMAL(20,8) DEFAULT 0.0,
        total_pnl DECIMAL(20,8) DEFAULT 0.0,
        last_price DECIMAL(20,8) DEFAULT 0.0,
        position_value DECIMAL(20,8) DEFAULT 0.0,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_symbol (symbol),
        UNIQUE KEY unique_mock_symbol (symbol)
    ) ENGINE=InnoDB;
"""

# Mock Trades Table - execution records
TABLES['mock_trades'] = """
    CREATE TABLE IF NOT EXISTS mock_trades (
        id INT AUTO_INCREMENT PRIMARY KEY,
        recommendation_id INT DEFAULT NULL,
        symbol VARCHAR(20) NOT NULL,
        side ENUM('BUY', 'SELL') NOT NULL,
        quantity DECIMAL(20,8) NOT NULL,
        price DECIMAL(20,8) NOT NULL,
        amount DECIMAL(20,8) NOT NULL,
        fee DECIMAL(20,8) DEFAULT 0.0,
        net_amount DECIMAL(20,8) NOT NULL,
        order_type VARCHAR(20) DEFAULT 'MARKET',
        status ENUM('PENDING', 'FILLED', 'CANCELLED', 'FAILED') DEFAULT 'FILLED',
        execution_time DATETIME NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        notes TEXT,
        confidence DECIMAL(5,4) DEFAULT NULL,
        regime VARCHAR(20) DEFAULT NULL,
        INDEX idx_symbol (symbol),
        INDEX idx_execution_time (execution_time),
        INDEX idx_status (status),
        INDEX idx_recommendation_id (recommendation_id)
    ) ENGINE=InnoDB;
"""

# Mock Portfolio Summary Table
TABLES['mock_portfolio'] = """
    CREATE TABLE IF NOT EXISTS mock_portfolio (
        id INT AUTO_INCREMENT PRIMARY KEY,
        total_value DECIMAL(20,8) NOT NULL DEFAULT 100000.0,
        cash_balance DECIMAL(20,8) NOT NULL DEFAULT 100000.0,
        invested_amount DECIMAL(20,8) NOT NULL DEFAULT 0.0,
        total_pnl DECIMAL(20,8) NOT NULL DEFAULT 0.0,
        total_pnl_percent DECIMAL(8,4) NOT NULL DEFAULT 0.0,
        total_trades INT DEFAULT 0,
        winning_trades INT DEFAULT 0,
        losing_trades INT DEFAULT 0,
        win_rate DECIMAL(5,2) DEFAULT 0.0,
        largest_win DECIMAL(20,8) DEFAULT 0.0,
        largest_loss DECIMAL(20,8) DEFAULT 0.0,
        sharpe_ratio DECIMAL(8,4) DEFAULT NULL,
        max_drawdown DECIMAL(8,4) DEFAULT 0.0,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_updated_at (updated_at)
    ) ENGINE=InnoDB;
"""

# Trading Performance Tracking
TABLES['mock_performance_history'] = """
    CREATE TABLE IF NOT EXISTS mock_performance_history (
        id INT AUTO_INCREMENT PRIMARY KEY,
        date DATE NOT NULL,
        portfolio_value DECIMAL(20,8) NOT NULL,
        cash_balance DECIMAL(20,8) NOT NULL,
        invested_amount DECIMAL(20,8) NOT NULL,
        daily_pnl DECIMAL(20,8) NOT NULL DEFAULT 0.0,
        daily_return DECIMAL(8,4) NOT NULL DEFAULT 0.0,
        cumulative_return DECIMAL(8,4) NOT NULL DEFAULT 0.0,
        num_positions INT DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY unique_date (date),
        INDEX idx_date (date)
    ) ENGINE=InnoDB;
"""

# Trading Orders (for pending orders and order management)
TABLES['mock_orders'] = """
    CREATE TABLE IF NOT EXISTS mock_orders (
        id INT AUTO_INCREMENT PRIMARY KEY,
        recommendation_id INT DEFAULT NULL,
        symbol VARCHAR(20) NOT NULL,
        side ENUM('BUY', 'SELL') NOT NULL,
        order_type ENUM('MARKET', 'LIMIT', 'STOP_LOSS', 'TAKE_PROFIT') DEFAULT 'MARKET',
        quantity DECIMAL(20,8) NOT NULL,
        price DECIMAL(20,8) DEFAULT NULL,
        stop_price DECIMAL(20,8) DEFAULT NULL,
        target_amount DECIMAL(20,8) DEFAULT NULL,
        status ENUM('PENDING', 'FILLED', 'PARTIALLY_FILLED', 'CANCELLED', 'FAILED') DEFAULT 'PENDING',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        filled_at DATETIME DEFAULT NULL,
        filled_quantity DECIMAL(20,8) DEFAULT 0.0,
        filled_price DECIMAL(20,8) DEFAULT NULL,
        notes TEXT,
        confidence DECIMAL(5,4) DEFAULT NULL,
        regime VARCHAR(20) DEFAULT NULL,
        INDEX idx_symbol (symbol),
        INDEX idx_status (status),
        INDEX idx_created_at (created_at),
        INDEX idx_recommendation_id (recommendation_id)
    ) ENGINE=InnoDB;
"""

# Enhanced trade_recommendations table to track execution status
TABLES['update_trade_recommendations'] = """
    ALTER TABLE trade_recommendations 
    ADD COLUMN executed_at DATETIME DEFAULT NULL,
    ADD COLUMN execution_price DECIMAL(20,8) DEFAULT NULL,
    ADD COLUMN execution_status ENUM('PENDING', 'EXECUTED', 'REJECTED', 'EXPIRED') DEFAULT 'PENDING',
    ADD COLUMN execution_notes TEXT,
    ADD INDEX idx_execution_status (execution_status),
    ADD INDEX idx_executed_at (executed_at);
"""


def create_mock_trading_schema():
    """Create the complete mock trading database schema"""
    try:
        # Connect to MySQL
        cnx = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = cnx.cursor()
        
        # Create database if it doesn't exist
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
        cursor.execute(f"USE {DB_NAME}")
        
        print(f"✅ Connected to database: {DB_NAME}")
        
        # Create all tables
        for table_name, ddl in TABLES.items():
            try:
                print(f"📊 Creating/updating table: {table_name}...")
                
                if table_name == 'update_trade_recommendations':
                    # Handle ALTER TABLE separately with error handling
                    try:
                        cursor.execute(ddl)
                    except mysql.connector.Error as alter_err:
                        if "Duplicate column name" in str(alter_err):
                            print(f"⚠️ Columns already exist in trade_recommendations table")
                        else:
                            print(f"⚠️ ALTER TABLE warning: {alter_err}")
                else:
                    cursor.execute(ddl)
                    
                print(f"✅ Table {table_name} created/updated successfully")
                
            except mysql.connector.Error as err:
                if "Duplicate column name" in str(err):
                    print(f"⚠️ Column already exists in {table_name}, skipping...")
                else:
                    print(f"❌ Error creating table {table_name}: {err}")
                    raise
        
        # Initialize mock portfolio with starting balance
        initialize_mock_portfolio(cursor)
        
        cnx.commit()
        cursor.close()
        cnx.close()
        
        print("✅ Mock trading schema created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating mock trading schema: {e}")
        return False


def initialize_mock_portfolio(cursor):
    """Initialize mock portfolio with starting values"""
    try:
        # Check if portfolio already exists
        cursor.execute("SELECT COUNT(*) FROM mock_portfolio")
        count = cursor.fetchone()[0]
        
        if count == 0:
            # Create initial portfolio record
            cursor.execute("""
                INSERT INTO mock_portfolio 
                (total_value, cash_balance, invested_amount) 
                VALUES (100000.0, 100000.0, 0.0)
            """)
            print("💰 Initialized mock portfolio with $100,000 starting balance")
        else:
            print("📊 Mock portfolio already exists")
            
        # Initialize today's performance record if not exists
        today = datetime.date.today()
        cursor.execute("""
            INSERT IGNORE INTO mock_performance_history 
            (date, portfolio_value, cash_balance, invested_amount) 
            VALUES (%s, 100000.0, 100000.0, 0.0)
        """, (today,))
        
    except Exception as e:
        print(f"❌ Error initializing mock portfolio: {e}")


def create_mock_trading_indexes():
    """Create additional indexes for performance optimization"""
    try:
        cnx = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = cnx.cursor()
        cursor.execute(f"USE {DB_NAME}")
        
        # Additional indexes for query optimization
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_mock_trades_symbol_time ON mock_trades(symbol, execution_time)",
            "CREATE INDEX IF NOT EXISTS idx_mock_trades_side_status ON mock_trades(side, status)",
            "CREATE INDEX IF NOT EXISTS idx_mock_orders_symbol_status ON mock_orders(symbol, status)",
            "CREATE INDEX IF NOT EXISTS idx_mock_performance_date_desc ON mock_performance_history(date DESC)",
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
                print(f"✅ Index created: {index_sql.split('ON')[1].split('(')[0].strip()}")
            except mysql.connector.Error as err:
                if "Duplicate key name" not in str(err):
                    print(f"⚠️ Index creation warning: {err}")
        
        cnx.commit()
        cursor.close()
        cnx.close()
        
        print("✅ Mock trading indexes created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating indexes: {e}")


def verify_schema():
    """Verify that all tables were created correctly"""
    try:
        cnx = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = cnx.cursor()
        cursor.execute(f"USE {DB_NAME}")
        
        # Check all tables exist
        cursor.execute("SHOW TABLES")
        existing_tables = [table[0] for table in cursor.fetchall()]
        
        expected_tables = ['mock_holdings', 'mock_trades', 'mock_portfolio', 
                          'mock_performance_history', 'mock_orders', 'trade_recommendations']
        
        print("\n📊 Schema Verification:")
        print("=" * 50)
        
        for table in expected_tables:
            if table in existing_tables:
                # Get table info
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"✅ {table}: EXISTS ({count} records)")
            else:
                print(f"❌ {table}: MISSING")
        
        # Check portfolio initialization
        cursor.execute("SELECT total_value, cash_balance FROM mock_portfolio LIMIT 1")
        portfolio = cursor.fetchone()
        if portfolio:
            print(f"💰 Portfolio: ${portfolio[0]:,.2f} total, ${portfolio[1]:,.2f} cash")
        
        cursor.close()
        cnx.close()
        
        print("=" * 50)
        print("✅ Schema verification complete!")
        
    except Exception as e:
        print(f"❌ Schema verification error: {e}")


if __name__ == "__main__":
    print("🚀 Creating Enhanced Mock Trading Schema...")
    print("=" * 60)
    
    # Create the schema
    if create_mock_trading_schema():
        # Create additional indexes
        create_mock_trading_indexes()
        
        # Verify everything was created
        verify_schema()
        
        print("\n🎉 Mock trading database schema setup complete!")
        print("📊 Ready for mock trading execution service")
    else:
        print("❌ Failed to create mock trading schema")
