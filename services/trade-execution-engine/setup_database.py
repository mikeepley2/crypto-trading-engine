#!/usr/bin/env python3
"""
Setup required database tables for live trading
"""

import mysql.connector
import os
from dotenv import load_dotenv

# Load environment
load_dotenv('e:/git/aitest/.env.live')

def create_database_connection():
    """Create database connection"""
    return mysql.connector.connect(
        host=os.getenv('DB_HOST', 'host.docker.internal'),
        user=os.getenv('DB_USER', 'news_collector'),
        password=os.getenv('DB_PASSWORD', '99Rules!'),
        database='crypto_prices'
    )

def setup_crypto_prices_schema():
    """Setup crypto_prices database schema"""
    conn = create_database_connection()
    cursor = conn.cursor()
    
    try:
        # Check if price column exists in crypto_prices table
        cursor.execute("SHOW COLUMNS FROM crypto_prices LIKE 'price'")
        if not cursor.fetchone():
            print("Adding price column to crypto_prices table...")
            cursor.execute("ALTER TABLE crypto_prices ADD COLUMN price DECIMAL(15,8)")
            print("[+] Price column added")
        else:
            print("[+] Price column already exists")
        
        # Check if unified_sentiment_aggregated table exists
        cursor.execute("SHOW TABLES LIKE 'unified_sentiment_aggregated'")
        if not cursor.fetchone():
            print("Creating unified_sentiment_aggregated table...")
            cursor.execute("""
                CREATE TABLE unified_sentiment_aggregated (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL,
                    timestamp DATETIME NOT NULL,
                    overall_sentiment DECIMAL(4,3),
                    social_sentiment DECIMAL(4,3),
                    news_sentiment DECIMAL(4,3),
                    sentiment_score DECIMAL(4,3),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_symbol_timestamp (symbol, timestamp)
                )
            """)
            print("[+] unified_sentiment_aggregated table created")
        else:
            print("[+] unified_sentiment_aggregated table already exists")
        
        conn.commit()
        print("[+] Crypto prices schema setup complete")
        
    except Exception as e:
        print(f"[!] Error setting up crypto_prices schema: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def setup_crypto_transactions_schema():
    """Setup crypto_transactions database schema"""
    conn = mysql.connector.connect(
        host=os.getenv('DB_HOST', 'host.docker.internal'),
        user=os.getenv('DB_USER', 'news_collector'),
        password=os.getenv('DB_PASSWORD', '99Rules!'),
        database='crypto_transactions'
    )
    cursor = conn.cursor()
    
    try:
        # Check if live_trades table exists
        cursor.execute("SHOW TABLES LIKE 'live_trades'")
        if not cursor.fetchone():
            print("Creating live_trades table...")
            cursor.execute("""
                CREATE TABLE live_trades (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    trade_id VARCHAR(100) UNIQUE NOT NULL,
                    symbol VARCHAR(10) NOT NULL,
                    side ENUM('BUY', 'SELL') NOT NULL,
                    size_usd DECIMAL(15,2) NOT NULL,
                    size_crypto DECIMAL(20,10),
                    price DECIMAL(15,8),
                    status ENUM('PENDING', 'FILLED', 'CANCELLED', 'FAILED') DEFAULT 'PENDING',
                    coinbase_order_id VARCHAR(100),
                    executed_at DATETIME,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_symbol (symbol),
                    INDEX idx_status (status),
                    INDEX idx_created_at (created_at)
                )
            """)
            print("[+] live_trades table created")
        else:
            print("[+] live_trades table already exists")
        
        # Check if live_portfolio table exists
        cursor.execute("SHOW TABLES LIKE 'live_portfolio'")
        if not cursor.fetchone():
            print("Creating live_portfolio table...")
            cursor.execute("""
                CREATE TABLE live_portfolio (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL UNIQUE,
                    quantity DECIMAL(20,10) NOT NULL DEFAULT 0,
                    avg_buy_price DECIMAL(15,8),
                    total_invested DECIMAL(15,2) DEFAULT 0,
                    current_value DECIMAL(15,2) DEFAULT 0,
                    unrealized_pnl DECIMAL(15,2) DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_symbol (symbol)
                )
            """)
            print("[+] live_portfolio table created")
        else:
            print("[+] live_portfolio table already exists")
        
        conn.commit()
        print("[+] Crypto transactions schema setup complete")
        
    except Exception as e:
        print(f"[!] Error setting up crypto_transactions schema: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def main():
    """Setup all required database schemas"""
    print("Setting up database schemas for live trading...")
    
    try:
        setup_crypto_prices_schema()
        setup_crypto_transactions_schema()
        print("\n[+] All database schemas setup successfully!")
        return True
    except Exception as e:
        print(f"\n[!] Database setup failed: {e}")
        return False

if __name__ == "__main__":
    main()
