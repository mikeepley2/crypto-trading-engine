#!/usr/bin/env python3
"""
Database migration script for portfolio optimization tables
"""
import os
import mysql.connector
import sys

def run_migration():
    try:
        # Database connection
        conn = mysql.connector.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            user=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', ''),
            database=os.getenv('DB_NAME_PRICES', 'crypto_trading')
        )
        
        cursor = conn.cursor()
        
        # Migration SQL
        migration_sql = """
        -- Create portfolio_optimizations table
        CREATE TABLE IF NOT EXISTS portfolio_optimizations (
            id INT AUTO_INCREMENT PRIMARY KEY,
            method VARCHAR(50) NOT NULL,
            weights JSON NOT NULL,
            expected_return DECIMAL(10, 6) NOT NULL,
            volatility DECIMAL(10, 6) NOT NULL,
            sharpe_ratio DECIMAL(10, 6) NOT NULL,
            optimization_time DECIMAL(10, 6) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_method (method),
            INDEX idx_created_at (created_at),
            INDEX idx_sharpe_ratio (sharpe_ratio)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        
        -- Create backtesting_results table
        CREATE TABLE IF NOT EXISTS backtesting_results (
            id INT AUTO_INCREMENT PRIMARY KEY,
            strategy_name VARCHAR(100) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            initial_capital DECIMAL(15, 2) NOT NULL,
            final_capital DECIMAL(15, 2) NOT NULL,
            total_return DECIMAL(10, 6) NOT NULL,
            annualized_return DECIMAL(10, 6) NOT NULL,
            volatility DECIMAL(10, 6) NOT NULL,
            sharpe_ratio DECIMAL(10, 6) NOT NULL,
            max_drawdown DECIMAL(10, 6) NOT NULL,
            win_rate DECIMAL(10, 6) NOT NULL,
            total_trades INT NOT NULL,
            profitable_trades INT NOT NULL,
            parameters JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_strategy (strategy_name),
            INDEX idx_symbol (symbol),
            INDEX idx_start_date (start_date),
            INDEX idx_sharpe_ratio (sharpe_ratio)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        
        -- Create backtesting_trades table
        CREATE TABLE IF NOT EXISTS backtesting_trades (
            id INT AUTO_INCREMENT PRIMARY KEY,
            backtesting_result_id INT NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            trade_type ENUM('BUY', 'SELL') NOT NULL,
            entry_price DECIMAL(20, 8) NOT NULL,
            exit_price DECIMAL(20, 8),
            quantity DECIMAL(20, 8) NOT NULL,
            entry_date TIMESTAMP NOT NULL,
            exit_date TIMESTAMP,
            pnl DECIMAL(15, 2),
            return_pct DECIMAL(10, 6),
            strategy_signal VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (backtesting_result_id) REFERENCES backtesting_results(id) ON DELETE CASCADE,
            INDEX idx_backtesting_result (backtesting_result_id),
            INDEX idx_symbol (symbol),
            INDEX idx_entry_date (entry_date),
            INDEX idx_trade_type (trade_type)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        
        # Execute migration
        for statement in migration_sql.split(';'):
            if statement.strip():
                cursor.execute(statement)
                print(f"✅ Executed: {statement.strip()[:50]}...")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("🎉 Database migration completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1)
