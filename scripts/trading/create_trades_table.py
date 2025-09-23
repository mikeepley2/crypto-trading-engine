#!/usr/bin/env python3
"""
Create the trades table in crypto_transactions database
"""
import mysql.connector
import sys

def create_trades_table():
    """Create the trades table with proper schema"""
    
    # Database configuration - using the standard config
    db_config = {
        'host': 'host.docker.internal',
        'user': 'news_collector',
        'password': '99Rules!',
        'database': 'crypto_transactions'
    }
    
    try:
        # Try localhost first if host.docker.internal doesn't work
        print("🔗 Connecting to MySQL database...")
        db = mysql.connector.connect(**db_config)
        cursor = db.cursor()
        
        # Check if trades table exists
        cursor.execute("SHOW TABLES LIKE 'trades'")
        table_exists = cursor.fetchone()
        
        if table_exists:
            print("✅ 'trades' table already exists")
            # Show current schema
            cursor.execute("DESCRIBE trades")
            schema = cursor.fetchall()
            print("\n📋 Current schema:")
            for row in schema:
                print(f"  {row[0]} - {row[1]} - {row[2]}")
        else:
            print("📋 Creating 'trades' table...")
            
            # Create the trades table with the schema expected by the trading engine
            create_table_sql = """
            CREATE TABLE trades (
                id INT AUTO_INCREMENT PRIMARY KEY,
                order_id VARCHAR(100) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                action VARCHAR(10) NOT NULL,
                size DECIMAL(20, 8) NOT NULL,
                price DECIMAL(20, 8) NOT NULL,
                size_usd DECIMAL(15, 2) NOT NULL,
                order_type VARCHAR(20) NOT NULL DEFAULT 'market',
                timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR(20) NOT NULL DEFAULT 'completed',
                INDEX idx_symbol (symbol),
                INDEX idx_timestamp (timestamp),
                INDEX idx_order_id (order_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
            
            cursor.execute(create_table_sql)
            db.commit()
            print("✅ 'trades' table created successfully!")
            
            # Show new schema
            cursor.execute("DESCRIBE trades")
            schema = cursor.fetchall()
            print("\n📋 New schema:")
            for row in schema:
                print(f"  {row[0]} - {row[1]} - {row[2]}")
        
        # Check if we have any existing trade data in other tables
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        trade_tables = [table[0] for table in tables if 'trade' in table[0].lower()]
        
        print(f"\n📊 Found {len(trade_tables)} trade-related tables:")
        for table in trade_tables:
            print(f"  - {table}")
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"    Records: {count}")
            except Exception as e:
                print(f"    Error counting: {e}")
        
        cursor.close()
        db.close()
        print("\n✅ Database operations completed successfully!")
        
    except mysql.connector.Error as e:
        if e.errno == 2003:
            print("❌ Cannot connect to MySQL - trying localhost...")
            # Try localhost instead
            db_config['host'] = 'localhost'
            try:
                db = mysql.connector.connect(**db_config)
                print("✅ Connected to localhost MySQL")
                # Repeat the same operations...
                create_trades_table_localhost(db)
            except Exception as e2:
                print(f"❌ Failed to connect to localhost MySQL: {e2}")
                print("💡 Make sure MySQL is running and accessible")
                return False
        else:
            print(f"❌ Database error: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

def create_trades_table_localhost(db):
    """Create trades table on localhost connection"""
    cursor = db.cursor()
    
    # Check if trades table exists
    cursor.execute("SHOW TABLES LIKE 'trades'")
    table_exists = cursor.fetchone()
    
    if table_exists:
        print("✅ 'trades' table already exists on localhost")
        # Show current schema
        cursor.execute("DESCRIBE trades")
        schema = cursor.fetchall()
        print("\n📋 Current schema:")
        for row in schema:
            print(f"  {row[0]} - {row[1]} - {row[2]}")
    else:
        print("📋 Creating 'trades' table on localhost...")
        
        create_table_sql = """
        CREATE TABLE trades (
            id INT AUTO_INCREMENT PRIMARY KEY,
            order_id VARCHAR(100) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            action VARCHAR(10) NOT NULL,
            size DECIMAL(20, 8) NOT NULL,
            price DECIMAL(20, 8) NOT NULL,
            size_usd DECIMAL(15, 2) NOT NULL,
            order_type VARCHAR(20) NOT NULL DEFAULT 'market',
            timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            status VARCHAR(20) NOT NULL DEFAULT 'completed',
            INDEX idx_symbol (symbol),
            INDEX idx_timestamp (timestamp),
            INDEX idx_order_id (order_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
        
        cursor.execute(create_table_sql)
        db.commit()
        print("✅ 'trades' table created successfully on localhost!")
        
        # Show new schema
        cursor.execute("DESCRIBE trades")
        schema = cursor.fetchall()
        print("\n📋 New schema:")
        for row in schema:
            print(f"  {row[0]} - {row[1]} - {row[2]}")
    
    # Check for other trade-related tables
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    trade_tables = [table[0] for table in tables if 'trade' in table[0].lower()]
    
    print(f"\n📊 Found {len(trade_tables)} trade-related tables:")
    for table in trade_tables:
        print(f"  - {table}")
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"    Records: {count}")
        except Exception as e:
            print(f"    Error counting: {e}")
    
    cursor.close()
    db.close()

if __name__ == "__main__":
    print("🚀 Starting trades table creation...")
    success = create_trades_table()
    if success:
        print("🎉 All done!")
        sys.exit(0)
    else:
        print("💥 Failed to create trades table")
        sys.exit(1)