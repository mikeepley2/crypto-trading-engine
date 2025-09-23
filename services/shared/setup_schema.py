#!/usr/bin/env python3
"""
Setup Enhanced Trading Engine V2 Database Schema
"""

import mysql.connector
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_database_schema():
    """Setup the Enhanced Trading Engine V2 database schema"""
    
    # Database configuration
    db_config = {
        'host': 'host.docker.internal',
        'user': 'news_collector',
        'password': '99Rules!',
        'database': 'crypto_prices'
    }
    
    # Read the schema file
    try:
        with open('enhanced_trading_engine_v2_schema.sql', 'r') as f:
            schema_sql = f.read()
    except FileNotFoundError:
        logger.error("❌ Schema file not found: enhanced_trading_engine_v2_schema.sql")
        return False
    
    # Split the SQL into individual statements
    statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
    
    try:
        # Connect to database
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        
        logger.info("🔗 Connected to database")
        
        # Execute each statement
        for i, statement in enumerate(statements):
            if statement and not statement.startswith('--'):  # Skip comments
                try:
                    cursor.execute(statement)
                    
                    # Consume all result sets to avoid "Unread result found" error
                    try:
                        while cursor.nextset():
                            pass
                    except:
                        pass
                    
                    logger.info(f"✅ Executed statement {i+1}/{len(statements)}")
                except mysql.connector.Error as e:
                    if "already exists" in str(e):
                        logger.info(f"⚠️ Statement {i+1}: {e}")
                    else:
                        logger.error(f"❌ Statement {i+1} failed: {e}")
        
        # Commit changes
        connection.commit()
        
        # Show created tables
        cursor.execute("SHOW TABLES LIKE '%v2%'")
        tables = cursor.fetchall()
        
        logger.info("📊 Created tables:")
        for table in tables:
            logger.info(f"   - {table[0]}")
        
        cursor.close()
        connection.close()
        
        logger.info("✅ Enhanced Trading Engine V2 schema setup complete!")
        return True
        
    except mysql.connector.Error as e:
        logger.error(f"❌ Database error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Setup error: {e}")
        return False

if __name__ == "__main__":
    success = setup_database_schema()
    if success:
        print("\n🚀 Enhanced Trading Engine V2 database ready!")
    else:
        print("\n❌ Database setup failed!")
        exit(1)
