#!/usr/bin/env python3
"""
Unified Sentiment Database Schema Setup
Creates the unified sentiment infrastructure for consistent data storage across all sentiment sources
"""

import mysql.connector
import logging
from datetime import datetime
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UnifiedSentimentSchema:
    """
    Sets up unified sentiment database schema for consistent sentiment storage
    """
    
    def __init__(self):
        self.db_config = {
            'host': 'host.docker.internal',
            'user': 'news_collector',
            'password': '99Rules!',
            'database': 'crypto_prices'
        }
        
    def create_unified_sentiment_table(self):
        """Create the unified sentiment data table"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Create unified sentiment data table
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS unified_sentiment_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                
                -- Source identification
                source_type ENUM('news', 'twitter', 'reddit', 'telegram', 'discord') NOT NULL,
                source_id VARCHAR(100) NOT NULL,
                
                -- Content identification
                symbol VARCHAR(10) NOT NULL,
                content_text TEXT NOT NULL,
                content_title VARCHAR(500),
                
                -- Timing
                content_timestamp DATETIME NOT NULL,
                collection_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                -- Sentiment analysis results
                composite_sentiment DECIMAL(10,6) NOT NULL,
                confidence_score DECIMAL(10,6) NOT NULL,
                
                -- Individual model scores
                vader_score DECIMAL(10,6),
                vader_confidence DECIMAL(10,6),
                textblob_score DECIMAL(10,6),
                textblob_confidence DECIMAL(10,6),
                cryptobert_score DECIMAL(10,6),
                cryptobert_confidence DECIMAL(10,6),
                
                -- Context and influence
                influence_score DECIMAL(10,6) DEFAULT 0,
                engagement_metrics JSON,
                crypto_keywords JSON,
                
                -- Classification
                sentiment_label ENUM('bullish', 'bearish', 'neutral') NOT NULL,
                signal_strength ENUM('weak', 'moderate', 'strong') NOT NULL,
                
                -- Metadata
                metadata JSON,
                processing_version VARCHAR(20) DEFAULT 'v1.0',
                
                -- Indexes for performance
                INDEX idx_symbol_timestamp (symbol, content_timestamp),
                INDEX idx_source_type (source_type, collection_timestamp),
                INDEX idx_sentiment_strength (composite_sentiment, signal_strength),
                INDEX idx_confidence (confidence_score),
                INDEX idx_influence (influence_score),
                
                UNIQUE KEY unique_source_content (source_type, source_id, symbol)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            
            cursor.execute(create_table_sql)
            logger.info("✅ Created unified_sentiment_data table")
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating unified sentiment table: {e}")
            return False
    
    def create_sentiment_aggregation_table(self):
        """Create enhanced sentiment aggregation table"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Create enhanced aggregation table
            create_agg_table_sql = """
            CREATE TABLE IF NOT EXISTS sentiment_aggregation (
                id INT AUTO_INCREMENT PRIMARY KEY,
                
                -- Time and symbol
                timestamp DATETIME NOT NULL,
                symbol VARCHAR(10) NOT NULL,
                time_window ENUM('1m', '5m', '15m', '1h', '4h', '1d') NOT NULL,
                
                -- Aggregated sentiment metrics
                composite_sentiment DECIMAL(10,6) NOT NULL,
                confidence_score DECIMAL(10,6) NOT NULL,
                signal_strength ENUM('weak', 'moderate', 'strong') NOT NULL,
                sentiment_label ENUM('bullish', 'bearish', 'neutral') NOT NULL,
                
                -- Source breakdown
                news_sentiment DECIMAL(10,6) DEFAULT NULL,
                news_count INT DEFAULT 0,
                social_sentiment DECIMAL(10,6) DEFAULT NULL,
                social_count INT DEFAULT 0,
                
                -- Volume and engagement
                total_data_points INT NOT NULL,
                total_influence DECIMAL(15,6) DEFAULT 0,
                weighted_sentiment DECIMAL(10,6) NOT NULL,
                
                -- Trend analysis
                sentiment_momentum DECIMAL(10,6) DEFAULT 0,
                volatility DECIMAL(10,6) DEFAULT 0,
                trend_direction ENUM('bullish', 'bearish', 'neutral') DEFAULT 'neutral',
                
                -- Quality metrics
                data_quality_score DECIMAL(5,3) DEFAULT 0,
                source_diversity DECIMAL(5,3) DEFAULT 0,
                
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                -- Indexes
                INDEX idx_symbol_time (symbol, timestamp),
                INDEX idx_time_window (time_window, timestamp),
                INDEX idx_sentiment_strength (composite_sentiment, signal_strength),
                INDEX idx_trend (trend_direction, sentiment_momentum),
                
                UNIQUE KEY unique_symbol_time_window (symbol, timestamp, time_window)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
            
            cursor.execute(create_agg_table_sql)
            logger.info("✅ Created sentiment_aggregation table")
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating sentiment aggregation table: {e}")
            return False
    
    def create_trading_signals_table(self):
        """Create enhanced trading signals table based on sentiment"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            create_signals_table_sql = """
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INT AUTO_INCREMENT PRIMARY KEY,
                
                -- Signal identification
                signal_id VARCHAR(64) UNIQUE NOT NULL,
                timestamp DATETIME NOT NULL,
                symbol VARCHAR(10) NOT NULL,
                
                -- Signal details
                signal_type ENUM('BUY', 'SELL', 'HOLD') NOT NULL,
                signal_strength DECIMAL(5,3) NOT NULL,
                confidence DECIMAL(5,3) NOT NULL,
                
                -- Sentiment basis
                sentiment_score DECIMAL(10,6) NOT NULL,
                sentiment_momentum DECIMAL(10,6) NOT NULL,
                sentiment_volatility DECIMAL(10,6) NOT NULL,
                
                -- Signal parameters
                entry_price DECIMAL(15,8),
                target_price DECIMAL(15,8),
                stop_loss DECIMAL(15,8),
                position_size DECIMAL(10,6),
                
                -- Risk metrics
                risk_score DECIMAL(5,3) DEFAULT 0,
                max_exposure DECIMAL(10,6) DEFAULT 0,
                
                -- Data sources
                news_weight DECIMAL(5,3) DEFAULT 0,
                social_weight DECIMAL(5,3) DEFAULT 0,
                source_count INT NOT NULL,
                
                -- Execution tracking
                status ENUM('pending', 'active', 'executed', 'cancelled', 'expired') DEFAULT 'pending',
                executed_at DATETIME,
                execution_price DECIMAL(15,8),
                
                -- Performance tracking
                pnl DECIMAL(15,8),
                pnl_percentage DECIMAL(8,4),
                
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                
                -- Indexes
                INDEX idx_symbol_timestamp (symbol, timestamp),
                INDEX idx_signal_type (signal_type, timestamp),
                INDEX idx_status (status, timestamp),
                INDEX idx_confidence (confidence, signal_strength),
                INDEX idx_performance (pnl_percentage, executed_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
            
            cursor.execute(create_signals_table_sql)
            logger.info("✅ Created trading_signals table")
            
            conn.commit()
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating sentiment trading signals table: {e}")
            return False
    
    def setup_unified_schema(self):
        """Set up complete unified sentiment schema"""
        logger.info("🚀 Setting up unified sentiment database schema...")
        
        success = True
        
        # Create main tables
        if not self.create_unified_sentiment_table():
            success = False
            
        if not self.create_sentiment_aggregation_table():
            success = False
            
        if not self.create_trading_signals_table():
            success = False
        
        if success:
            logger.info("✅ Unified sentiment schema setup completed successfully!")
            self.print_schema_summary()
        else:
            logger.error("❌ Schema setup completed with errors")
            
        return success
    
    def print_schema_summary(self):
        """Print summary of created schema"""
        print("\n" + "="*80)
        print("UNIFIED SENTIMENT DATABASE SCHEMA SUMMARY")
        print("="*80)
        print("📊 Tables Created:")
        print("  1. unified_sentiment_data - Standardized sentiment collection from all sources")
        print("  2. sentiment_aggregation - Time-window aggregated sentiment metrics")
        print("  3. trading_signals - Trading signals based on sentiment analysis")
        print("\n🔄 Data Flow:")
        print("  News/Twitter/Reddit → unified_sentiment_data → sentiment_aggregation → trading_signals")
        print("\n🎯 Benefits:")
        print("  • Consistent sentiment analysis across all sources")
        print("  • Optimized for time-series analysis and trading")
        print("  • Comprehensive metadata and performance tracking")
        print("  • Backward compatible with existing systems")
        print("="*80)

def main():
    """Main function to set up unified sentiment schema"""
    try:
        schema_manager = UnifiedSentimentSchema()
        success = schema_manager.setup_unified_schema()
        
        if success:
            print("\n🎉 Unified sentiment schema is ready for use!")
            sys.exit(0)
        else:
            print("\n❌ Schema setup failed. Check logs for details.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Critical error in schema setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
