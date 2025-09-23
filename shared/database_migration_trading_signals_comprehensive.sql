-- Comprehensive trading_signals table schema migration
-- This creates a comprehensive schema that supports all signal generation services

USE crypto_prices;

-- First, let's see what currently exists
-- DESCRIBE trading_signals;

-- Drop existing table if we need to recreate (backup data first!)
-- CREATE TABLE trading_signals_backup AS SELECT * FROM trading_signals;

-- Create comprehensive trading_signals table
CREATE TABLE IF NOT EXISTS trading_signals_new (
    -- Core identification
    id INT AUTO_INCREMENT PRIMARY KEY,
    signal_id VARCHAR(100) UNIQUE,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Trading data
    symbol VARCHAR(20) NOT NULL,
    price DECIMAL(15,8) DEFAULT 0.0,
    signal_type VARCHAR(20) NOT NULL,
    signal_strength DECIMAL(5,4) DEFAULT 0.0,
    confidence DECIMAL(5,4) NOT NULL,
    position_size DECIMAL(15,8) DEFAULT 0.0,
    
    -- Model and analysis data
    model VARCHAR(50) DEFAULT 'orchestrator',
    model_name VARCHAR(50) DEFAULT 'orchestrator',
    model_version VARCHAR(20) DEFAULT 'v1.0',
    features_used INT DEFAULT 0,
    xgboost_confidence DECIMAL(5,4) DEFAULT 0.0,
    threshold DECIMAL(5,4) DEFAULT 0.5,
    
    -- Market context
    regime VARCHAR(20) DEFAULT 'UNCERTAIN',
    market_conditions JSON,
    
    -- Sentiment data
    sentiment_score DECIMAL(5,4) DEFAULT 0.0,
    sentiment_momentum DECIMAL(5,4) DEFAULT 0.0,
    sentiment_volatility DECIMAL(5,4) DEFAULT 0.0,
    
    -- Risk management
    risk_score DECIMAL(5,4) DEFAULT 0.5,
    risk_assessment JSON,
    entry_price DECIMAL(15,8) DEFAULT 0.0,
    
    -- Signal metadata
    source_count INT DEFAULT 1,
    data_source VARCHAR(50) DEFAULT 'orchestrator',
    is_mock TINYINT(1) DEFAULT 0,
    status VARCHAR(20) DEFAULT 'pending',
    
    -- Strategy details (for multi-strategy signals)
    strategies_total INT DEFAULT 1,
    strategies_bullish INT DEFAULT 0,
    strategies_bearish INT DEFAULT 0,
    strategies_execution_time_ms INT DEFAULT 0,
    strategy_full_details JSON,
    
    -- Individual strategy flags (from create_strategy_demo_data.py)
    moving_average_crossover TINYINT(1) DEFAULT 0,
    rsi_divergence TINYINT(1) DEFAULT 0,
    volume_spike TINYINT(1) DEFAULT 0,
    bollinger_breakout TINYINT(1) DEFAULT 0,
    macd_signal TINYINT(1) DEFAULT 0,
    support_resistance TINYINT(1) DEFAULT 0,
    fibonacci_retracement TINYINT(1) DEFAULT 0,
    momentum_oscillator TINYINT(1) DEFAULT 0,
    pattern_recognition TINYINT(1) DEFAULT 0,
    news_sentiment TINYINT(1) DEFAULT 0,
    social_sentiment TINYINT(1) DEFAULT 0,
    whale_movement TINYINT(1) DEFAULT 0,
    
    -- Indices for performance
    INDEX idx_symbol_created (symbol, created_at),
    INDEX idx_confidence (confidence),
    INDEX idx_signal_type (signal_type),
    INDEX idx_created_at (created_at),
    INDEX idx_is_mock (is_mock),
    INDEX idx_status (status)
);

-- Migrate existing data if table exists
INSERT IGNORE INTO trading_signals_new (
    symbol, signal_type, confidence, price, created_at, is_mock, timestamp
)
SELECT 
    symbol, 
    signal_type, 
    confidence, 
    COALESCE(price, 0.0) as price,
    COALESCE(created_at, NOW()) as created_at,
    COALESCE(is_mock, 0) as is_mock,
    COALESCE(created_at, NOW()) as timestamp
FROM trading_signals
WHERE NOT EXISTS (
    SELECT 1 FROM trading_signals_new tn 
    WHERE tn.symbol = trading_signals.symbol 
    AND tn.created_at = trading_signals.created_at
);

-- Rename tables (backup old, activate new)
RENAME TABLE trading_signals TO trading_signals_old_backup,
             trading_signals_new TO trading_signals;

-- Show the new schema
DESCRIBE trading_signals;
SELECT COUNT(*) as total_signals FROM trading_signals;