-- Enhanced Trading Engine V2 Database Schema
-- Creates tables for the unified trading engine that combines legacy functionality with modern architecture

USE crypto_prices;

-- Enhanced trading signals table
CREATE TABLE IF NOT EXISTS trading_signals_v2 (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    price DECIMAL(15,8) NOT NULL,
    signal_type ENUM('BUY', 'SELL', 'HOLD', 'STRONG_BUY', 'STRONG_SELL') NOT NULL,
    confidence DECIMAL(6,4) NOT NULL,
    threshold DECIMAL(6,4) NOT NULL,
    regime ENUM('strong_bull', 'bull', 'sideways', 'bear', 'strong_bear') NOT NULL,
    
    -- XGBoost model information
    model_version VARCHAR(50) NOT NULL DEFAULT 'xgboost_4h',
    features_used INT NOT NULL DEFAULT 0,
    xgboost_confidence DECIMAL(6,4) NOT NULL,
    
    -- Data source tracking
    data_source VARCHAR(50) NOT NULL DEFAULT 'database',
    real_time_available BOOLEAN DEFAULT FALSE,
    
    -- Market context
    volume_24h DECIMAL(20,8) DEFAULT NULL,
    rsi DECIMAL(6,2) DEFAULT NULL,
    crypto_sentiment DECIMAL(6,4) DEFAULT NULL,
    vix DECIMAL(6,2) DEFAULT NULL,
    
    -- LLM analysis (if available)
    llm_analysis JSON DEFAULT NULL,
    llm_confidence DECIMAL(6,4) DEFAULT NULL,
    llm_reasoning TEXT DEFAULT NULL,
    
    -- Performance tracking
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_symbol_timestamp (symbol, timestamp),
    INDEX idx_signal_type (signal_type),
    INDEX idx_confidence (confidence),
    INDEX idx_regime (regime),
    INDEX idx_timestamp (timestamp),
    INDEX idx_symbol (symbol)
);

-- Performance metrics table for trading engine monitoring
CREATE TABLE IF NOT EXISTS trading_engine_v2_metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    engine_version VARCHAR(50) NOT NULL DEFAULT 'enhanced_v2',
    
    -- Signal generation metrics
    total_signals INT DEFAULT 0,
    buy_signals INT DEFAULT 0,
    hold_signals INT DEFAULT 0,
    avg_confidence DECIMAL(6,4) DEFAULT NULL,
    
    -- Error tracking
    api_errors INT DEFAULT 0,
    db_errors INT DEFAULT 0,
    feature_engineering_errors INT DEFAULT 0,
    model_prediction_errors INT DEFAULT 0,
    
    -- Performance indicators
    cycle_duration_seconds DECIMAL(8,2) DEFAULT NULL,
    symbols_processed INT DEFAULT 0,
    real_time_data_available BOOLEAN DEFAULT FALSE,
    llm_enabled BOOLEAN DEFAULT FALSE,
    
    -- System information
    session_id VARCHAR(50) DEFAULT NULL,
    host_info VARCHAR(100) DEFAULT NULL,
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_timestamp (timestamp),
    INDEX idx_engine_version (engine_version),
    INDEX idx_session_id (session_id)
);

-- Model performance tracking table
CREATE TABLE IF NOT EXISTS model_performance_v2 (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    
    -- Prediction vs actual tracking
    predicted_signal ENUM('BUY', 'SELL', 'HOLD', 'STRONG_BUY', 'STRONG_SELL') NOT NULL,
    predicted_confidence DECIMAL(6,4) NOT NULL,
    actual_price_change_1h DECIMAL(10,6) DEFAULT NULL,
    actual_price_change_4h DECIMAL(10,6) DEFAULT NULL,
    actual_price_change_24h DECIMAL(10,6) DEFAULT NULL,
    
    -- Performance metrics
    prediction_accuracy DECIMAL(6,4) DEFAULT NULL,
    sharpe_ratio DECIMAL(8,4) DEFAULT NULL,
    max_drawdown DECIMAL(6,4) DEFAULT NULL,
    
    -- Feature importance tracking
    top_features JSON DEFAULT NULL,
    feature_count INT DEFAULT 0,
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_model_symbol (model_version, symbol),
    INDEX idx_timestamp (timestamp),
    INDEX idx_symbol (symbol),
    INDEX idx_model_version (model_version)
);

-- Market regime tracking for analysis
CREATE TABLE IF NOT EXISTS market_regimes_v2 (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    regime ENUM('strong_bull', 'bull', 'sideways', 'bear', 'strong_bear') NOT NULL,
    
    -- Regime indicators
    bull_score INT NOT NULL DEFAULT 0,
    threshold_used DECIMAL(6,4) NOT NULL,
    
    -- Market context at time of regime detection
    price DECIMAL(15,8) NOT NULL,
    sma_20 DECIMAL(15,8) DEFAULT NULL,
    rsi_14 DECIMAL(6,2) DEFAULT NULL,
    macd_line DECIMAL(10,6) DEFAULT NULL,
    vix DECIMAL(6,2) DEFAULT NULL,
    crypto_sentiment DECIMAL(6,4) DEFAULT NULL,
    
    -- Duration tracking
    regime_start_time DATETIME DEFAULT NULL,
    regime_duration_hours DECIMAL(8,2) DEFAULT NULL,
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_symbol_timestamp (symbol, timestamp),
    INDEX idx_regime (regime),
    INDEX idx_timestamp (timestamp),
    INDEX idx_symbol (symbol),
    INDEX idx_bull_score (bull_score)
);

-- Feature engineering quality tracking
CREATE TABLE IF NOT EXISTS feature_quality_v2 (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    
    -- Feature set completeness
    total_features_expected INT DEFAULT 113,
    total_features_generated INT DEFAULT 0,
    database_features_count INT DEFAULT 0,
    realtime_features_count INT DEFAULT 0,
    
    -- Data quality indicators
    null_feature_count INT DEFAULT 0,
    invalid_feature_count INT DEFAULT 0,
    stale_data_minutes INT DEFAULT 0,
    
    -- Data sources
    database_connection_ok BOOLEAN DEFAULT FALSE,
    realtime_api_ok BOOLEAN DEFAULT FALSE,
    feature_engineering_time_ms INT DEFAULT 0,
    
    -- Feature categories
    price_features INT DEFAULT 0,
    volume_features INT DEFAULT 0,
    technical_features INT DEFAULT 0,
    sentiment_features INT DEFAULT 0,
    macro_features INT DEFAULT 0,
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_symbol_timestamp (symbol, timestamp),
    INDEX idx_timestamp (timestamp),
    INDEX idx_symbol (symbol),
    INDEX idx_data_quality (null_feature_count, invalid_feature_count)
);

-- Create initial performance summary view
CREATE OR REPLACE VIEW trading_engine_v2_summary AS
SELECT 
    DATE(timestamp) as trading_date,
    COUNT(*) as total_signals,
    SUM(CASE WHEN signal_type IN ('BUY', 'STRONG_BUY') THEN 1 ELSE 0 END) as buy_signals,
    SUM(CASE WHEN signal_type = 'HOLD' THEN 1 ELSE 0 END) as hold_signals,
    AVG(confidence) as avg_confidence,
    AVG(features_used) as avg_features_used,
    COUNT(DISTINCT symbol) as symbols_traded,
    MIN(timestamp) as first_signal,
    MAX(timestamp) as last_signal
FROM trading_signals_v2 
GROUP BY DATE(timestamp)
ORDER BY trading_date DESC;

-- Daily regime summary view
CREATE OR REPLACE VIEW daily_regime_summary AS
SELECT 
    DATE(timestamp) as trading_date,
    symbol,
    regime,
    COUNT(*) as regime_count,
    AVG(bull_score) as avg_bull_score,
    AVG(threshold_used) as avg_threshold,
    MIN(timestamp) as regime_start,
    MAX(timestamp) as regime_end
FROM market_regimes_v2 
GROUP BY DATE(timestamp), symbol, regime
ORDER BY trading_date DESC, symbol, regime;

-- Show table creation results
SELECT 'Enhanced Trading Engine V2 schema created successfully' as status;

-- Show table structure
SHOW TABLES LIKE '%trading%v2%' OR SHOW TABLES LIKE '%regime%v2%' OR SHOW TABLES LIKE '%feature%v2%';

-- Show indexes created
SELECT 
    table_name,
    index_name,
    column_name,
    index_type
FROM information_schema.statistics 
WHERE table_schema = 'crypto_prices' 
AND table_name LIKE '%v2%'
ORDER BY table_name, index_name;
