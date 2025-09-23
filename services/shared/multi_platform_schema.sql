-- Multi-Platform Trading Database Schema
-- Adds support for multiple trading platforms to the existing system

USE crypto_transactions;

-- Drop existing foreign key constraints that reference old tables
-- (Add error handling in case constraints don't exist)

-- Create trading platforms table
CREATE TABLE IF NOT EXISTS trading_platforms (
    id INT AUTO_INCREMENT PRIMARY KEY,
    platform_name VARCHAR(50) NOT NULL UNIQUE,
    display_name VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    base_url VARCHAR(255),
    api_version VARCHAR(20),
    rate_limit_per_minute INT DEFAULT 1000,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_platform_name (platform_name),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB;

-- Insert default platforms
INSERT IGNORE INTO trading_platforms (platform_name, display_name, base_url, api_version) VALUES
('coinbase', 'Coinbase Advanced Trade', 'https://api.coinbase.com', 'v3'),
('binance_us', 'Binance US', 'https://api.binance.us', 'v3'),
('kucoin', 'KuCoin', 'https://api.kucoin.com', 'v1');

-- Create supported assets table per platform
CREATE TABLE IF NOT EXISTS platform_assets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    platform_id INT NOT NULL,
    asset_symbol VARCHAR(20) NOT NULL,
    asset_name VARCHAR(100),
    platform_symbol VARCHAR(20) NOT NULL, -- Platform-specific symbol format
    is_active BOOLEAN DEFAULT TRUE,
    min_order_size DECIMAL(20,8),
    max_order_size DECIMAL(20,8),
    precision_digits INT DEFAULT 8,
    can_trade BOOLEAN DEFAULT TRUE,
    can_deposit BOOLEAN DEFAULT TRUE,
    can_withdraw BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (platform_id) REFERENCES trading_platforms(id),
    UNIQUE KEY unique_platform_asset (platform_id, asset_symbol),
    INDEX idx_asset_symbol (asset_symbol),
    INDEX idx_platform_symbol (platform_symbol),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB;

-- Create trading pairs table per platform
CREATE TABLE IF NOT EXISTS platform_trading_pairs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    platform_id INT NOT NULL,
    symbol VARCHAR(20) NOT NULL, -- Standardized symbol (BTC-USD)
    platform_symbol VARCHAR(20) NOT NULL, -- Platform-specific symbol (BTCUSD, BTC/USD, etc.)
    base_asset VARCHAR(20) NOT NULL,
    quote_asset VARCHAR(20) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    min_order_size DECIMAL(20,8),
    max_order_size DECIMAL(20,8),
    price_precision INT DEFAULT 2,
    quantity_precision INT DEFAULT 8,
    maker_fee DECIMAL(6,4) DEFAULT 0.0050,
    taker_fee DECIMAL(6,4) DEFAULT 0.0050,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (platform_id) REFERENCES trading_platforms(id),
    UNIQUE KEY unique_platform_pair (platform_id, symbol),
    INDEX idx_symbol (symbol),
    INDEX idx_platform_symbol (platform_symbol),
    INDEX idx_base_asset (base_asset),
    INDEX idx_quote_asset (quote_asset),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB;

-- Update trades table to include platform information
ALTER TABLE trades 
ADD COLUMN IF NOT EXISTS platform_id INT,
ADD COLUMN IF NOT EXISTS platform_order_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS client_order_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS fee_asset VARCHAR(20) DEFAULT 'USD',
ADD COLUMN IF NOT EXISTS trade_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS maker_taker ENUM('maker', 'taker'),
ADD COLUMN IF NOT EXISTS platform_symbol VARCHAR(20);

-- Add foreign key constraint for platform_id
ALTER TABLE trades 
ADD CONSTRAINT fk_trades_platform 
FOREIGN KEY (platform_id) REFERENCES trading_platforms(id),
ADD INDEX idx_platform_id (platform_id),
ADD INDEX idx_platform_order_id (platform_order_id),
ADD INDEX idx_trade_id (trade_id);

-- Update mock_trades table similarly
ALTER TABLE mock_trades 
ADD COLUMN IF NOT EXISTS platform_id INT DEFAULT 1, -- Default to Coinbase for existing records
ADD COLUMN IF NOT EXISTS platform_order_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS client_order_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS fee_asset VARCHAR(20) DEFAULT 'USD',
ADD COLUMN IF NOT EXISTS trade_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS maker_taker ENUM('maker', 'taker'),
ADD COLUMN IF NOT EXISTS platform_symbol VARCHAR(20);

-- Add foreign key constraint for mock_trades
ALTER TABLE mock_trades 
ADD CONSTRAINT fk_mock_trades_platform 
FOREIGN KEY (platform_id) REFERENCES trading_platforms(id),
ADD INDEX idx_mock_platform_id (platform_id),
ADD INDEX idx_mock_platform_order_id (platform_order_id),
ADD INDEX idx_mock_trade_id (trade_id);

-- Create portfolio positions table per platform
CREATE TABLE IF NOT EXISTS portfolio_positions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    platform_id INT NOT NULL,
    asset VARCHAR(20) NOT NULL,
    total_quantity DECIMAL(20,8) NOT NULL DEFAULT 0,
    available_quantity DECIMAL(20,8) NOT NULL DEFAULT 0,
    locked_quantity DECIMAL(20,8) NOT NULL DEFAULT 0,
    average_cost DECIMAL(20,8),
    total_cost_basis DECIMAL(20,8),
    current_price DECIMAL(20,8),
    unrealized_pnl DECIMAL(20,8),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (platform_id) REFERENCES trading_platforms(id),
    UNIQUE KEY unique_platform_asset_position (platform_id, asset),
    INDEX idx_asset (asset),
    INDEX idx_platform_id (platform_id),
    INDEX idx_last_updated (last_updated)
) ENGINE=InnoDB;

-- Update trade_recommendations table to include platform
ALTER TABLE trade_recommendations 
ADD COLUMN IF NOT EXISTS platform_id INT,
ADD COLUMN IF NOT EXISTS target_platform VARCHAR(50) DEFAULT 'coinbase';

-- Add foreign key constraint for trade_recommendations
ALTER TABLE trade_recommendations 
ADD CONSTRAINT fk_recommendations_platform 
FOREIGN KEY (platform_id) REFERENCES trading_platforms(id),
ADD INDEX idx_recommendations_platform_id (platform_id),
ADD INDEX idx_target_platform (target_platform);

-- Create platform API credentials table (encrypted)
CREATE TABLE IF NOT EXISTS platform_credentials (
    id INT AUTO_INCREMENT PRIMARY KEY,
    platform_id INT NOT NULL,
    credential_name VARCHAR(100) NOT NULL, -- e.g., 'api_key', 'secret_key', 'passphrase'
    encrypted_value TEXT NOT NULL, -- Encrypted credential value
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (platform_id) REFERENCES trading_platforms(id),
    UNIQUE KEY unique_platform_credential (platform_id, credential_name),
    INDEX idx_platform_id (platform_id),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB;

-- Create platform configuration table
CREATE TABLE IF NOT EXISTS platform_configurations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    platform_id INT NOT NULL,
    config_key VARCHAR(100) NOT NULL,
    config_value TEXT,
    config_type ENUM('string', 'integer', 'decimal', 'boolean', 'json') DEFAULT 'string',
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (platform_id) REFERENCES trading_platforms(id),
    UNIQUE KEY unique_platform_config (platform_id, config_key),
    INDEX idx_platform_id (platform_id),
    INDEX idx_config_key (config_key)
) ENGINE=InnoDB;

-- Create platform health status table
CREATE TABLE IF NOT EXISTS platform_health_status (
    id INT AUTO_INCREMENT PRIMARY KEY,
    platform_id INT NOT NULL,
    is_connected BOOLEAN DEFAULT FALSE,
    is_trading_enabled BOOLEAN DEFAULT FALSE,
    last_ping TIMESTAMP,
    last_error TEXT,
    rate_limit_remaining INT,
    response_time_ms INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (platform_id) REFERENCES trading_platforms(id),
    INDEX idx_platform_id (platform_id),
    INDEX idx_last_ping (last_ping),
    INDEX idx_is_connected (is_connected)
) ENGINE=InnoDB;

-- Create symbol mapping table for cross-platform symbol normalization
CREATE TABLE IF NOT EXISTS symbol_mappings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    standard_symbol VARCHAR(20) NOT NULL, -- Our standardized format (BTC-USD)
    platform_id INT NOT NULL,
    platform_symbol VARCHAR(20) NOT NULL, -- Platform-specific format
    base_asset VARCHAR(20) NOT NULL,
    quote_asset VARCHAR(20) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (platform_id) REFERENCES trading_platforms(id),
    UNIQUE KEY unique_platform_symbol_mapping (platform_id, platform_symbol),
    INDEX idx_standard_symbol (standard_symbol),
    INDEX idx_platform_symbol (platform_symbol),
    INDEX idx_base_asset (base_asset),
    INDEX idx_quote_asset (quote_asset)
) ENGINE=InnoDB;

-- Insert default symbol mappings for common pairs
INSERT IGNORE INTO symbol_mappings (standard_symbol, platform_id, platform_symbol, base_asset, quote_asset) 
SELECT 'BTC-USD', p.id, 
    CASE 
        WHEN p.platform_name = 'coinbase' THEN 'BTC-USD'
        WHEN p.platform_name = 'binance_us' THEN 'BTCUSD'
        WHEN p.platform_name = 'kucoin' THEN 'BTC-USDT'
    END,
    'BTC', 
    CASE 
        WHEN p.platform_name = 'kucoin' THEN 'USDT'
        ELSE 'USD'
    END
FROM trading_platforms p;

INSERT IGNORE INTO symbol_mappings (standard_symbol, platform_id, platform_symbol, base_asset, quote_asset) 
SELECT 'ETH-USD', p.id, 
    CASE 
        WHEN p.platform_name = 'coinbase' THEN 'ETH-USD'
        WHEN p.platform_name = 'binance_us' THEN 'ETHUSD'
        WHEN p.platform_name = 'kucoin' THEN 'ETH-USDT'
    END,
    'ETH',
    CASE 
        WHEN p.platform_name = 'kucoin' THEN 'USDT'
        ELSE 'USD'
    END
FROM trading_platforms p;

-- Create platform performance metrics table
CREATE TABLE IF NOT EXISTS platform_performance_metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    platform_id INT NOT NULL,
    metric_date DATE NOT NULL,
    total_trades INT DEFAULT 0,
    total_volume_usd DECIMAL(20,2) DEFAULT 0,
    total_fees_usd DECIMAL(20,2) DEFAULT 0,
    successful_trades INT DEFAULT 0,
    failed_trades INT DEFAULT 0,
    average_response_time_ms INT,
    uptime_percentage DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (platform_id) REFERENCES trading_platforms(id),
    UNIQUE KEY unique_platform_date (platform_id, metric_date),
    INDEX idx_platform_id (platform_id),
    INDEX idx_metric_date (metric_date)
) ENGINE=InnoDB;

-- Update existing data to reference Coinbase platform (platform_id = 1)
UPDATE trades SET platform_id = 1 WHERE platform_id IS NULL;
UPDATE mock_trades SET platform_id = 1 WHERE platform_id IS NULL;
UPDATE trade_recommendations SET platform_id = 1 WHERE platform_id IS NULL;

-- Create view for unified portfolio across all platforms
CREATE OR REPLACE VIEW unified_portfolio AS
SELECT 
    asset,
    SUM(total_quantity) as total_quantity_all_platforms,
    SUM(available_quantity) as available_quantity_all_platforms,
    SUM(locked_quantity) as locked_quantity_all_platforms,
    AVG(average_cost) as average_cost_weighted,
    SUM(total_cost_basis) as total_cost_basis_all_platforms,
    AVG(current_price) as current_price,
    SUM(unrealized_pnl) as total_unrealized_pnl,
    COUNT(DISTINCT platform_id) as platforms_count,
    MAX(last_updated) as last_updated
FROM portfolio_positions
WHERE total_quantity > 0
GROUP BY asset;

-- Create view for platform-specific portfolio
CREATE OR REPLACE VIEW platform_portfolio_summary AS
SELECT 
    p.platform_name,
    p.display_name,
    COUNT(DISTINCT pp.asset) as unique_assets,
    SUM(pp.total_cost_basis) as total_portfolio_value,
    SUM(pp.unrealized_pnl) as total_unrealized_pnl,
    pp.last_updated
FROM trading_platforms p
LEFT JOIN portfolio_positions pp ON p.id = pp.platform_id
WHERE p.is_active = TRUE
GROUP BY p.id, p.platform_name, p.display_name, pp.last_updated;

-- Create index for better performance
CREATE INDEX idx_trades_platform_symbol_timestamp ON trades(platform_id, symbol, timestamp);
CREATE INDEX idx_mock_trades_platform_symbol_timestamp ON mock_trades(platform_id, symbol, timestamp);
CREATE INDEX idx_recommendations_platform_created ON trade_recommendations(platform_id, created_at);
