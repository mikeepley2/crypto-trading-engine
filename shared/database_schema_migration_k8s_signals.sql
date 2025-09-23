-- Database Schema Migration for K8s Signal Generation Integration
-- Fixes compatibility between K8s signal generation services and Docker Compose signal bridge
-- Date: September 5, 2025

USE crypto_prices;

-- Add missing columns to trading_signals table for K8s compatibility
ALTER TABLE trading_signals 
ADD COLUMN signal_id VARCHAR(128) UNIQUE COMMENT 'Unique identifier for signal from K8s services',
ADD COLUMN signal_strength DECIMAL(6,4) DEFAULT 1.0 COMMENT 'Signal strength score from ML models',
ADD COLUMN processed_at TIMESTAMP NULL COMMENT 'When signal was processed by trading bridge',
ADD INDEX idx_signal_id (signal_id),
ADD INDEX idx_processed_at (processed_at),
ADD INDEX idx_timestamp_confidence (timestamp, confidence);

-- Update existing records to have signal_id and processed_at
UPDATE trading_signals 
SET signal_id = CONCAT('legacy_', id, '_', UNIX_TIMESTAMP(timestamp))
WHERE signal_id IS NULL;

-- Set processed_at for already processed signals
UPDATE trading_signals 
SET processed_at = created_at 
WHERE processed = 1 AND processed_at IS NULL;

-- Set default signal_strength based on confidence
UPDATE trading_signals 
SET signal_strength = confidence 
WHERE signal_strength IS NULL;

-- Create index for performance on common queries
CREATE INDEX idx_unprocessed_signals ON trading_signals (timestamp, confidence, processed_at);

-- Verify the migration
SELECT 
    COUNT(*) as total_signals,
    COUNT(signal_id) as signals_with_id,
    COUNT(processed_at) as signals_with_processed_at,
    COUNT(signal_strength) as signals_with_strength
FROM trading_signals;

-- Show sample of updated records
SELECT 
    id, signal_id, symbol, signal_type, confidence, signal_strength, 
    timestamp, processed, processed_at
FROM trading_signals 
ORDER BY timestamp DESC 
LIMIT 5;

-- Display updated table structure
DESCRIBE trading_signals;