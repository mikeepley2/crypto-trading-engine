-- Add LLM validation fields to trade_recommendations table
-- This script adds fields to support LLM validation of trade recommendations

USE crypto_prices;

-- Add LLM validation fields to trade_recommendations table
ALTER TABLE trade_recommendations 
ADD COLUMN IF NOT EXISTS llm_validation VARCHAR(20) DEFAULT NULL COMMENT 'LLM validation result: APPROVE, REJECT, MODIFY',
ADD COLUMN IF NOT EXISTS llm_confidence DECIMAL(3,2) DEFAULT NULL COMMENT 'LLM confidence score 0.0-1.0',
ADD COLUMN IF NOT EXISTS llm_reasoning TEXT DEFAULT NULL COMMENT 'LLM reasoning for validation decision',
ADD COLUMN IF NOT EXISTS risk_assessment VARCHAR(10) DEFAULT NULL COMMENT 'Risk assessment: LOW, MEDIUM, HIGH',
ADD COLUMN IF NOT EXISTS suggested_amount DECIMAL(15,8) DEFAULT NULL COMMENT 'LLM suggested trade amount',
ADD COLUMN IF NOT EXISTS validation_timestamp TIMESTAMP NULL DEFAULT NULL COMMENT 'When LLM validation was performed';

-- Add index for faster queries on validation status
CREATE INDEX IF NOT EXISTS idx_llm_validation ON trade_recommendations(llm_validation);
CREATE INDEX IF NOT EXISTS idx_validation_timestamp ON trade_recommendations(validation_timestamp);

-- Show the updated table structure
DESCRIBE trade_recommendations;
