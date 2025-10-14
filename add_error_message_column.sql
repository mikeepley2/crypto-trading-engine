-- Add error_message column to trade_recommendations table
USE crypto_transactions;

-- Check if column exists first
SELECT COLUMN_NAME 
FROM INFORMATION_SCHEMA.COLUMNS 
WHERE TABLE_SCHEMA = 'crypto_transactions' 
  AND TABLE_NAME = 'trade_recommendations' 
  AND COLUMN_NAME = 'error_message';

-- Add the column if it doesn't exist
ALTER TABLE trade_recommendations 
ADD COLUMN error_message TEXT DEFAULT NULL COMMENT 'Error message for failed trades';

-- Verify the column was added
DESCRIBE trade_recommendations;


