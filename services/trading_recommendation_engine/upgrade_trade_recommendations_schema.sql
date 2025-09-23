-- Add status column to trade_recommendations
ALTER TABLE trade_recommendations ADD COLUMN status VARCHAR(20) DEFAULT 'pending' AFTER is_mock;

-- Create audit_log table
CREATE TABLE IF NOT EXISTS audit_log (
	id INT AUTO_INCREMENT PRIMARY KEY,
	event_time DATETIME NOT NULL,
	event_type VARCHAR(50) NOT NULL,
	details TEXT
) ENGINE=InnoDB;
