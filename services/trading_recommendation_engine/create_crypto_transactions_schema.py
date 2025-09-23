import mysql.connector

# Update these as needed for your environment

# For WSL, update these as needed for your MySQL setup
MYSQL_CONFIG = {
	'host': '172.22.32.1',  # Windows host IP for WSL2
	'user': 'news_collector',
	'password': '99Rules!',
	'port': 3306
}

DB_NAME = 'crypto_transactions'

TABLES = {}

TABLES['holdings'] = (
	"""
	CREATE TABLE IF NOT EXISTS holdings (
		id INT AUTO_INCREMENT PRIMARY KEY,
		symbol VARCHAR(20) NOT NULL,
		quantity DECIMAL(20,8) NOT NULL,
		avg_entry_price DECIMAL(20,8) NOT NULL,
		realized_pnl DECIMAL(20,8) DEFAULT 0.0,
		updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
		is_mock BOOLEAN DEFAULT TRUE,
		UNIQUE(symbol, is_mock)
	) ENGINE=InnoDB;
	"""
)

TABLES['trades'] = (
	"""
	CREATE TABLE IF NOT EXISTS trades (
		id INT AUTO_INCREMENT PRIMARY KEY,
		timestamp DATETIME NOT NULL,
		symbol VARCHAR(20) NOT NULL,
		side ENUM('BUY', 'SELL') NOT NULL,
		quantity DECIMAL(20,8) NOT NULL,
		price DECIMAL(20,8) NOT NULL,
		fee DECIMAL(20,8) DEFAULT 0.0,
		order_type VARCHAR(20) DEFAULT 'MARKET',
		status VARCHAR(20) DEFAULT 'FILLED',
		is_mock BOOLEAN DEFAULT TRUE
	) ENGINE=InnoDB;
	"""
)

TABLES['trade_recommendations'] = (
	"""
	CREATE TABLE IF NOT EXISTS trade_recommendations (
		id INT AUTO_INCREMENT PRIMARY KEY,
		generated_at DATETIME NOT NULL,
		symbol VARCHAR(20) NOT NULL,
		action ENUM('BUY', 'SELL', 'HOLD') NOT NULL,
		confidence DECIMAL(5,4) DEFAULT NULL,
		entry_price DECIMAL(20,8) DEFAULT NULL,
		stop_loss DECIMAL(20,8) DEFAULT NULL,
		take_profit DECIMAL(20,8) DEFAULT NULL,
		position_size_percent DECIMAL(5,2) DEFAULT NULL,
		reasoning TEXT,
		is_mock BOOLEAN DEFAULT TRUE
	) ENGINE=InnoDB;
	"""
)


def create_database_and_tables():
	cnx = mysql.connector.connect(**MYSQL_CONFIG)
	cursor = cnx.cursor()
	cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
	cursor.execute(f"USE {DB_NAME}")
	for name, ddl in TABLES.items():
		print(f"Creating table {name}...")
		cursor.execute(ddl)
	cursor.close()
	cnx.close()

def insert_mock_holdings():
	cnx = mysql.connector.connect(database=DB_NAME, **MYSQL_CONFIG)
	cursor = cnx.cursor()
	mock_data = [
		("BTC", 0.5, 60000.00, 0.0, True),
		("ETH", 10.0, 3500.00, 0.0, True),
		("SOL", 100.0, 120.00, 0.0, True),
	]
	for symbol, qty, price, pnl, is_mock in mock_data:
		cursor.execute(
			"REPLACE INTO holdings (symbol, quantity, avg_entry_price, realized_pnl, is_mock) VALUES (%s, %s, %s, %s, %s)",
			(symbol, qty, price, pnl, is_mock)
		)
	cnx.commit()
	cursor.close()
	cnx.close()

if __name__ == '__main__':
	create_database_and_tables()
	insert_mock_holdings()
