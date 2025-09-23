from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
import sys
import os

import mysql.connector
import datetime

# Add path for health framework
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))

# Try to import health framework
try:
    from backend.shared.health_framework import HealthFramework, HealthStatus
    HEALTH_FRAMEWORK_AVAILABLE = True
except ImportError:
    HEALTH_FRAMEWORK_AVAILABLE = False
    print("Health framework not available, using basic health checks")

app = FastAPI(title="Trade Recommendation Service")

# Initialize health framework if available
if HEALTH_FRAMEWORK_AVAILABLE:
    health = HealthFramework(app, "trade-recommendation-service", "1.0.0")
    
    # Add database check for crypto_transactions
    health.add_database_check("primary_database", "crypto_transactions")
    
    # Add custom checks specific to this service
    @health.add_custom_check("recommendation_system")
    async def check_recommendation_system():
        """Check if recommendation system is operational"""
        try:
            cnx = mysql.connector.connect(**MYSQL_CONFIG)
            cursor = cnx.cursor(dictionary=True)
            
            # Check if we can access recommendations table
            cursor.execute("SELECT COUNT(*) as count FROM trade_recommendations LIMIT 1")
            result = cursor.fetchone()
            
            # Check recent recommendations
            cursor.execute("""
                SELECT COUNT(*) as recent_count 
                FROM trade_recommendations 
                WHERE generated_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
            """)
            recent_result = cursor.fetchone()
            
            cursor.close()
            cnx.close()
            
            status = HealthStatus.HEALTHY
            if recent_result['recent_count'] == 0:
                status = HealthStatus.DEGRADED  # No recent recommendations
            
            return {
                "status": status,
                "total_recommendations": result['count'],
                "recent_recommendations": recent_result['recent_count'],
                "database_accessible": True
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": str(e),
                "database_accessible": False
            }
    
    @health.add_custom_check("signal_processing")
    async def check_signal_processing():
        """Check signal processing capability"""
        try:
            cnx = mysql.connector.connect(
                host=MYSQL_CONFIG['host'],
                user=MYSQL_CONFIG['user'],
                password=MYSQL_CONFIG['password'],
                database='crypto_prices'  # Check signals from crypto_prices DB
            )
            cursor = cnx.cursor(dictionary=True)
            
            # Check if we can access trading signals
            cursor.execute("""
                SELECT COUNT(*) as signal_count 
                FROM trading_signals 
                WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
            """)
            result = cursor.fetchone()
            
            cursor.close()
            cnx.close()
            
            status = HealthStatus.HEALTHY if result['signal_count'] > 0 else HealthStatus.DEGRADED
            
            return {
                "status": status,
                "recent_signals": result['signal_count'],
                "signals_accessible": True
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.DEGRADED,  # Not critical if signals are unavailable temporarily
                "error": str(e),
                "signals_accessible": False
            }

MYSQL_CONFIG = {
	'host': os.environ.get('MYSQL_HOST', 'localhost'),
	'user': os.environ.get('MYSQL_USER', 'news_collector'),
	'password': os.environ.get('MYSQL_PASSWORD', '99Rules!'),
	'port': int(os.environ.get('MYSQL_PORT', 3306)),
	'database': os.environ.get('MYSQL_DATABASE', 'crypto_analysis'),
}

class TradeSignal(BaseModel):
	symbol: str
	action: str  # BUY, SELL, HOLD
	confidence: Optional[float] = None
	entry_price: Optional[float] = None
	stop_loss: Optional[float] = None
	take_profit: Optional[float] = None
	position_size_percent: Optional[float] = None
	reasoning: Optional[str] = None
	is_mock: bool = True
	timeframe: Optional[str] = None
	generated_at: Optional[str] = None

@app.get("/health")
def health():
	return {"status": "healthy"}

@app.get("/status")
def get_status():
	"""Get detailed service status."""
	try:
		# Test database connection
		cnx = mysql.connector.connect(**MYSQL_CONFIG)
		cursor = cnx.cursor()
		cursor.execute("SELECT 1")
		cursor.fetchone()
		cursor.close()
		cnx.close()
		db_connected = True
	except Exception as e:
		db_connected = False
		
	return {
		"status": "operational" if db_connected else "degraded", 
		"service": "trade_recommendation_service",
		"version": "1.0.0",
		"database_connected": db_connected,
		"timestamp": datetime.datetime.now().isoformat()
	}

@app.get("/metrics")
def get_metrics():
	"""Get service metrics."""
	try:
		cnx = mysql.connector.connect(**MYSQL_CONFIG)
		cursor = cnx.cursor(dictionary=True)
		
		# Get recent recommendations count
		cursor.execute("""
			SELECT COUNT(*) as recent_recommendations
			FROM trade_recommendations 
			WHERE generated_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
		""")
		recent_recs = cursor.fetchone()['recent_recommendations']
		
		# Get pending recommendations count
		cursor.execute("""
			SELECT COUNT(*) as pending_recommendations
			FROM trade_recommendations 
			WHERE execution_status = 'PENDING'
		""")
		pending_recs = cursor.fetchone()['pending_recommendations']
		
		cursor.close()
		cnx.close()
		
		return {
			"service": "trade_recommendation_service",
			"metrics": {
				"recommendations_24h": recent_recs,
				"pending_recommendations": pending_recs
			},
			"timestamp": datetime.datetime.now().isoformat()
		}
	except Exception as e:
		return {
			"error": f"Metrics collection failed: {e}",
			"timestamp": datetime.datetime.now().isoformat()
		}

@app.get("/holdings", response_model=List[dict])
def get_holdings(is_mock: bool = True):
	cnx = mysql.connector.connect(**MYSQL_CONFIG)
	cursor = cnx.cursor(dictionary=True)
	cursor.execute("SELECT * FROM holdings WHERE is_mock=%s", (is_mock,))
	results = cursor.fetchall()
	cursor.close()
	cnx.close()
	return results

@app.post("/recommendation")
async def create_recommendation(signal: TradeSignal, background_tasks: BackgroundTasks):
	async def insert_recommendation():
		cnx = mysql.connector.connect(**MYSQL_CONFIG)
		cursor = cnx.cursor()
		cursor.execute(
			"""
			INSERT INTO trade_recommendations (
				generated_at, symbol, action, confidence, entry_price, stop_loss, take_profit, position_size_percent, reasoning, is_mock
			) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
			""",
			(
				datetime.datetime.utcnow(),
				signal.symbol,
				signal.action,
				signal.confidence,
				signal.entry_price,
				signal.stop_loss,
				signal.take_profit,
				signal.position_size_percent,
				signal.reasoning,
				signal.is_mock
			)
		)
		rec_id = cursor.lastrowid
		cnx.commit()
		cursor.close()
		cnx.close()
		
		# Trigger mock trading engine if this is a high-confidence signal
		if signal.confidence and signal.confidence >= 0.75 and signal.action in ['BUY', 'SELL']:
			await trigger_mock_trading_engine(rec_id)
			
		return rec_id
	
	background_tasks.add_task(insert_recommendation)
	return {"status": "recommendation_queued"}

@app.get("/recommendations", response_model=List[dict])
def get_recommendations(is_mock: bool = True):
	cnx = mysql.connector.connect(**MYSQL_CONFIG)
	cursor = cnx.cursor(dictionary=True)
	cursor.execute("SELECT * FROM trade_recommendations WHERE is_mock=%s ORDER BY generated_at DESC", (is_mock,))
	results = cursor.fetchall()
	cursor.close()
	cnx.close()
	return results


# --- Signal Monitoring and Automated Recommendation Logic ---

import random
import requests
import httpx
from apscheduler.schedulers.background import BackgroundScheduler
import threading

# Unified Trading Engine (mock/live via EXECUTION_MODE)
MOCK_TRADING_ENGINE_URL = os.environ.get('MOCK_TRADING_ENGINE_URL', 'http://localhost:8024')



# --- Fetch signals from crypto_prices.trading_signals table ---
def fetch_real_signals():
	"""
	Fetch signals from the crypto_prices.trading_signals table in MySQL.
	"""
	signals = []
	try:
		# Use a separate connection config for crypto_prices DB
		db_cfg = MYSQL_CONFIG.copy()
		db_cfg['database'] = 'crypto_prices'
		cnx = mysql.connector.connect(**db_cfg)
		cursor = cnx.cursor(dictionary=True)
		cursor.execute("""
			SELECT symbol, signal_type as action, confidence, price as entry_price, NULL as stop_loss, NULL as take_profit, NULL as position_size_percent, regime as rationale, created_at as generated_at
			FROM trading_signals
			WHERE created_at >= NOW() - INTERVAL 1 HOUR
			AND model LIKE 'xgboost%'
			ORDER BY timestamp DESC
		""")
		for row in cursor.fetchall():
			signals.append(TradeSignal(
				symbol=row['symbol'],
				action=row['action'],
				confidence=row.get('confidence'),
				entry_price=row.get('entry_price'),
				stop_loss=row.get('stop_loss'),
				take_profit=row.get('take_profit'),
				position_size_percent=row.get('position_size_percent'),
				reasoning=row.get('rationale'),
				is_mock=False
			))
		cursor.close()
		cnx.close()
	except Exception as e:
		print(f"Error fetching trading signals: {e}")
	if not signals:
		signals = fetch_mock_signals()
	return signals

def fetch_mock_signals():
	coins = ["BTC", "ETH", "SOL"]
	actions = ["BUY", "SELL", "HOLD"]
	signals = []
	for symbol in coins:
		action = random.choice(actions)
		confidence = round(random.uniform(0.6, 0.99), 2) if action != "HOLD" else None
		entry_price = round(random.uniform(100, 70000), 2)
		stop_loss = entry_price * 0.97 if action == "BUY" else (entry_price * 1.03 if action == "SELL" else None)
		take_profit = entry_price * 1.05 if action == "BUY" else (entry_price * 0.95 if action == "SELL" else None)
		position_size_percent = random.choice([1.0, 2.0, 5.0]) if action != "HOLD" else None
		signals.append(TradeSignal(
			symbol=symbol,
			action=action,
			confidence=confidence,
			entry_price=entry_price,
			stop_loss=stop_loss,
			take_profit=take_profit,
			position_size_percent=position_size_percent,
			reasoning=f"Mock {action} signal for {symbol}",
			is_mock=True
		))
	return signals

def store_recommendation(signal: TradeSignal, status="pending", audit_note=None):
	cnx = mysql.connector.connect(**MYSQL_CONFIG)
	cursor = cnx.cursor()
	cursor.execute(
		"""
		INSERT INTO trade_recommendations (
			generated_at, symbol, action, confidence, entry_price, stop_loss, take_profit, position_size_percent, reasoning, is_mock, status
		) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
		""",
		(
			signal.generated_at or datetime.datetime.utcnow(),
			signal.symbol,
			signal.action,
			signal.confidence,
			signal.entry_price,
			signal.stop_loss,
			signal.take_profit,
			signal.position_size_percent,
			signal.reasoning,
			signal.is_mock,
			status
		)
	)
	rec_id = cursor.lastrowid
	cnx.commit()
	if audit_note:
		cursor.execute(
			"INSERT INTO audit_log (event_time, event_type, details) VALUES (%s, %s, %s)",
			(datetime.datetime.utcnow(), "recommendation", audit_note)
		)
		cnx.commit()
	cursor.close()
	cnx.close()
	
	# Trigger mock trading engine if this is a high-confidence signal
	if signal.confidence and signal.confidence >= 0.75 and signal.action in ['BUY', 'SELL']:
		trigger_mock_trading_engine(rec_id)
	
	return rec_id

async def trigger_mock_trading_engine(rec_id: int):
	"""
	Trigger the mock trading engine to process a specific recommendation.
	"""
	try:
		async with httpx.AsyncClient(timeout=10.0) as client:
			# trade-execution-engine exposes /process_recommendation/{id}
			response = await client.post(
				f"{MOCK_TRADING_ENGINE_URL}/process_recommendation/{rec_id}"
			)
			if response.status_code == 200:
				print(f"Successfully triggered mock trading engine for recommendation {rec_id}")
			else:
				print(f"Failed to trigger mock trading engine for recommendation {rec_id}: {response.status_code}")
	except Exception as e:
		print(f"Error triggering mock trading engine for recommendation {rec_id}: {e}")


# --- Scheduler for periodic polling ---
class PollingConfig(BaseModel):
	interval_seconds: int

polling_interval = int(os.environ.get('POLLING_INTERVAL', 300))  # default 5 min
scheduler = BackgroundScheduler()
polling_lock = threading.Lock()

def get_active_recommendations():
	cnx = mysql.connector.connect(**MYSQL_CONFIG)
	cursor = cnx.cursor(dictionary=True)
	cursor.execute("SELECT * FROM trade_recommendations WHERE status='pending' ORDER BY generated_at DESC")
	results = cursor.fetchall()
	cursor.close()
	cnx.close()
	return results

def poll_and_store_signals():
	with polling_lock:
		signals = fetch_real_signals()
		min_conf = float(os.environ.get("MIN_CONFIDENCE", 0.7))
		now = datetime.datetime.utcnow()
		holdings = {h['symbol']: h for h in get_holdings(is_mock=False)}
		active_recs = get_active_recommendations()
		active_keys = {(r['symbol'], r['action']) for r in active_recs}
		for signal in signals:
			if signal.confidence is not None and signal.confidence < min_conf:
				continue
			sig_time = None
			if signal.generated_at:
				try:
					sig_time = datetime.datetime.fromisoformat(signal.generated_at)
				except Exception:
					sig_time = now
			else:
				sig_time = now
			if (now - sig_time).total_seconds() > 1800:
				continue
			if (signal.symbol, signal.action) in active_keys:
				continue
			holding = holdings.get(signal.symbol)
			if signal.action == "BUY" and holding and holding['quantity'] > 0:
				continue
			if signal.action == "SELL" and (not holding or holding['quantity'] <= 0):
				continue
			if not signal.position_size_percent:
				signal.position_size_percent = 1.0 if not holding else min(5.0, 100.0 - holding['quantity'])
			if signal.position_size_percent > 10.0:
				signal.position_size_percent = 10.0
			if not signal.reasoning:
				signal.reasoning = f"Signal confidence: {signal.confidence}, action: {signal.action}"
			if not signal.stop_loss and signal.entry_price:
				signal.stop_loss = round(signal.entry_price * 0.97, 2) if signal.action == "BUY" else round(signal.entry_price * 1.03, 2)
			if not signal.take_profit and signal.entry_price:
				signal.take_profit = round(signal.entry_price * 1.05, 2) if signal.action == "BUY" else round(signal.entry_price * 0.95, 2)
			audit_note = f"Generated recommendation for {signal.symbol} {signal.action} at {now.isoformat()} with conf {signal.confidence}"
			rec_id = store_recommendation(signal, status="pending", audit_note=audit_note)
			
			# Trigger mock trading engine for high-confidence signals
			if signal.confidence and signal.confidence >= 0.75 and signal.action in ['BUY', 'SELL']:
				try:
					import asyncio
					asyncio.create_task(trigger_mock_trading_engine(rec_id))
				except Exception as e:
					print(f"Error triggering trading engine for rec {rec_id}: {e}")
@app.post("/recommendations/trigger_trading/{rec_id}")
async def trigger_trading_for_recommendation(rec_id: int):
	"""
	Manually trigger the mock trading engine for a specific recommendation.
	"""
	try:
		await trigger_mock_trading_engine(rec_id)
		return {"status": "triggered", "rec_id": rec_id}
	except Exception as e:
		return {"status": "error", "rec_id": rec_id, "error": str(e)}

@app.post("/recommendations/process_pending")
async def process_pending_recommendations():
	"""
	Process all pending recommendations with the mock trading engine.
	"""
	active_recs = get_active_recommendations()
	processed = 0
	errors = 0
	
	for rec in active_recs:
		if rec['confidence'] and rec['confidence'] >= 0.75 and rec['action'] in ['BUY', 'SELL']:
			try:
				await trigger_mock_trading_engine(rec['id'])
				processed += 1
			except Exception as e:
				print(f"Error processing recommendation {rec['id']}: {e}")
				errors += 1
	
	return {
		"status": "completed",
		"processed": processed,
		"errors": errors,
		"total_pending": len(active_recs)
	}

# --- Endpoint: Get all recommendations (optionally filter by status) ---
@app.get("/recommendations", response_model=List[dict])
def get_recommendations(is_mock: bool = True, status: Optional[str] = None):
	cnx = mysql.connector.connect(**MYSQL_CONFIG)
	cursor = cnx.cursor(dictionary=True)
	if status:
		cursor.execute("SELECT * FROM trade_recommendations WHERE is_mock=%s AND status=%s ORDER BY generated_at DESC", (is_mock, status))
	else:
		cursor.execute("SELECT * FROM trade_recommendations WHERE is_mock=%s ORDER BY generated_at DESC", (is_mock,))
	results = cursor.fetchall()
	cursor.close()
	cnx.close()
	return results

# --- Endpoint: Get only active (pending) recommendations ---
@app.get("/recommendations/active", response_model=List[dict])
def api_get_active_recommendations():
	return get_active_recommendations()

# --- Endpoint: Mark recommendation as executed or rejected ---
class RecStatusUpdate(BaseModel):
	rec_id: int
	new_status: str  # executed, rejected, expired
	note: Optional[str] = None

@app.post("/recommendations/update_status")
def update_recommendation_status(update: RecStatusUpdate):
	cnx = mysql.connector.connect(**MYSQL_CONFIG)
	cursor = cnx.cursor()
	cursor.execute("UPDATE trade_recommendations SET status=%s WHERE id=%s", (update.new_status, update.rec_id))
	cnx.commit()
	if update.note:
		cursor.execute(
			"INSERT INTO audit_log (event_time, event_type, details) VALUES (%s, %s, %s)",
			(datetime.datetime.utcnow(), "rec_status_update", f"rec_id={update.rec_id}, status={update.new_status}, note={update.note}")
		)
		cnx.commit()
	cursor.close()
	cnx.close()
	return {"status": "updated"}

@app.on_event("startup")
def start_scheduler():
	scheduler.add_job(poll_and_store_signals, 'interval', seconds=polling_interval, id='signal_polling', replace_existing=True)
	scheduler.start()

@app.post("/monitor_signals")
def monitor_signals(background_tasks: BackgroundTasks):
	"""
	Trigger immediate polling and recommendation generation (async).
	"""
	background_tasks.add_task(poll_and_store_signals)
	return {"status": "recommendations_queued_async"}

@app.post("/set_polling_interval")
def set_polling_interval(cfg: PollingConfig):
	global polling_interval
	polling_interval = cfg.interval_seconds
	scheduler.reschedule_job('signal_polling', trigger='interval', seconds=polling_interval)
	return {"status": "polling_interval_updated", "interval_seconds": polling_interval}

@app.get("/polling_status")
def polling_status():
	job = scheduler.get_job('signal_polling')
	return {
		"interval_seconds": polling_interval,
		"next_run_time": str(job.next_run_time) if job else None,
		"running": job is not None
	}

if __name__ == "__main__":
	import uvicorn
	print("Starting Trade Recommendation Service...")
	print(f"Service will connect to trading engine at: {MOCK_TRADING_ENGINE_URL}")
	uvicorn.run(app, host="0.0.0.0", port=8022, log_level="info")
