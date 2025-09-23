from fastapi import FastAPI, BackgroundTasks, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import mysql.connector
import datetime
import os
import time

try:
	from ..shared.config import get_db_configs, get_service_urls, feature_flags
	from ..shared.metrics import inc, timing, snapshot, to_prometheus
	from ..shared.auth import api_key_required
	from ..shared.coinbase_asset_filter import is_asset_supported, get_trade_size_minimum, should_trade_amount, log_filtered_symbol
except ImportError:  # direct run fallback
	from backend.services.trading.shared.config import get_db_configs, get_service_urls, feature_flags  # type: ignore
	from backend.services.trading.shared.metrics import inc, timing, snapshot, to_prometheus  # type: ignore
	from backend.services.trading.shared.auth import api_key_required  # type: ignore
	try:
		from backend.services.trading.shared.coinbase_asset_filter import is_asset_supported, get_trade_size_minimum, should_trade_amount, log_filtered_symbol  # type: ignore
	except ImportError:
		# Fallback functions if asset filter not available
		def is_asset_supported(symbol): return symbol not in ['RNDR', 'RENDER']
		def get_trade_size_minimum(): return 5.0
		def should_trade_amount(amount): return amount >= 5.0
		def log_filtered_symbol(symbol, reason=""): return f"[FILTER] Excluding {symbol}"

app = FastAPI(title="Trade Recommendation Service")

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_CFGS = get_db_configs()
MYSQL_CONFIG = DB_CFGS['transactions']
FLAGS = feature_flags()
SERVICE_URLS = get_service_urls()

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
	return {"status": "healthy", "metrics": snapshot()}

@app.get("/metrics")
def metrics():
    return Response(content=to_prometheus(), media_type="text/plain; version=0.0.4")

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
	# ASSET FILTER: Check if asset is supported before creating recommendation
	if not is_asset_supported(signal.symbol):
		print(log_filtered_symbol(signal.symbol, "not supported on Coinbase Advanced Trade API"))
		return {"status": "filtered_out", "reason": "Asset not supported on Coinbase Advanced Trade API"}
	
	async def insert_recommendation():
		cnx = mysql.connector.connect(**MYSQL_CONFIG)
		cursor = cnx.cursor()
		start = time.time()
		cursor.execute(
			"""
			INSERT INTO trade_recommendations (
				generated_at, symbol, action, confidence, entry_price, stop_loss, take_profit, position_size_percent, reasoning, is_mock, status, execution_status
			) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending', 'PENDING')
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
		timing("recommendation.insert", time.time() - start)
		inc("recommendations.created")
		
		# Trigger mock trading engine if this is a high-confidence signal
		if signal.confidence and signal.confidence >= 0.75 and signal.action in ['BUY', 'SELL']:
			await trigger_mock_trading_engine(rec_id)
			
		return rec_id
	
	background_tasks.add_task(insert_recommendation)
	return {"status": "recommendation_queued"}

"""Removed earlier duplicate /recommendations endpoint; consolidated version with optional status filter appears later in file."""


# --- Signal Monitoring and Automated Recommendation Logic ---

import random
import requests
import httpx
import threading
try:
	from apscheduler.schedulers.background import BackgroundScheduler  # type: ignore
except ImportError:  # Fallback stub for test environments without apscheduler installed
	class BackgroundScheduler:  # minimal no-op scheduler
		def __init__(self,*a,**k): pass
		def add_job(self, *a, **k): pass
		def start(self): pass
		def get_job(self, job_id): return None
		def reschedule_job(self, *a, **k): pass

# Unified Trading Engine (mock/live via EXECUTION_MODE)
# Kept env var name for backward compatibility
MOCK_TRADING_ENGINE_URL = os.environ.get('MOCK_TRADING_ENGINE_URL', 'http://host.docker.internal:8024')



# --- Fetch signals from crypto_market_data.trading_signals table ---
def fetch_real_signals():
	"""
	Fetch signals from the crypto_market_data.trading_signals table in MySQL.
	"""
	signals = []
	try:
		# Use a separate connection config for crypto_prices DB
		db_cfg = MYSQL_CONFIG.copy()
		db_cfg['database'] = 'crypto_prices'
		cnx = mysql.connector.connect(**db_cfg)
		cursor = cnx.cursor(dictionary=True)
		cursor.execute("""
			SELECT symbol, signal_type as action, confidence, price as entry_price, NULL as stop_loss, NULL as take_profit, NULL as position_size_percent, regime as rationale, timestamp as generated_at
			FROM trading_signals
			WHERE timestamp >= NOW() - INTERVAL 1 HOUR
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
	if not signals and FLAGS.get("use_signals_service", False):
		# Attempt to retrieve from external signals microservice
		try:
			import httpx
			with httpx.Client(timeout=5.0) as client:
				r = client.get(f"{SERVICE_URLS.get('signals', '')}/signals/recent?limit=25")
				if r.status_code == 200:
					for row in r.json():
						try:
							signals.append(TradeSignal(**row))
						except Exception:
							continue
		except Exception:
			pass
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
def get_recommendations(
	is_mock: bool = True, 
	status: Optional[str] = None, 
	generated_after: Optional[str] = None,
	limit: Optional[int] = None,
	min_confidence: Optional[float] = None
):
	"""Retrieve recommendations.

	Status field values currently used:
	- 'PENDING' (awaiting execution) - uses execution_status field
	- 'EXECUTED' (trade executed) - uses execution_status field
	- 'pending' (legacy status field)
	- 'executed' (legacy status field)
	
	Args:
		is_mock: Filter by mock/live mode
		status: Filter by recommendation execution_status (PENDING/EXECUTED) or legacy status
		generated_after: Only return recommendations generated after this timestamp (YYYY-MM-DD HH:MM:SS)
		limit: Maximum number of recommendations to return
		min_confidence: Minimum confidence threshold (excludes NULL confidence records)
	
	Schema extension also defines execution_status columns (EXECUTED / PENDING etc.). Future refactor should unify these.
	"""
	cnx = mysql.connector.connect(**MYSQL_CONFIG)
	cursor = cnx.cursor(dictionary=True)
	
	# Build query with optional filters
	where_conditions = ["is_mock=%s"]
	params = [is_mock]
	
	# Handle both execution_status (PENDING/EXECUTED) and legacy status (pending/executed)
	if status:
		if status.upper() in ['PENDING', 'EXECUTED', 'REJECTED', 'EXPIRED']:
			where_conditions.append("execution_status=%s")
			params.append(status.upper())
		else:
			where_conditions.append("status=%s")
			params.append(status.lower())
	
	if generated_after:
		where_conditions.append("generated_at >= %s")
		params.append(generated_after)
	
	# CRITICAL FIX: Add confidence filtering to exclude NULL confidence records
	if min_confidence is not None:
		where_conditions.append("confidence IS NOT NULL AND confidence >= %s")
		params.append(min_confidence)
	
	query = f"SELECT * FROM trade_recommendations WHERE {' AND '.join(where_conditions)} ORDER BY generated_at DESC"
	
	if limit:
		query += f" LIMIT {int(limit)}"
	
	cursor.execute(query, params)
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
	status_map = {
		"executed": "EXECUTED",
		"rejected": "REJECTED",
		"failed": "REJECTED",
		"expired": "EXPIRED",
		"pending": "PENDING"
	}
	exec_status = status_map.get(update.new_status.lower())
	if not exec_status:
		return {"status": "error", "error": "invalid status"}
	cnx = mysql.connector.connect(**MYSQL_CONFIG)
	cursor = cnx.cursor()
	cursor.execute("UPDATE trade_recommendations SET status=%s, execution_status=%s WHERE id=%s", (update.new_status, exec_status, update.rec_id))
	cnx.commit()
	if update.note:
		cursor.execute(
			"INSERT INTO audit_log (event_time, event_type, details) VALUES (%s, %s, %s)",
			(datetime.datetime.utcnow(), "rec_status_update", f"rec_id={update.rec_id}, status={update.new_status}, exec_status={exec_status}, note={update.note}")
		)
		cnx.commit()
	cursor.close()
	cnx.close()
	return {"status": "updated", "execution_status": exec_status}

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
