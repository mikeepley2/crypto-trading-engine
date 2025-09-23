#!/usr/bin/env python3
"""Signals Service
Port: 8028
Provides recent processed signals abstraction (placeholder for future aggregation logic).
"""
from __future__ import annotations
from fastapi import FastAPI, Depends, Response
from pydantic import BaseModel
from typing import List
import datetime
import mysql.connector

try:
    from ..shared.config import get_db_configs, feature_flags
    from ..shared.metrics import inc, snapshot, to_prometheus
    from ..shared.auth import api_key_required
except ImportError:
    from backend.services.trading.shared.config import get_db_configs, feature_flags  # type: ignore
    from backend.services.trading.shared.metrics import inc, snapshot, to_prometheus  # type: ignore
    from backend.services.trading.shared.auth import api_key_required  # type: ignore

app = FastAPI(title="Signals Service", version="0.1.0")
DB_CFGS = get_db_configs()
PRICES_DB = DB_CFGS['prices']

class Signal(BaseModel):
    symbol: str
    action: str
    confidence: float | None = None
    source: str | None = None
    generated_at: str

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "signals", "metrics": snapshot()}

@app.get("/metrics")
def metrics():
    return Response(content=to_prometheus(), media_type="text/plain; version=0.0.4")

@app.get("/signals/recent", response_model=List[Signal])
async def recent_signals(limit: int = 20):
    try:
        cnx = mysql.connector.connect(**PRICES_DB)
        cursor = cnx.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT symbol, signal_type as action, confidence, model as source, DATE_FORMAT(timestamp, '%Y-%m-%dT%H:%i:%sZ') as generated_at
            FROM trading_signals
            ORDER BY timestamp DESC
            LIMIT %s
            """, (limit,)
        )
        rows = cursor.fetchall()
        cursor.close()
        cnx.close()
        inc("signals.fetched")
        return [Signal(**row) for row in rows]
    except Exception as e:
        # Return empty list if table doesn't exist or connection fails
        inc("signals.errors")
        return []

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8028)
