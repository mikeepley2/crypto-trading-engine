"""Lightweight in-memory metrics for trading services.
Not production-grade; replace with Prometheus / OpenTelemetry later.
"""
from __future__ import annotations
import threading
import time
from typing import Dict, Any

_lock = threading.Lock()
_counters: Dict[str, int] = {}
_timings: Dict[str, list] = {}

def inc(name: str, value: int = 1) -> None:
    with _lock:
        _counters[name] = _counters.get(name, 0) + value

def timing(name: str, duration_seconds: float) -> None:
    with _lock:
        bucket = _timings.setdefault(name, [])
        bucket.append(duration_seconds)
        # Keep only last 500 samples to bound memory
        if len(bucket) > 500:
            del bucket[:-500]

def snapshot() -> Dict[str, Any]:
    with _lock:
        return {
            "counters": dict(_counters),
            "timings": {k: {
                "count": len(v),
                "avg_ms": (sum(v) / len(v) * 1000) if v else 0.0,
                "p95_ms": _percentile_ms(v, 95),
            } for k, v in _timings.items()}
        }

def to_prometheus() -> str:
    """Render a minimal Prometheus text exposition format for current metrics."""
    lines = []
    snap = snapshot()
    for name, val in snap["counters"].items():
        metric_name = name.replace('.', '_')
        lines.append(f"# TYPE {metric_name} counter")
        lines.append(f"{metric_name} {val}")
    for name, stats in snap["timings"].items():
        base = name.replace('.', '_')
        lines.append(f"# TYPE {base}_avg_ms gauge")
        lines.append(f"{base}_avg_ms {stats['avg_ms']}")
        lines.append(f"# TYPE {base}_p95_ms gauge")
        lines.append(f"{base}_p95_ms {stats['p95_ms']}")
        lines.append(f"# TYPE {base}_count counter")
        lines.append(f"{base}_count {stats['count']}")
    return "\n".join(lines) + "\n"

def _percentile_ms(values, pct):
    if not values:
        return 0.0
    data = sorted(values)
    k = int(round((pct / 100.0) * (len(data) - 1)))
    return data[k] * 1000.0

__all__ = ["inc", "timing", "snapshot", "to_prometheus"]
