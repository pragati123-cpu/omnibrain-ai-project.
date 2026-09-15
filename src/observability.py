"""
observability.py
-----------------
Week 3, Task 2: API & Backend Latency Tracking.

Provides Langfuse tracing plus LOCAL latency aggregation for
OmniBrain's backend, so "optimize backend performance metrics" has
something concrete to look at even before/without a configured
Langfuse project.

Every call to a @traced function does two things:

  1. Langfuse tracing via the official @observe decorator (SDK v4,
     OpenTelemetry-based). Verified directly against the installed
     SDK that this degrades gracefully -- no exception -- when
     LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY aren't set; it just logs
     a warning and disables the client. Safe to leave wired in
     everywhere even before the team has real Langfuse credentials.

  2. Local latency recording (duration, success/failure) into an
     in-memory store, aggregated into count/mean/p50/p95/max per
     operation name. Zero external dependencies, zero configuration
     needed -- useful for a quick "what's actually slow" check
     without opening a dashboard, and works even offline.

Works on both sync and async functions, since a future FastAPI layer
handling "asynchronous document queries" will likely use async
endpoints even though today's agent functions are sync.

Environment variables (all optional -- tracing no-ops without them):
  LANGFUSE_PUBLIC_KEY
  LANGFUSE_SECRET_KEY
  LANGFUSE_HOST          (defaults to Langfuse Cloud if unset)
"""

import functools
import inspect
import statistics
import threading
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional

from langfuse import observe

# --- Local latency store --------------------------------------------

_lock = threading.Lock()
_latency_records: Dict[str, List[dict]] = defaultdict(list)

MAX_RECORDS_PER_OPERATION = 500  # cap memory growth for long-running sessions


def _record_latency(operation: str, duration_ms: float, success: bool):
    with _lock:
        records = _latency_records[operation]
        records.append({"duration_ms": duration_ms, "success": success, "ts": time.time()})
        if len(records) > MAX_RECORDS_PER_OPERATION:
            del records[: len(records) - MAX_RECORDS_PER_OPERATION]


def _percentile(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (pct / 100)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def get_stats(operation: Optional[str] = None) -> dict:
    """
    Returns aggregated latency stats. If `operation` is given, returns
    stats for just that operation; otherwise returns a dict of stats
    for every traced operation seen so far.
    """
    with _lock:
        ops = [operation] if operation else list(_latency_records.keys())
        result = {}
        for op in ops:
            records = _latency_records.get(op, [])
            if not records:
                result[op] = {"count": 0}
                continue
            durations = sorted(r["duration_ms"] for r in records)
            successes = sum(1 for r in records if r["success"])
            result[op] = {
                "count": len(durations),
                "success_count": successes,
                "error_count": len(durations) - successes,
                "mean_ms": round(statistics.mean(durations), 1),
                "p50_ms": round(_percentile(durations, 50), 1),
                "p95_ms": round(_percentile(durations, 95), 1),
                "min_ms": round(min(durations), 1),
                "max_ms": round(max(durations), 1),
            }
        return result if operation is None else result[operation]


def reset_stats(operation: Optional[str] = None):
    """Clears recorded latency data. Mainly useful for tests."""
    with _lock:
        if operation:
            _latency_records.pop(operation, None)
        else:
            _latency_records.clear()


# --- The traced decorator ---------------------------------------------

def traced(name: str, as_type: Optional[str] = None):
    """
    Decorator: wraps a function with BOTH Langfuse tracing and local
    latency recording under `name`. Works on sync and async functions.
    Re-raises any exception from the wrapped function after recording
    it as a failed call -- tracing must never swallow real errors.

    as_type: passed through to Langfuse's @observe (e.g. "generation"
    for LLM calls, "retriever" for vector search, "tool" for
    structured actions like SQL execution) -- purely cosmetic for how
    the trace renders in the Langfuse dashboard; local latency
    tracking doesn't use it.

    Usage:
        @traced("search_agent", as_type="retriever")
        def search_agent(query: str, top_k: int = 3) -> dict:
            ...
    """
    def decorator(func: Callable) -> Callable:
        langfuse_wrapped = observe(name=name, as_type=as_type)(func)

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.perf_counter()
                success = True
                try:
                    return await langfuse_wrapped(*args, **kwargs)
                except Exception:
                    success = False
                    raise
                finally:
                    duration_ms = (time.perf_counter() - start) * 1000
                    _record_latency(name, duration_ms, success)

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.perf_counter()
                success = True
                try:
                    return langfuse_wrapped(*args, **kwargs)
                except Exception:
                    success = False
                    raise
                finally:
                    duration_ms = (time.perf_counter() - start) * 1000
                    _record_latency(name, duration_ms, success)

            return sync_wrapper

    return decorator
