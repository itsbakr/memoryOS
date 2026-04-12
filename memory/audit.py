from __future__ import annotations

import json
import time

from .working import get_redis


def _audit_key(agent_id: str) -> str:
    return f"agent:{agent_id}:audit"


def _metrics_key(agent_id: str) -> str:
    return f"agent:{agent_id}:metrics"


async def log_event(agent_id: str, event_type: str, payload: dict) -> None:
    r = await get_redis()
    event = {"type": event_type, "timestamp": time.time(), **payload}
    await r.lpush(_audit_key(agent_id), json.dumps(event))
    await r.ltrim(_audit_key(agent_id), 0, 199)
    await r.hincrby(_metrics_key(agent_id), event_type, 1)


async def fetch_events(agent_id: str, limit: int = 50) -> list[dict]:
    r = await get_redis()
    raw = await r.lrange(_audit_key(agent_id), 0, limit - 1)
    events = []
    for item in raw:
        try:
            events.append(json.loads(item))
        except json.JSONDecodeError:
            continue
    return events


async def fetch_metrics(agent_id: str) -> dict:
    r = await get_redis()
    raw = await r.hgetall(_metrics_key(agent_id))
    return {k: int(v) for k, v in raw.items()}
