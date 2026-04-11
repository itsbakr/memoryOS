import json
import os
import time

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from memory.contradiction import resolve_contradiction
from memory.decay import calculate_current_confidence
from memory.episodic import retrieve_memories
from memory.working import get_working_memory

app = FastAPI(title="Temporal Memory OS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DEFAULT_AGENT = "demo-agent"


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    with open("dashboard/index.html") as f:
        return f.read()


@app.get("/api/memory/stats")
async def memory_stats(agent_id: str = DEFAULT_AGENT):
    """Returns memory stats for the dashboard."""
    memories = await retrieve_memories(
        "everything important",
        agent_id,
        k=50,
        min_confidence=0.0,  # get all including stale
    )

    working = await get_working_memory(agent_id)

    # Bin by confidence
    bins = {"high": 0, "medium": 0, "low": 0, "stale": 0}
    for m in memories:
        conf = calculate_current_confidence(m)
        if conf >= 0.7:
            bins["high"] += 1
        elif conf >= 0.4:
            bins["medium"] += 1
        elif conf >= 0.1:
            bins["low"] += 1
        else:
            bins["stale"] += 1

    return {
        "total_memories": len(memories),
        "confidence_distribution": bins,
        "working_memory": working.model_dump() if working else None,
        "memories": [
            {
                "content": m.content[:60] + "..." if len(m.content) > 60 else m.content,
                "confidence": round(calculate_current_confidence(m), 2),
                "source": m.source,
                "age_hours": round((time.time() - m.created_at) / 3600, 1),
            }
            for m in sorted(
                memories, key=lambda x: calculate_current_confidence(x), reverse=True
            )
        ][:20],
    }


@app.get("/api/contradictions")
async def get_contradictions(agent_id: str = DEFAULT_AGENT):
    r = await aioredis.from_url(REDIS_URL, decode_responses=True)
    ids = await r.lrange(f"agent:{agent_id}:contradictions", 0, 19)
    events = []
    for cid in ids:
        raw = await r.get(f"contradiction:{cid}")
        if raw:
            events.append(json.loads(raw))
    await r.aclose()
    return {"contradictions": events}


@app.post("/api/contradictions/{event_id}/resolve")
async def resolve(event_id: str, chosen_fact: str, agent_id: str = DEFAULT_AGENT):
    await resolve_contradiction(event_id, chosen_fact, agent_id)
    return {"status": "resolved"}
