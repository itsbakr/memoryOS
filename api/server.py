import json
import os
import time
import uuid

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agent.loop import run_agent
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


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    with open("dashboard/index.html") as f:
        return f.read()


@app.post("/api/sessions")
async def create_session():
    """Create a new chat session."""
    session_id = str(uuid.uuid4())
    agent_id = str(uuid.uuid4())
    created_at = time.time()
    
    r = await aioredis.from_url(REDIS_URL, decode_responses=True)
    await r.hset(
        f"session:{session_id}",
        mapping={
            "agent_id": agent_id,
            "title": "New Chat",
            "previous_response_id": "",
            "created_at": str(created_at)
        }
    )
    await r.lpush("sessions:list", session_id)
    await r.aclose()
    
    return {"session_id": session_id, "agent_id": agent_id, "created_at": created_at}


@app.get("/api/sessions")
async def list_sessions():
    """List all available sessions."""
    r = await aioredis.from_url(REDIS_URL, decode_responses=True)
    session_ids = await r.lrange("sessions:list", 0, -1)
    
    sessions = []
    for sid in session_ids:
        data = await r.hgetall(f"session:{sid}")
        if data:
            sessions.append({
                "session_id": sid,
                "agent_id": data.get("agent_id"),
                "title": data.get("title", "New Chat"),
                "created_at": float(data.get("created_at", 0))
            })
            
    await r.aclose()
    return {"sessions": sessions}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Process a single chat turn within a session."""
    r = await aioredis.from_url(REDIS_URL, decode_responses=True)
    session_data = await r.hgetall(f"session:{request.session_id}")
    
    if not session_data:
        await r.aclose()
        raise HTTPException(status_code=404, detail="Session not found")
        
    agent_id = session_data["agent_id"]
    previous_response_id = session_data.get("previous_response_id", "") or None
    
    # Update title if it's the first message
    if session_data.get("title") == "New Chat":
        new_title = request.message[:40] + ("..." if len(request.message) > 40 else "")
        await r.hset(f"session:{request.session_id}", "title", new_title)
        
    # Run the agent
    try:
        reply_text, new_response_id, contradiction = await run_agent(
            agent_id=agent_id,
            user_message=request.message,
            previous_response_id=previous_response_id
        )
    except Exception as e:
        await r.aclose()
        raise HTTPException(status_code=500, detail=str(e))
        
    # Save the new response ID for next turn
    if new_response_id:
        await r.hset(f"session:{request.session_id}", "previous_response_id", new_response_id)
        
    await r.aclose()
    
    return {
        "reply": reply_text,
        "response_id": new_response_id,
        "contradiction": contradiction
    }


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
