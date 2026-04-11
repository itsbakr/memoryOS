import json
import os
import time
import uuid
from typing import Optional

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agent.loop import run_agent
from memory.categories import (
    PROFILE_CATEGORIES,
    PROJECT_CATEGORIES,
    WORKFLOW_CATEGORIES,
)
from memory.contradiction import resolve_contradiction
from memory.decay import calculate_current_confidence
from memory.episodic import add_memory, retrieve_memories
from memory.extractor import extract_facts
from memory.models import MemoryEntry
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


class IngestRequest(BaseModel):
    agent_id: str
    transcript: str  # full conversation text to extract memories from
    task_context: str = "unknown"


class StoreRequest(BaseModel):
    agent_id: str
    content: str
    category: str = "general"
    source: str = "user_said"


import os
from fastapi.staticfiles import StaticFiles

from memory.working import get_redis

@app.post("/api/sessions")
async def create_session():
    """Create a new chat session."""
    session_id = str(uuid.uuid4())
    # Use a persistent agent_id so memory is shared across all chats for this user!
    agent_id = DEFAULT_AGENT
    created_at = time.time()
    
    r = await get_redis()
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
    
    return {"session_id": session_id, "agent_id": agent_id, "created_at": created_at}


@app.get("/api/sessions")
async def list_sessions():
    """List all available sessions."""
    r = await get_redis()
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
            
    return {"sessions": sessions}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Process a single chat turn within a session."""
    r = await get_redis()
    session_data = await r.hgetall(f"session:{request.session_id}")
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
        
    agent_id = session_data["agent_id"]
    previous_response_id = session_data.get("previous_response_id", "") or None
    
    # Update title if it's the first message
    if session_data.get("title") == "New Chat":
        new_title = request.message[:40] + ("..." if len(request.message) > 40 else "")
        await r.hset(f"session:{request.session_id}", "title", new_title)
        
    # Store user message in history
    await r.rpush(f"session_history:{request.session_id}", json.dumps({"role": "user", "content": request.message}))

    # Run the agent
    try:
        reply_text, new_response_id, contradiction = await run_agent(
            agent_id=agent_id,
            user_message=request.message,
            previous_response_id=previous_response_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    # Save the new response ID for next turn
    if new_response_id:
        await r.hset(f"session:{request.session_id}", "previous_response_id", new_response_id)
        
    # Store agent reply in history
    await r.rpush(f"session_history:{request.session_id}", json.dumps({"role": "agent", "content": reply_text}))

    return {
        "reply": reply_text,
        "response_id": new_response_id,
        "contradiction": contradiction
    }


@app.get("/api/chat/history")
async def chat_history(session_id: str):
    """Get the message history for a session."""
    r = await get_redis()
    raw_history = await r.lrange(f"session_history:{session_id}", 0, -1)
    
    history = []
    for item in raw_history:
        history.append(json.loads(item))
        
    return {"history": history}


@app.get("/api/memory/stats")
async def memory_stats(agent_id: str = DEFAULT_AGENT):
    """Returns memory stats for the dashboard."""
    from memory.episodic import count_memories
    total_memories_count = await count_memories(agent_id)
    
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
        "total_memories": total_memories_count,
        "confidence_distribution": bins,
        "working_memory": working.model_dump() if working else None,
        "memories": [
            {
                "content": m.content[:60] + "..." if len(m.content) > 60 else m.content,
                "confidence": round(calculate_current_confidence(m), 2),
                "source": m.source,
                "category": m.category or "general",
                "age_hours": round((time.time() - m.created_at) / 3600, 1),
            }
            for m in sorted(
                memories, key=lambda x: calculate_current_confidence(x), reverse=True
            )
        ][:20],
    }


@app.post("/api/memory/store")
async def memory_store(request: StoreRequest):
    """
    Directly store a single pre-formed memory fact (used by the MCP remember tool).
    Bypasses LLM extraction — the content is stored exactly as provided.
    Deduplication still applies.
    """
    from memory.categories import decay_rate_for, layer_for
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="content is required")
    layer = layer_for(request.category)
    if layer == "working":
        layer = "episodic"
    mem = MemoryEntry(
        agent_id=request.agent_id,
        content=request.content.strip(),
        layer=layer,
        source=request.source,
        category=request.category,
        decay_rate=decay_rate_for(request.category),
    )
    memory_id = await add_memory(mem)
    return {"status": "ok", "memory_id": memory_id, "category": request.category}


@app.get("/api/memory/search")
async def memory_search(query: str, agent_id: str = DEFAULT_AGENT, limit: int = 8):
    """
    Semantic vector search over episodic memories.
    Used by the MCP recall tool for meaningful context retrieval.
    """
    limit = min(limit, 20)
    memories = await retrieve_memories(query, agent_id, k=limit, min_confidence=0.05)
    return {
        "query": query,
        "results": [
            {
                "content": m.content,
                "category": m.category or "general",
                "source": m.source,
                "confidence": round(calculate_current_confidence(m), 3),
                "age_hours": round((time.time() - m.created_at) / 3600, 1),
            }
            for m in memories
        ],
    }


@app.get("/api/contradictions")
async def get_contradictions(agent_id: str = DEFAULT_AGENT):
    r = await get_redis()
    ids = await r.lrange(f"agent:{agent_id}:contradictions", 0, 19)
    events = []
    for cid in ids:
        raw = await r.get(f"contradiction:{cid}")
        if raw:
            events.append(json.loads(raw))
    return {"contradictions": events}


@app.post("/api/contradictions/{event_id}/resolve")
async def resolve(event_id: str, chosen_fact: str, agent_id: str = DEFAULT_AGENT):
    await resolve_contradiction(event_id, chosen_fact, agent_id)
    return {"status": "resolved"}


# ---------------------------------------------------------------------------
# Claude Code integration endpoints
# ---------------------------------------------------------------------------

@app.get("/api/context/snapshot")
async def context_snapshot(agent_id: str = DEFAULT_AGENT):
    """
    Returns a structured memory snapshot grouped by developer category.
    Used by mcp_server.py and scripts/sync_claude_md.py to build
    the CLAUDE.md context file and respond to MCP tool calls.
    """
    all_memories = await retrieve_memories(
        "everything", agent_id, k=200, min_confidence=0.0
    )
    working = await get_working_memory(agent_id)

    def _bucket(categories: set) -> list[dict]:
        items = []
        for m in all_memories:
            cat = m.category or "general"
            if cat not in categories:
                continue
            live_conf = calculate_current_confidence(m)
            if live_conf < 0.05:
                continue
            items.append({
                "id": m.id,
                "content": m.content,
                "category": cat,
                "source": m.source,
                "confidence": round(live_conf, 3),
                "age_hours": round((time.time() - m.created_at) / 3600, 1),
            })
        # Sort by confidence descending
        items.sort(key=lambda x: x["confidence"], reverse=True)
        return items

    return {
        "agent_id": agent_id,
        "profile": _bucket(PROFILE_CATEGORIES),
        "project": _bucket(PROJECT_CATEGORIES),
        "workflow": _bucket(WORKFLOW_CATEGORIES),
        "working_memory": working.model_dump() if working else None,
        "generated_at": time.time(),
    }


@app.post("/api/context/ingest")
async def context_ingest(request: IngestRequest):
    """
    Bulk-extract and store memories from a conversation transcript.
    Called by the Cursor post-conversation hook after each session ends.
    """
    facts = await extract_facts(request.transcript, request.task_context, request.agent_id)
    stored_ids = []
    for fact in facts:
        memory_id = await add_memory(fact)
        stored_ids.append(memory_id)
    return {
        "status": "ok",
        "stored": len(stored_ids),
        "memory_ids": stored_ids,
    }

# Setup React Frontend distribution serving
if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
