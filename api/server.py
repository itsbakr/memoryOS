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
from memory.audit import fetch_events, fetch_metrics
from memory.categories import (
    PROFILE_CATEGORIES,
    PROJECT_CATEGORIES,
    WORKFLOW_CATEGORIES,
)
from memory.contradiction import resolve_contradiction
from memory.decay import calculate_current_confidence
from memory.episodic import get_memory_by_id, list_memory_ids, retrieve_memories, update_memory_fields
from memory.graph import graph_stats
from memory.retrieval import hybrid_retrieve
from memory.extractor import extract_facts
from memory.models import MemoryEntry, WorkingMemory
from memory.working import get_working_memory, set_working_memory
from memory.write_gate import write_memory_entries

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


class WorkingRequest(BaseModel):
    agent_id: str
    task: str
    progress_pct: float = 0.0
    last_action: str = ""
    subtask: Optional[str] = None


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
        reply_text, new_response_id, contradiction, provenance = await run_agent(
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
    await r.rpush(
        f"session_history:{request.session_id}",
        json.dumps({"role": "agent", "content": reply_text, "provenance": provenance}),
    )

    return {
        "reply": reply_text,
        "response_id": new_response_id,
        "contradiction": contradiction,
        "provenance": provenance,
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
        active_only=False,
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
                "is_active": m.is_active,
                "version": m.version,
                "supersedes_id": m.supersedes_id,
                "superseded_by_id": m.superseded_by_id,
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
    results = await write_memory_entries([mem], conflict_policy="surface")
    result = results[0] if results else {"status": "skipped", "reason": "empty_content"}
    return result


@app.post("/api/memory/working")
async def memory_working(request: WorkingRequest):
    working = WorkingMemory(
        agent_id=request.agent_id,
        task=request.task,
        subtask=request.subtask,
        progress_pct=request.progress_pct,
        last_action=request.last_action,
    )
    await set_working_memory(working)
    return {"status": "ok"}


@app.get("/api/memory/search")
async def memory_search(query: str, agent_id: str = DEFAULT_AGENT, limit: int = 8):
    """
    Semantic vector search over episodic memories.
    Used by the MCP recall tool for meaningful context retrieval.
    """
    limit = min(limit, 20)
    memories, _provenance = await hybrid_retrieve(
        query, agent_id, k=limit, min_confidence=0.05
    )
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


@app.get("/api/memory/graph/stats")
async def graph_memory_stats(agent_id: str = DEFAULT_AGENT):
    return await graph_stats(agent_id)


@app.get("/api/memory/export")
async def memory_export(agent_id: str = DEFAULT_AGENT, include_inactive: bool = False):
    memory_ids = await list_memory_ids(agent_id, limit=2000)
    export_items = []
    for memory_id in memory_ids:
        mem = await get_memory_by_id(memory_id)
        if not mem:
            continue
        if not include_inactive and not mem.is_active:
            continue
        export_items.append(mem.model_dump())
    return {"agent_id": agent_id, "count": len(export_items), "memories": export_items}


@app.delete("/api/memory/{memory_id}")
async def memory_delete(memory_id: str, agent_id: str = DEFAULT_AGENT):
    mem = await get_memory_by_id(memory_id)
    if not mem or mem.agent_id != agent_id:
        raise HTTPException(status_code=404, detail="Memory not found")
    now = time.time()
    await update_memory_fields(
        memory_id,
        {
            "is_active": 0,
            "valid_to": now,
            "confidence": 0.0,
        },
    )
    return {"status": "deleted", "memory_id": memory_id}


@app.get("/api/memory/audit")
async def memory_audit(agent_id: str = DEFAULT_AGENT, limit: int = 50):
    events = await fetch_events(agent_id, limit=min(limit, 200))
    return {"agent_id": agent_id, "events": events}


@app.get("/api/memory/metrics")
async def memory_metrics(agent_id: str = DEFAULT_AGENT):
    metrics = await fetch_metrics(agent_id)
    return {"agent_id": agent_id, "metrics": metrics}


@app.get("/api/health")
async def health_check():
    r = await get_redis()
    try:
        pong = await r.ping()
    except Exception:
        pong = False
    return {"status": "ok" if pong else "degraded", "redis": bool(pong), "timestamp": time.time()}


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
                "is_active": m.is_active,
                "version": m.version,
                "supersedes_id": m.supersedes_id,
                "superseded_by_id": m.superseded_by_id,
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
    results = await write_memory_entries(facts, conflict_policy="skip")
    stored_ids = [r["memory_id"] for r in results if r.get("status") == "stored"]
    return {
        "status": "ok",
        "stored": len(stored_ids),
        "memory_ids": stored_ids,
        "skipped": len([r for r in results if r.get("status") != "stored"]),
    }

# Setup React Frontend distribution serving
if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
