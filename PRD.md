═══════════════════════════════════════════════════════════════
TEMPORAL MEMORY OS — FULL AGENT PRD
Hackathon: Wordware Beach House, April 11 2026
Stack: Redis + Blaxel + OpenAI Codex
═══════════════════════════════════════════════════════════════

GIVE THIS ENTIRE DOCUMENT TO YOUR CODING AGENT AS SYSTEM CONTEXT.
DO NOT SKIP ANY SECTION. COMPLETE TASKS IN ORDER.
```

---

## SECTION 0 — RESEARCH TASKS (DO BEFORE WRITING CODE)

The agent MUST complete these research tasks first by reading the linked
documentation, then synthesize the findings into constants/config at
the top of each relevant module.

### Research Task 1 — Blaxel SDK

Read https://github.com/blaxel-ai/sdk-python and
https://docs.blaxel.ai/Sandboxes/Overview

Learn and confirm:
- How `SandboxInstance.create_if_not_exists()` works and what params it takes
- How `sandbox.process.exec()` works (name, command, working_dir, wait_for_completion)
- How `sandbox.filesystem` works for reading/writing files (use this for checkpointing working memory to the sandbox filesystem itself, NOT just Redis)
- How standby/resume works — confirm the sandbox auto-scales to zero after inactivity and resumes from standby in <25ms with identical process + filesystem state
- How to get a public preview URL for the dashboard

Install: `pip install blaxel`

Auth: Set env vars `BL_WORKSPACE` and `BL_API_KEY` — do NOT hardcode.

Key code pattern to internalize:
```python
from blaxel.core import SandboxInstance

sandbox = await SandboxInstance.create_if_not_exists({
    "name": "temporal-memory-agent",
    "image": "blaxel/python:latest",
    "memory": 2048,
    "region": "us-pdx-1",
    "ttl": "24h"
})

# Write checkpoint file to sandbox filesystem
await sandbox.filesystem.write(
    "/memory/checkpoint.json",
    json.dumps(checkpoint_data)
)

# Read it back
content = await sandbox.filesystem.read("/memory/checkpoint.json")
```

### Research Task 2 — Redis Vector Search (RedisVL)

Read https://github.com/redis/redis-vl-python and
https://redis.io/docs/latest/develop/get-started/vector-database/
and https://redis.io/tutorials/what-is-agent-memory-example-using-langgraph-and-redis/

Learn and confirm:
- How to create a SearchIndex with a VectorField schema using RedisVL
- How VectorQuery works (embedding, field name, num_results, filter_expression)
- How HybridQuery works (Redis 8+ — combines BM25 text search + vector similarity)
- How to use TagField filters to scope queries by `agent_id` and `layer`
- How to use NumericField for filtering by `confidence` score (>= threshold)
- How Redis Streams work for ordered time-indexed episodic memory (`XADD`, `XRANGE`)
- How Redis Hash works for working memory (`HSET`, `HGETALL`, `EXPIRE`)

Install: `pip install redisvl redis`

Connect via `REDIS_URL` env var — do NOT hardcode credentials.

Key schema to internalize:
```python
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema

EPISODIC_SCHEMA = IndexSchema.from_dict({
    "index": {"name": "episodic_memory", "prefix": "mem:episodic"},
    "fields": [
        {"name": "agent_id", "type": "tag"},
        {"name": "content", "type": "text"},
        {"name": "source", "type": "tag"},  # user_said | agent_inferred | tool_result
        {"name": "confidence", "type": "numeric"},
        {"name": "decay_rate", "type": "numeric"},
        {"name": "created_at", "type": "numeric"},
        {"name": "last_reinforced", "type": "numeric"},
        {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "dims": 1536,
                "distance_metric": "cosine",
                "algorithm": "hnsw",
                "datatype": "float32"
            }
        }
    ]
})
```

### Research Task 3 — Mem0 Architecture (to differentiate from, not copy)

Read https://arxiv.org/html/2504.19413v1 (skim the technical sections)
and https://docs.mem0.ai/open-source/features/graph-memory

Understand deeply:
- Mem0's two-phase pipeline: EXTRACTION (LLM extracts atomic facts from
  conversation) → UPDATE (LLM compares candidate facts against top-10
  similar existing memories, chooses ADD/UPDATE/DELETE/NOOP)
- Mem0's four operations and WHY they exist (coherence + non-redundancy)
- What Mem0 DOES NOT do:
  a) No temporal confidence decay — facts stored forever at full weight
  b) No explicit contradiction surfacing — resolves silently
  c) No session recovery — stateless between invocations
  d) No actor-aware memory scoping in base version
  e) No memory health observability

YOUR SYSTEM fills exactly these gaps. Make this framing explicit in
the README and in the demo script. Do NOT use Mem0 as a dependency —
you are building what Mem0 hasn't built yet.

### Research Task 4 — OpenAI Codex + Responses API

Read https://platform.openai.com/docs/guides/responses

Understand:
- How the Responses API works (vs Chat Completions) — use `client.responses.create()`
- How to pass tool definitions so the agent can call memory tools
- How to pass previous_response_id for multi-turn without re-sending full history
- Codex model string to use: `codex-mini-latest` (cheap, fast, good for agent loop)

The agent you're building will use Codex to reason over tasks. Memory
tools are injected as function tools. The agent calls them explicitly.

---

## SECTION 1 — PROJECT STRUCTURE

After research, create this exact directory structure:

```
temporal-memory-os/
├── main.py                  # Entry point: starts agent loop + dashboard
├── memory/
│   ├── __init__.py
│   ├── models.py            # Pydantic models for all memory types
│   ├── working.py           # Layer 1: Redis Hash working memory
│   ├── episodic.py          # Layer 2: Redis Streams + RedisVL episodic
│   ├── semantic.py          # Layer 3: RedisVL semantic compression
│   ├── decay.py             # Confidence decay engine
│   ├── extractor.py         # LLM-powered fact extraction (Mem0-inspired)
│   └── contradiction.py     # Contradiction detection + surfacing
├── sandbox/
│   ├── __init__.py
│   ├── manager.py           # Blaxel sandbox lifecycle
│   └── checkpoint.py        # Checkpoint/resume logic
├── agent/
│   ├── __init__.py
│   ├── loop.py              # Codex agent reasoning loop
│   └── tools.py             # Memory tools exposed to Codex
├── api/
│   ├── __init__.py
│   └── server.py            # FastAPI server for dashboard + endpoints
├── dashboard/
│   └── index.html           # Memory health dashboard (inline HTML/JS)
├── prompts/
│   ├── extraction.py        # Fact extraction prompt
│   └── contradiction.py     # Contradiction detection prompt
├── tests/
│   └── demo.py              # Full demo script for judges
├── requirements.txt
├── .env.example
└── README.md
```

---

## SECTION 2 — DATA MODELS (memory/models.py)

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional
from uuid import uuid4
import time

class MemoryEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    content: str                   # atomic extracted fact, max 200 chars
    layer: Literal["working", "episodic", "semantic"]
    confidence: float = 1.0        # 0.0-1.0, starts at 1.0
    source: Literal["user_said", "agent_inferred", "tool_result"]
    created_at: float = Field(default_factory=time.time)
    last_reinforced: float = Field(default_factory=time.time)
    # DECAY RATES (per hour):
    # user_preference: 0.001  (very slow — weeks to decay meaningfully)
    # task_context:    0.05   (hours — current task facts)
    # tool_result:     0.1    (fast — stale data decays quickly)
    decay_rate: float = 0.05
    embedding: Optional[list[float]] = None

class WorkingMemory(BaseModel):
    agent_id: str
    task: str
    subtask: Optional[str] = None
    progress_pct: float = 0.0
    last_action: Optional[str] = None
    last_checkpoint: float = Field(default_factory=time.time)
    tool_calls_made: int = 0
    context_summary: Optional[str] = None

class ContradictionEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    new_fact: str
    conflicting_memory_id: str
    conflicting_memory_content: str
    confidence_score: float        # how confident we are it's a contradiction
    explanation: str
    resolution: Literal["pending", "user_resolved", "auto_resolved"] = "pending"
    chosen_fact: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    resolved_at: Optional[float] = None

class CheckpointPayload(BaseModel):
    agent_id: str
    working: WorkingMemory
    recent_episodic: list[MemoryEntry]   # last 10 for quick resume context
    checkpoint_version: int
    created_at: float = Field(default_factory=time.time)
```

---

## SECTION 3 — CONFIDENCE DECAY ENGINE (memory/decay.py)

This is the core intellectual contribution vs Mem0. Implement carefully.

```python
import math
import time
from .models import MemoryEntry

DECAY_RATES = {
    "user_said": 0.001,       # user preferences — very slow
    "agent_inferred": 0.05,   # agent reasoning — moderate
    "tool_result": 0.1,       # tool outputs — fast (data goes stale)
}

def calculate_current_confidence(memory: MemoryEntry, now: float = None) -> float:
    """
    Exponential decay: conf(t) = initial_conf * e^(-decay_rate * hours_elapsed)
    
    A confidence of 0.3 means the memory is still used but weighted low.
    A confidence of 0.1 means it's effectively stale — filter it out.
    """
    if now is None:
        now = time.time()
    hours_elapsed = (now - memory.last_reinforced) / 3600
    decayed = memory.confidence * math.exp(-memory.decay_rate * hours_elapsed)
    return round(max(0.0, min(1.0, decayed)), 4)

def reinforce_memory(memory: MemoryEntry) -> MemoryEntry:
    """Called when a memory is retrieved and used — refreshes confidence."""
    memory.confidence = min(1.0, memory.confidence + 0.2)
    memory.last_reinforced = time.time()
    return memory

def should_prune(memory: MemoryEntry, threshold: float = 0.05) -> bool:
    """Returns True if memory should be removed from active retrieval."""
    return calculate_current_confidence(memory) < threshold
```

---

## SECTION 4 — FACT EXTRACTION (memory/extractor.py)

Inspired by Mem0's extraction phase but simpler — no graph.

```python
from openai import AsyncOpenAI
from .models import MemoryEntry, DECAY_RATES
import json

client = AsyncOpenAI()

EXTRACTION_PROMPT = """
You are a memory extraction system for an AI agent.
Given a conversation turn, extract ATOMIC FACTS worth remembering.

Rules:
- Each fact must be a single, standalone statement (max 100 chars)
- Only extract facts that matter beyond this conversation turn
- Label each fact with source: user_said | agent_inferred | tool_result
- Ignore small talk, filler, and transient state
- Return JSON array only, no explanation

Output format:
[
  {"content": "User prefers morning meetings", "source": "user_said"},
  {"content": "Project deadline is Friday April 18", "source": "user_said"},
  {"content": "Task completed: research phase", "source": "agent_inferred"}
]

Conversation turn:
{conversation_turn}

Current task context:
{task_context}
"""

async def extract_facts(
    conversation_turn: str,
    task_context: str,
    agent_id: str
) -> list[MemoryEntry]:
    response = await client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{
            "role": "user",
            "content": EXTRACTION_PROMPT.format(
                conversation_turn=conversation_turn,
                task_context=task_context
            )
        }],
        response_format={"type": "json_object"}
    )
    
    try:
        raw = json.loads(response.choices[0].message.content)
        facts = raw if isinstance(raw, list) else raw.get("facts", [])
        return [
            MemoryEntry(
                agent_id=agent_id,
                content=f["content"],
                layer="episodic",
                source=f["source"],
                decay_rate=DECAY_RATES.get(f["source"], 0.05)
            )
            for f in facts
            if f.get("content")
        ]
    except Exception:
        return []
```

---

## SECTION 5 — WORKING MEMORY (memory/working.py)

```python
import redis.asyncio as aioredis
import json
import os
from .models import WorkingMemory

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
WORKING_MEMORY_TTL = 86400  # 24 hours

async def get_redis():
    return await aioredis.from_url(REDIS_URL, decode_responses=True)

def _key(agent_id: str) -> str:
    return f"agent:{agent_id}:working"

async def set_working_memory(working: WorkingMemory) -> None:
    r = await get_redis()
    await r.hset(_key(working.agent_id), mapping=working.model_dump())
    await r.expire(_key(working.agent_id), WORKING_MEMORY_TTL)
    await r.aclose()

async def get_working_memory(agent_id: str) -> WorkingMemory | None:
    r = await get_redis()
    data = await r.hgetall(_key(agent_id))
    await r.aclose()
    if not data:
        return None
    return WorkingMemory(**data)

async def increment_tool_calls(agent_id: str) -> None:
    r = await get_redis()
    await r.hincrby(_key(agent_id), "tool_calls_made", 1)
    await r.aclose()

async def clear_working_memory(agent_id: str) -> None:
    r = await get_redis()
    await r.delete(_key(agent_id))
    await r.aclose()
```

---

## SECTION 6 — EPISODIC MEMORY (memory/episodic.py)

Uses RedisVL for vector search. Read the RedisVL docs before implementing.

```python
import os
import time
import numpy as np
from openai import AsyncOpenAI
from redisvl.index import AsyncSearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag, Num
from .models import MemoryEntry, EPISODIC_SCHEMA
from .decay import calculate_current_confidence, reinforce_memory

openai_client = AsyncOpenAI()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

async def get_index() -> AsyncSearchIndex:
    index = AsyncSearchIndex(EPISODIC_SCHEMA, redis_url=REDIS_URL)
    await index.create(overwrite=False)
    return index

async def embed(text: str) -> list[float]:
    response = await openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

async def add_memory(memory: MemoryEntry) -> str:
    """Store a new episodic memory with embedding."""
    index = await get_index()
    embedding = await embed(memory.content)
    memory.embedding = embedding
    
    doc = {
        "id": memory.id,
        "agent_id": memory.agent_id,
        "content": memory.content,
        "source": memory.source,
        "confidence": memory.confidence,
        "decay_rate": memory.decay_rate,
        "created_at": memory.created_at,
        "last_reinforced": memory.last_reinforced,
        "embedding": np.array(embedding, dtype=np.float32).tobytes()
    }
    await index.load([doc], id_field="id")
    return memory.id

async def retrieve_memories(
    query: str,
    agent_id: str,
    k: int = 10,
    min_confidence: float = 0.3
) -> list[MemoryEntry]:
    """
    Vector search filtered by agent_id and min_confidence.
    Applies decay before filtering — confidence is live, not stored.
    Returns memories ranked by relevance * current_confidence.
    """
    index = await get_index()
    embedding = await embed(query)
    
    agent_filter = Tag("agent_id") == agent_id
    
    query_obj = VectorQuery(
        vector=embedding,
        vector_field_name="embedding",
        filter_expression=agent_filter,
        num_results=k * 3,  # oversample, then filter by live confidence
        return_fields=["id", "content", "source", "confidence",
                       "decay_rate", "created_at", "last_reinforced",
                       "vector_distance"]
    )
    
    results = await index.query(query_obj)
    
    memories = []
    for r in results:
        mem = MemoryEntry(
            id=r["id"],
            agent_id=agent_id,
            content=r["content"],
            layer="episodic",
            source=r["source"],
            confidence=float(r["confidence"]),
            decay_rate=float(r["decay_rate"]),
            created_at=float(r["created_at"]),
            last_reinforced=float(r["last_reinforced"])
        )
        live_conf = calculate_current_confidence(mem)
        if live_conf >= min_confidence:
            relevance = 1 - float(r["vector_distance"])  # cosine: 0=identical
            mem.confidence = live_conf
            memories.append((mem, relevance * live_conf))
    
    # Sort by combined score, return top k
    memories.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in memories[:k]]

async def update_memory_confidence(memory_id: str, new_confidence: float) -> None:
    """Called after contradiction resolution or reinforcement."""
    index = await get_index()
    # RedisVL partial update — only update confidence fields
    await index.client.json().set(
        f"mem:episodic:{memory_id}",
        "$.confidence",
        new_confidence
    )
    await index.client.json().set(
        f"mem:episodic:{memory_id}",
        "$.last_reinforced",
        time.time()
    )
```

---

## SECTION 7 — CONTRADICTION DETECTION (memory/contradiction.py)

This is the DEMO MOMENT. Make it work reliably.

```python
import json
import os
from openai import AsyncOpenAI
from .models import MemoryEntry, ContradictionEvent
from .episodic import retrieve_memories

openai_client = AsyncOpenAI()

# Store pending contradictions in Redis for the dashboard
import redis.asyncio as aioredis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

CONTRADICTION_PROMPT = """
You are a contradiction detector for an AI agent's memory system.

New fact being considered for storage:
"{new_fact}"

Existing memories that might conflict:
{existing_memories}

Task: Determine if the new fact DIRECTLY CONTRADICTS any existing memory.
A contradiction means both CANNOT be true simultaneously.
(e.g., "meeting is Thursday" vs "meeting is Wednesday" = contradiction)
(e.g., "user likes Python" vs "user started learning Rust" = NOT contradiction)

Return JSON only:
{{
  "is_contradiction": true/false,
  "confidence": 0.0-1.0,
  "conflicting_memory_index": 0-N or null,
  "explanation": "brief explanation"
}}
"""

async def check_contradiction(
    new_fact: str,
    agent_id: str
) -> ContradictionEvent | None:
    """
    Returns a ContradictionEvent if detected, None otherwise.
    Threshold: 0.75 confidence to surface to user.
    """
    # Search including low-confidence memories (they could still conflict)
    similar = await retrieve_memories(new_fact, agent_id, k=5, min_confidence=0.0)
    
    if not similar:
        return None
    
    existing_str = "\n".join([
        f"{i}. [{m.source}] {m.content} (confidence: {m.confidence:.2f})"
        for i, m in enumerate(similar)
    ])
    
    response = await openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{
            "role": "user",
            "content": CONTRADICTION_PROMPT.format(
                new_fact=new_fact,
                existing_memories=existing_str
            )
        }],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    
    if not result.get("is_contradiction"):
        return None
    
    conf = float(result.get("confidence", 0))
    if conf < 0.75:
        return None
    
    idx = result.get("conflicting_memory_index")
    conflicting = similar[idx] if idx is not None and idx < len(similar) else similar[0]
    
    event = ContradictionEvent(
        agent_id=agent_id,
        new_fact=new_fact,
        conflicting_memory_id=conflicting.id,
        conflicting_memory_content=conflicting.content,
        confidence_score=conf,
        explanation=result.get("explanation", "")
    )
    
    # Store in Redis for dashboard + API resolution
    r = await aioredis.from_url(REDIS_URL, decode_responses=True)
    await r.setex(
        f"contradiction:{event.id}",
        3600,  # 1 hour TTL
        event.model_dump_json()
    )
    await r.lpush(f"agent:{agent_id}:contradictions", event.id)
    await r.aclose()
    
    return event

async def resolve_contradiction(
    event_id: str,
    chosen_fact: str,
    agent_id: str
) -> None:
    """User or auto-resolver picks which fact wins."""
    r = await aioredis.from_url(REDIS_URL, decode_responses=True)
    raw = await r.get(f"contradiction:{event_id}")
    if not raw:
        return
    
    event = ContradictionEvent.model_validate_json(raw)
    event.resolution = "user_resolved"
    event.chosen_fact = chosen_fact
    import time
    event.resolved_at = time.time()
    
    await r.set(f"contradiction:{event_id}", event.model_dump_json())
    await r.aclose()
```

---

## SECTION 8 — BLAXEL CHECKPOINT/RESUME (sandbox/checkpoint.py)

```python
import json
import os
import asyncio
from blaxel.core import SandboxInstance
from memory.models import CheckpointPayload, WorkingMemory, MemoryEntry
from memory.working import get_working_memory, set_working_memory
from memory.episodic import retrieve_memories

SANDBOX_NAME = os.getenv("SANDBOX_NAME", "temporal-memory-agent")
CHECKPOINT_PATH = "/memory/checkpoint.json"
CHECKPOINT_VERSION_KEY = "checkpoint_version"

async def get_or_create_sandbox() -> SandboxInstance:
    """
    Creates sandbox if not exists, otherwise gets existing.
    Blaxel resumes from standby in <25ms — no cold start penalty.
    """
    sandbox = await SandboxInstance.create_if_not_exists({
        "name": SANDBOX_NAME,
        "image": "blaxel/python:latest",
        "memory": 2048,
        "region": "us-pdx-1",
        "ttl": "48h"
    })
    return sandbox

async def checkpoint(agent_id: str) -> str:
    """
    Save full agent state to BOTH Redis (fast retrieval) and
    Blaxel sandbox filesystem (persistent, survives Redis flush).
    Returns checkpoint ID.
    """
    sandbox = await get_or_create_sandbox()
    working = await get_working_memory(agent_id)
    
    if not working:
        return None
    
    recent = await retrieve_memories(
        working.task or "current task",
        agent_id,
        k=10,
        min_confidence=0.2
    )
    
    # Get current version
    import redis.asyncio as aioredis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    r = await aioredis.from_url(REDIS_URL, decode_responses=True)
    version = int(await r.incr(f"agent:{agent_id}:{CHECKPOINT_VERSION_KEY}"))
    await r.aclose()
    
    payload = CheckpointPayload(
        agent_id=agent_id,
        working=working,
        recent_episodic=recent,
        checkpoint_version=version
    )
    
    # Write to Blaxel sandbox filesystem
    checkpoint_json = payload.model_dump_json(indent=2)
    await sandbox.filesystem.write(CHECKPOINT_PATH, checkpoint_json)
    
    print(f"[CHECKPOINT] v{version} saved to Blaxel sandbox + Redis")
    return str(version)

async def resume(agent_id: str) -> CheckpointPayload | None:
    """
    Load checkpoint from Blaxel sandbox filesystem.
    Called when agent starts and finds existing sandbox.
    """
    try:
        sandbox = await SandboxInstance.get(SANDBOX_NAME)
        raw = await sandbox.filesystem.read(CHECKPOINT_PATH)
        payload = CheckpointPayload.model_validate_json(raw)
        
        # Restore working memory to Redis
        await set_working_memory(payload.working)
        
        print(f"[RESUME] Restored checkpoint v{payload.checkpoint_version}")
        print(f"[RESUME] Task: {payload.working.task}")
        print(f"[RESUME] Progress: {payload.working.progress_pct}%")
        print(f"[RESUME] Last action: {payload.working.last_action}")
        
        return payload
    except Exception as e:
        print(f"[RESUME] No existing checkpoint found: {e}")
        return None

async def auto_checkpoint_loop(agent_id: str, interval_seconds: int = 30):
    """Run in background — checkpoints every N seconds."""
    while True:
        await asyncio.sleep(interval_seconds)
        await checkpoint(agent_id)
```

---

## SECTION 9 — MEMORY TOOLS FOR CODEX AGENT (agent/tools.py)

These are the function tools Codex calls during reasoning.

```python
from memory.episodic import retrieve_memories, add_memory
from memory.working import get_working_memory, set_working_memory
from memory.extractor import extract_facts
from memory.contradiction import check_contradiction, resolve_contradiction
from memory.models import WorkingMemory, MemoryEntry
import time

# Tool definitions in OpenAI function format
MEMORY_TOOLS = [
    {
        "type": "function",
        "name": "store_memory",
        "description": "Store a new fact in episodic memory. "
                       "Call this when the user says something important "
                       "or when you learn something worth remembering.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The atomic fact to store (max 150 chars)"
                },
                "source": {
                    "type": "string",
                    "enum": ["user_said", "agent_inferred", "tool_result"]
                }
            },
            "required": ["content", "source"]
        }
    },
    {
        "type": "function",
        "name": "retrieve_memory",
        "description": "Search episodic memory for relevant facts. "
                       "Call this before answering questions about past context.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What you're looking for"
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum confidence threshold (0.0-1.0). "
                                   "Default 0.3. Use 0.0 to see stale memories.",
                    "default": 0.3
                }
            },
            "required": ["query"]
        }
    },
    {
        "type": "function",
        "name": "update_task_progress",
        "description": "Update working memory with current task state. "
                       "Call after completing each subtask.",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "subtask": {"type": "string"},
                "progress_pct": {"type": "number"},
                "last_action": {"type": "string"}
            },
            "required": ["task", "progress_pct", "last_action"]
        }
    },
    {
        "type": "function",
        "name": "resolve_contradiction",
        "description": "Called when a contradiction has been surfaced and user has chosen.",
        "parameters": {
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
                "chosen_fact": {"type": "string"}
            },
            "required": ["event_id", "chosen_fact"]
        }
    }
]

async def handle_tool_call(
    tool_name: str,
    tool_args: dict,
    agent_id: str
) -> dict:
    """Dispatch tool calls from Codex to memory functions."""
    
    if tool_name == "store_memory":
        # FIRST: check for contradiction before storing
        contradiction = await check_contradiction(
            tool_args["content"],
            agent_id
        )
        if contradiction:
            return {
                "status": "contradiction_detected",
                "contradiction_id": contradiction.id,
                "new_fact": contradiction.new_fact,
                "conflicts_with": contradiction.conflicting_memory_content,
                "explanation": contradiction.explanation,
                "action_required": "Ask user which is correct before storing."
            }
        
        mem = MemoryEntry(
            agent_id=agent_id,
            content=tool_args["content"],
            layer="episodic",
            source=tool_args["source"]
        )
        mem_id = await add_memory(mem)
        return {"status": "stored", "memory_id": mem_id}
    
    elif tool_name == "retrieve_memory":
        memories = await retrieve_memories(
            query=tool_args["query"],
            agent_id=agent_id,
            min_confidence=tool_args.get("min_confidence", 0.3)
        )
        return {
            "memories": [
                {
                    "content": m.content,
                    "source": m.source,
                    "confidence": round(m.confidence, 2),
                    "age_hours": round(
                        (time.time() - m.created_at) / 3600, 1
                    )
                }
                for m in memories
            ],
            "count": len(memories)
        }
    
    elif tool_name == "update_task_progress":
        working = WorkingMemory(
            agent_id=agent_id,
            **tool_args
        )
        await set_working_memory(working)
        return {"status": "updated"}
    
    elif tool_name == "resolve_contradiction":
        await resolve_contradiction(
            tool_args["event_id"],
            tool_args["chosen_fact"],
            agent_id
        )
        return {"status": "resolved"}
    
    return {"error": f"Unknown tool: {tool_name}"}
```

---

## SECTION 10 — CODEX AGENT LOOP (agent/loop.py)

```python
import os
import json
import asyncio
from openai import AsyncOpenAI
from agent.tools import MEMORY_TOOLS, handle_tool_call
from memory.working import get_working_memory
from memory.episodic import retrieve_memories
from sandbox.checkpoint import checkpoint, auto_checkpoint_loop, resume

openai_client = AsyncOpenAI()

SYSTEM_PROMPT = """You are a long-running AI agent with persistent memory.

You have access to memory tools:
- store_memory: save important facts before you forget them
- retrieve_memory: search your memory for relevant context
- update_task_progress: track where you are in a task
- resolve_contradiction: resolve conflicting information

CRITICAL RULES:
1. ALWAYS call retrieve_memory at the start of each task to load relevant context
2. ALWAYS call store_memory when the user shares important information
3. ALWAYS call update_task_progress when you complete a subtask
4. If store_memory returns contradiction_detected, STOP and ask the user
   to clarify before proceeding. Show them both facts. Do NOT guess.
5. You persist across sessions. When you start, you may already have
   context from previous sessions — always check.

Be direct. Tell the user when you're remembering something, when you're
detecting a conflict, and when you're resuming from a previous session."""

async def run_agent(agent_id: str, user_message: str, previous_response_id: str = None):
    """
    Single turn of the Codex agent loop.
    Returns (response_text, response_id, contradiction_event_or_None)
    """
    
    # Inject relevant memory context into the message
    memories = await retrieve_memories(user_message, agent_id, k=5)
    memory_context = ""
    if memories:
        memory_lines = "\n".join([
            f"- [{m.source}, conf={m.confidence:.2f}] {m.content}"
            for m in memories
        ])
        memory_context = f"\n\n[MEMORY CONTEXT]\n{memory_lines}"
    
    working = await get_working_memory(agent_id)
    working_context = ""
    if working:
        working_context = f"\n\n[CURRENT TASK]\n{working.task} ({working.progress_pct}% complete)\nLast action: {working.last_action}"
    
    enriched_message = user_message + memory_context + working_context
    
    response = await openai_client.responses.create(
        model="codex-mini-latest",
        instructions=SYSTEM_PROMPT,
        input=enriched_message,
        tools=MEMORY_TOOLS,
        previous_response_id=previous_response_id
    )
    
    # Handle tool calls
    contradiction_event = None
    while response.status == "requires_action":
        tool_calls = response.required_action.submit_tool_outputs.tool_calls
        tool_outputs = []
        
        for tc in tool_calls:
            args = json.loads(tc.function.arguments)
            result = await handle_tool_call(tc.function.name, args, agent_id)
            
            # Surface contradiction to caller
            if result.get("status") == "contradiction_detected":
                contradiction_event = result
            
            tool_outputs.append({
                "tool_call_id": tc.id,
                "output": json.dumps(result)
            })
        
        response = await openai_client.responses.submit_tool_outputs(
            response_id=response.id,
            tool_outputs=tool_outputs
        )
    
    return response.output_text, response.id, contradiction_event

async def interactive_session(agent_id: str = "demo-agent"):
    """Full interactive session with checkpoint loop."""
    
    # Check for existing checkpoint on startup
    existing = await resume(agent_id)
    if existing:
        print(f"\n[SYSTEM] Resuming from checkpoint v{existing.checkpoint_version}")
        print(f"[SYSTEM] Previous task: {existing.working.task}")
        print(f"[SYSTEM] Progress: {existing.working.progress_pct}%")
        print()
    
    # Start background checkpoint loop
    asyncio.create_task(auto_checkpoint_loop(agent_id, interval_seconds=30))
    
    previous_response_id = None
    print("Temporal Memory OS — Type 'quit' to exit, 'memory' to see stats\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            await checkpoint(agent_id)
            break
        if user_input.lower() == "memory":
            memories = await retrieve_memories("everything", agent_id, k=20, min_confidence=0.0)
            print(f"\n[MEMORY DUMP] {len(memories)} memories:")
            for m in memories:
                print(f"  [{m.confidence:.2f}] {m.content}")
            print()
            continue
        
        response, previous_response_id, contradiction = await run_agent(
            agent_id, user_input, previous_response_id
        )
        
        if contradiction:
            print(f"\n[CONTRADICTION DETECTED]")
            print(f"  New: {contradiction['new_fact']}")
            print(f"  Conflicts with: {contradiction['conflicts_with']}")
            print(f"  {contradiction['explanation']}")
            print(f"  Which is correct? (type 'new' or 'old')")
            choice = input(">>> ").strip().lower()
            chosen = contradiction["new_fact"] if choice == "new" else contradiction["conflicts_with"]
            await resolve_contradiction(contradiction["contradiction_id"], chosen, agent_id)
            print(f"[SYSTEM] Resolved. Storing: '{chosen}'\n")
        
        print(f"\nAgent: {response}\n")
```

---

## SECTION 11 — FASTAPI DASHBOARD (api/server.py)

```python
import os
import time
import json
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as aioredis
from memory.episodic import retrieve_memories
from memory.working import get_working_memory
from memory.decay import calculate_current_confidence

app = FastAPI(title="Temporal Memory OS")
app.add_middleware(CORSMiddleware, allow_origins=["*"])

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
        min_confidence=0.0  # get all including stale
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
                "age_hours": round((time.time() - m.created_at) / 3600, 1)
            }
            for m in sorted(memories, key=lambda x: calculate_current_confidence(x), reverse=True)
        ][:20]
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
    from memory.contradiction import resolve_contradiction
    await resolve_contradiction(event_id, chosen_fact, agent_id)
    return {"status": "resolved"}
```

---

## SECTION 12 — DASHBOARD HTML (dashboard/index.html)

Build a single-file HTML dashboard that:
- Auto-refreshes every 3 seconds via `fetch("/api/memory/stats")`
- Shows a confidence heatmap: colored cards for each memory
  (green = high conf, yellow = medium, orange = low, red = stale)
- Shows working memory: current task, progress bar, last action
- Shows contradiction events feed from `/api/contradictions`
- Shows live token savings counter (calculate: avg_tokens_saved =
  total_memories * 150 - retrieved_memories * 150, show as big number)
- Design: dark background, minimal, hackathon-appropriate
- NO external dependencies — inline CSS + vanilla JS only

The dashboard is the visual WOW for judges. Make it look good.
Use CSS variables for colors. Animate the confidence bars.

---

## SECTION 13 — DEMO SCRIPT (tests/demo.py)

This is what you run in front of judges. Build it so it runs
automatically with no interaction needed (for the backup case),
but also works interactively.

```python
"""
DEMO SCRIPT - Run this for judges.
Shows three demo scenes in sequence.
"""
import asyncio
import time

async def demo_scene_1_contradiction():
    """Scene 1: Contradiction detection (90 seconds)"""
    print("\n" + "="*60)
    print("SCENE 1: Contradiction Detection")
    print("="*60 + "\n")
    
    from agent.loop import run_agent
    
    agent_id = "demo-agent"
    
    # Plant a fact
    print("Step 1: Planting a fact...")
    resp, rid, _ = await run_agent(
        agent_id,
        "Just so you know, our team standup is every Wednesday at 10am.",
        None
    )
    print(f"Agent: {resp}\n")
    await asyncio.sleep(2)
    
    # Now contradict it
    print("Step 2: Contradicting the fact...")
    resp, rid, contradiction = await run_agent(
        agent_id,
        "By the way, the standup was moved to Thursday at 9am.",
        rid
    )
    
    if contradiction:
        print("\n[LIVE DEMO MOMENT]")
        print(f"Contradiction detected with confidence: {contradiction['confidence_score']:.0%}")
        print(f"  Old: {contradiction['conflicts_with']}")
        print(f"  New: {contradiction['new_fact']}")
        print(f"  Explanation: {contradiction['explanation']}")
    
    print(f"\nAgent: {resp}\n")

async def demo_scene_2_session_recovery():
    """Scene 2: Session Recovery (60 seconds)"""
    print("\n" + "="*60)
    print("SCENE 2: Session Recovery")
    print("="*60 + "\n")
    
    from agent.loop import run_agent
    from sandbox.checkpoint import checkpoint, resume
    
    agent_id = "recovery-demo"
    
    print("Step 1: Starting a task...")
    resp, rid, _ = await run_agent(
        agent_id,
        "Research the top 5 AI memory papers from 2025 and summarize each.",
        None
    )
    print(f"Agent: {resp[:200]}...\n")
    
    print("Step 2: Simulating agent interruption — checkpointing now...")
    await checkpoint(agent_id)
    print("[CHECKPOINT SAVED TO BLAXEL SANDBOX + REDIS]\n")
    
    await asyncio.sleep(2)
    
    print("Step 3: Resuming agent (simulating new session)...")
    restored = await resume(agent_id)
    if restored:
        print(f"[RESTORED] Task: {restored.working.task}")
        print(f"[RESTORED] Progress: {restored.working.progress_pct}%")
        print(f"[RESTORED] Resuming with {len(restored.recent_episodic)} memories loaded")
    print()

async def demo_scene_3_confidence_decay():
    """Scene 3: Confidence decay visualization (45 seconds)"""
    print("\n" + "="*60)
    print("SCENE 3: Confidence Decay")
    print("="*60 + "\n")
    
    from memory.models import MemoryEntry
    from memory.decay import calculate_current_confidence
    
    # Show confidence of a memory over simulated time
    mem = MemoryEntry(
        id="demo",
        agent_id="demo",
        content="Meeting is at 3pm today",
        layer="episodic",
        source="user_said",
        decay_rate=0.05,
        created_at=time.time() - 7200  # pretend it's 2 hours old
    )
    
    print("Simulating confidence decay for: 'Meeting is at 3pm today'")
    print("(decay_rate=0.05/hr, currently 2 hours old)\n")
    
    for hours in [0, 2, 6, 12, 24, 48, 72]:
        sim_time = mem.created_at + (hours * 3600)
        conf = calculate_current_confidence(mem, now=sim_time)
        bar = "█" * int(conf * 20)
        print(f"  {hours:3d}h: [{bar:<20}] {conf:.2f}")
    
    print("\nAt 0.1 threshold, memory is filtered from context window.")
    print("At 0.0, memory is pruned from storage entirely.\n")

if __name__ == "__main__":
    asyncio.run(demo_scene_1_contradiction())
    asyncio.run(demo_scene_2_session_recovery())
    asyncio.run(demo_scene_3_confidence_decay())
    
    print("\n" + "="*60)
    print("DASHBOARD: http://localhost:8000")
    print("Start with: uvicorn api.server:app --reload")
    print("="*60 + "\n")
```

---

## SECTION 14 — REQUIREMENTS.TXT

```
openai>=1.75.0
blaxel>=0.1.0
redis>=5.0.0
redisvl>=0.5.1
fastapi>=0.115.0
uvicorn>=0.32.0
pydantic>=2.10.0
python-dotenv>=1.0.0
numpy>=1.26.0
```

---

## SECTION 15 — .ENV.EXAMPLE

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Redis Cloud (get from redis.com - free tier)
REDIS_URL=redis://default:password@host:port

# Blaxel (get from blaxel.ai - free $200 credits)
BL_WORKSPACE=your-workspace-name
BL_API_KEY=bl-...

# Agent config
SANDBOX_NAME=temporal-memory-os
DEFAULT_AGENT_ID=demo-agent
```

---

## SECTION 16 — README.md (for judges)

Write a README that opens with:

```
# Temporal Memory OS
### What Mem0 doesn't do — but should.

Mem0 makes agents remember. We make agents remember the RIGHT things,
surface conflicts before they poison reasoning, and never lose their
place mid-task.
```

Then explain the three core innovations vs Mem0:
1. Temporal confidence decay with exponential half-life
2. Explicit contradiction surfacing (not silent resolution)
3. Blaxel session recovery — agents resume mid-task

Then the benchmark: show what the LOCOMO benchmark is (Mem0 scores
26% over OpenAI on it) and explain which failure modes your system
addresses that LOCOMO doesn't even test for (temporal consistency
and session continuity).

---

## BUILD ORDER FOR TODAY

Do these in order. Do not start the next step until the previous passes a quick smoke test.

```
HOUR 1 (10am-11am):
  [1a] Complete all 4 Research Tasks above
  [1b] Set up project structure + requirements.txt
  [1c] Set up .env with working Redis Cloud + Blaxel credentials
  [1d] Write memory/models.py — smoke test with a quick `python -c "from memory.models import MemoryEntry; print(MemoryEntry(agent_id='test', content='hello', layer='episodic', source='user_said'))"`

HOUR 2 (11am-12pm):
  [2a] Write memory/decay.py + test decay function with print statements
  [2b] Write memory/working.py + test Redis Hash round-trip
  [2c] Write memory/extractor.py + test with one conversation turn
  [2d] Write memory/episodic.py + test add/retrieve round-trip

HOUR 3 (12pm-1pm):
  [3a] Write memory/contradiction.py — this is critical, test carefully
  [3b] Write agent/tools.py
  [3c] Write sandbox/checkpoint.py + test checkpoint/resume round-trip with Blaxel

HOUR 4 (1pm-2pm):
  [4a] Write agent/loop.py
  [4b] Run full interactive session locally, fix bugs
  [4c] Write api/server.py + start FastAPI server

HOUR 5 (2pm-4pm):
  [5a] Build dashboard/index.html
  [5b] Write tests/demo.py
  [5c] Run full demo end-to-end, time each scene

HOUR 6 (4pm-5pm):
  [6a] Write README.md
  [6b] Polish demo timing
  [6c] Practice the verbal pitch — 3 minutes max

HOUR 7 (5pm-7pm):
  [7a] Buffer for debugging
  [7b] Final demo run before presentations
```