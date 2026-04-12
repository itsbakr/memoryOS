import os
import time

from openai import AsyncOpenAI
from redisvl.index import AsyncSearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag

import redis.asyncio as aioredis

from .decay import calculate_current_confidence, reinforce_memory
from .models import EPISODIC_SCHEMA, MemoryEntry
from .working import get_redis

openai_client = AsyncOpenAI()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
STREAM_MAXLEN = 2000


_index_instance: AsyncSearchIndex | None = None


async def get_index() -> AsyncSearchIndex:
    """
    Returns a ready AsyncSearchIndex, creating it if needed.
    Always verifies the index actually exists in Redis — if the cached instance
    points to a dropped index (e.g. after Redis Cloud instability) it recreates it.
    """
    global _index_instance

    if EPISODIC_SCHEMA is None:
        raise RuntimeError("EPISODIC_SCHEMA is not available. Install redisvl.")

    client = await get_redis()

    if _index_instance is not None:
        # Verify the cached instance is still live
        try:
            await _index_instance.info()
            return _index_instance
        except Exception:
            # Index was dropped — fall through to recreate
            _index_instance = None

    index = AsyncSearchIndex(EPISODIC_SCHEMA, redis_client=client)
    await index.create(overwrite=False)
    _index_instance = index
    return index

def _stream_key(agent_id: str) -> str:
    return f"agent:{agent_id}:episodic_stream"

async def count_memories(agent_id: str) -> int:
    """Fast O(1) way to count total memories stored for an agent via the stream."""
    r = await get_redis()
    return await r.xlen(_stream_key(agent_id))


async def list_memory_ids(agent_id: str, limit: int = 500) -> list[str]:
    r = await get_redis()
    entries = await r.xrevrange(_stream_key(agent_id), count=limit)
    ids = []
    for _stream_id, payload in entries:
        mem_id = payload.get("id")
        if mem_id:
            ids.append(mem_id)
    return list(dict.fromkeys(ids))


async def embed(text: str) -> list[float]:
    response = await openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


DEDUP_THRESHOLD = 0.93  # cosine similarity above this = duplicate


async def add_memory(memory: MemoryEntry) -> str:
    """
    Store a new episodic memory with embedding.
    Skips storing if a near-identical memory already exists (cosine sim > DEDUP_THRESHOLD)
    to prevent the same fact being stored multiple times from repeated extraction.
    Returns the id of the stored (or existing) memory.
    """
    index = await get_index()
    embedding = await embed(memory.content)
    memory.embedding = embedding

    # Deduplication check — vector search for near-identical content
    agent_filter = Tag("agent_id") == memory.agent_id
    dedup_query = VectorQuery(
        vector=embedding,
        vector_field_name="embedding",
        filter_expression=agent_filter,
        num_results=1,
        return_fields=["id", "content", "vector_distance"],
    )
    try:
        dedup_results = await index.query(dedup_query)
        if dedup_results:
            top = dedup_results[0]
            similarity = 1 - float(top.get("vector_distance", 1))
            if similarity >= DEDUP_THRESHOLD:
                return top["id"]  # already stored — skip
    except Exception:
        pass  # if dedup check fails, proceed with storing

    doc = {
        "id": memory.id,
        "agent_id": memory.agent_id,
        "content": memory.content,
        "source": memory.source,
        "category": memory.category or "general",
        "confidence": memory.confidence,
        "decay_rate": memory.decay_rate,
        "created_at": memory.created_at,
        "last_reinforced": memory.last_reinforced,
        "valid_from": memory.valid_from,
        "valid_to": memory.valid_to,
        "expires_at": memory.expires_at,
        "supersedes_id": memory.supersedes_id,
        "superseded_by_id": memory.superseded_by_id,
        "is_active": 1 if memory.is_active else 0,
        "version": memory.version,
        "embedding": embedding,
    }
    await index.load([doc], id_field="id")
    await _append_stream_event(memory)
    return memory.id


async def _append_stream_event(memory: MemoryEntry) -> None:
    r = await get_redis()
    await r.xadd(
        _stream_key(memory.agent_id),
        {
            "id": memory.id,
            "content": memory.content,
            "source": memory.source,
            "confidence": str(memory.confidence),
            "created_at": str(memory.created_at),
        },
        maxlen=STREAM_MAXLEN,
        approximate=True,
    )


async def retrieve_memories(
    query: str,
    agent_id: str,
    k: int = 10,
    min_confidence: float = 0.3,
    active_only: bool = True,
) -> list[MemoryEntry]:
    """
    Vector search filtered by agent_id and min_confidence.
    Applies decay before filtering — confidence is live, not stored.
    Returns memories ranked by relevance * current_confidence.
    """
    results = await vector_search_with_scores(
        query,
        agent_id,
        k=k,
        min_confidence=min_confidence,
        active_only=active_only,
    )
    return [item["memory"] for item in results]


async def vector_search_with_scores(
    query: str,
    agent_id: str,
    k: int = 10,
    min_confidence: float = 0.3,
    active_only: bool = True,
) -> list[dict]:
    index = await get_index()
    embedding = await embed(query)

    agent_filter = Tag("agent_id") == agent_id

    query_obj = VectorQuery(
        vector=embedding,
        vector_field_name="embedding",
        filter_expression=agent_filter,
        num_results=k * 3,  # oversample, then filter by live confidence
        return_fields=[
            "id",
            "content",
            "source",
            "category",
            "confidence",
            "decay_rate",
            "created_at",
            "last_reinforced",
            "valid_from",
            "valid_to",
            "expires_at",
            "supersedes_id",
            "superseded_by_id",
            "is_active",
            "version",
            "vector_distance",
        ],
    )

    results = await index.query(query_obj)

    memories: list[dict] = []
    for r in results:
        # RedisVL returns the full key in r["id"]. Strip prefix if present.
        raw_id = r["id"]
        if raw_id.startswith("mem:episodic:"):
            raw_id = raw_id.replace("mem:episodic:", "", 1)

        mem = MemoryEntry(
            id=raw_id,
            agent_id=agent_id,
            content=r["content"],
            layer="episodic",
            source=r["source"],
            category=r.get("category") or "general",
            confidence=float(r["confidence"]),
            decay_rate=float(r["decay_rate"]),
            created_at=float(r["created_at"]),
            last_reinforced=float(r["last_reinforced"]),
            valid_from=float(r.get("valid_from", r["created_at"])),
            valid_to=float(r["valid_to"]) if r.get("valid_to") else None,
            expires_at=float(r["expires_at"]) if r.get("expires_at") else None,
            supersedes_id=r.get("supersedes_id") or None,
            superseded_by_id=r.get("superseded_by_id") or None,
            is_active=bool(int(r.get("is_active", 1))),
            version=int(r.get("version", 1)),
        )
        live_conf = calculate_current_confidence(mem)
        if mem.expires_at and time.time() >= mem.expires_at:
            continue
        if active_only and not mem.is_active:
            continue
        if live_conf >= min_confidence:
            relevance = 1 - float(r["vector_distance"])  # cosine: 0=identical
            mem.confidence = live_conf
            score = relevance * live_conf
            memories.append(
                {
                    "memory": mem,
                    "score": score,
                    "vector_similarity": relevance,
                    "live_confidence": live_conf,
                }
            )

    memories.sort(key=lambda x: x["score"], reverse=True)
    return memories[:k]


async def reinforce_retrieved(memories: list[MemoryEntry]) -> None:
    for memory in memories:
        reinforce_memory(memory)
        await update_memory_confidence(memory.id, memory.confidence)


async def update_memory_confidence(memory_id: str, new_confidence: float) -> None:
    """Called after contradiction resolution or reinforcement."""
    await update_memory_fields(
        memory_id,
        {"confidence": new_confidence, "last_reinforced": time.time()},
    )


async def update_memory_fields(memory_id: str, updates: dict[str, object]) -> None:
    """Patch mutable fields on a stored episodic memory."""
    index = await get_index()
    # In redis-py async, json() is likely a property or method returning an object with set
    json_client = index.client.json()
    for key, value in updates.items():
        await json_client.set(f"mem:episodic:{memory_id}", f"$.{key}", value)


async def get_memory_by_id(memory_id: str) -> MemoryEntry | None:
    index = await get_index()
    json_client = index.client.json()
    raw = await json_client.get(f"mem:episodic:{memory_id}")
    if not raw:
        return None
    return MemoryEntry(
        id=memory_id,
        agent_id=raw["agent_id"],
        content=raw["content"],
        layer="episodic",
        source=raw["source"],
        category=raw.get("category") or "general",
        confidence=float(raw["confidence"]),
        decay_rate=float(raw["decay_rate"]),
        created_at=float(raw["created_at"]),
        last_reinforced=float(raw["last_reinforced"]),
        valid_from=float(raw.get("valid_from", raw["created_at"])),
        valid_to=float(raw["valid_to"]) if raw.get("valid_to") else None,
        expires_at=float(raw["expires_at"]) if raw.get("expires_at") else None,
        supersedes_id=raw.get("supersedes_id") or None,
        superseded_by_id=raw.get("superseded_by_id") or None,
        is_active=bool(int(raw.get("is_active", 1))),
        version=int(raw.get("version", 1)),
    )
