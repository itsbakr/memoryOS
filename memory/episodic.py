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
        return_fields=[
            "id",
            "content",
            "source",
            "category",
            "confidence",
            "decay_rate",
            "created_at",
            "last_reinforced",
            "vector_distance",
        ],
    )

    results = await index.query(query_obj)

    memories: list[tuple[MemoryEntry, float]] = []
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
        )
        live_conf = calculate_current_confidence(mem)
        if live_conf >= min_confidence:
            relevance = 1 - float(r["vector_distance"])  # cosine: 0=identical
            mem.confidence = live_conf
            memories.append((mem, relevance * live_conf))

    memories.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in memories[:k]]


async def reinforce_retrieved(memories: list[MemoryEntry]) -> None:
    for memory in memories:
        reinforce_memory(memory)
        await update_memory_confidence(memory.id, memory.confidence)


async def update_memory_confidence(memory_id: str, new_confidence: float) -> None:
    """Called after contradiction resolution or reinforcement."""
    index = await get_index()
    # In redis-py async, json() is likely a property or method returning an object with set
    json_client = index.client.json()
    await json_client.set(
        f"mem:episodic:{memory_id}",
        "$.confidence",
        new_confidence,
    )
    await json_client.set(
        f"mem:episodic:{memory_id}",
        "$.last_reinforced",
        time.time(),
    )
