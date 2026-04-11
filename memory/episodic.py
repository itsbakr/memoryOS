import os
import time

from openai import AsyncOpenAI
from redisvl.index import AsyncSearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag

from .decay import calculate_current_confidence, reinforce_memory
from .models import EPISODIC_SCHEMA, MemoryEntry

import redis.asyncio as aioredis

openai_client = AsyncOpenAI()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
STREAM_MAXLEN = 2000


_index_instance: AsyncSearchIndex | None = None

async def get_index() -> AsyncSearchIndex:
    global _index_instance
    if _index_instance is not None:
        return _index_instance
        
    if EPISODIC_SCHEMA is None:
        raise RuntimeError("EPISODIC_SCHEMA is not available. Install redisvl.")
    index = AsyncSearchIndex(EPISODIC_SCHEMA, redis_url=REDIS_URL)
    await index.create(overwrite=False)
    _index_instance = index
    return index


_redis_pool = None

async def get_redis():
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis_pool


def _stream_key(agent_id: str) -> str:
    return f"agent:{agent_id}:episodic_stream"


async def embed(text: str) -> list[float]:
    response = await openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
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
        mem = MemoryEntry(
            id=r["id"],
            agent_id=agent_id,
            content=r["content"],
            layer="episodic",
            source=r["source"],
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
