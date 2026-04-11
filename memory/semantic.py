import os
import time

from openai import AsyncOpenAI
from redisvl.index import AsyncSearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag


from .models import SEMANTIC_SCHEMA

openai_client = AsyncOpenAI()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


async def get_semantic_index() -> AsyncSearchIndex:
    if SEMANTIC_SCHEMA is None:
        raise RuntimeError("SEMANTIC_SCHEMA is not available. Install redisvl.")
    index = AsyncSearchIndex(SEMANTIC_SCHEMA, redis_url=REDIS_URL)
    await index.create(overwrite=False)
    return index


async def embed(text: str) -> list[float]:
    response = await openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


async def add_semantic_memory(agent_id: str, summary: str, source: str) -> str:
    """Store a summarized semantic memory."""
    index = await get_semantic_index()
    embedding = await embed(summary)

    import uuid

    mem_id = str(uuid.uuid4())

    doc = {
        "id": mem_id,
        "agent_id": agent_id,
        "summary": summary,
        "source": source,
        "confidence": 1.0,
        "created_at": time.time(),
        "embedding": embedding,
    }
    await index.load([doc], id_field="id")
    return mem_id


async def retrieve_semantic_memories(
    query: str,
    agent_id: str,
    k: int = 5,
) -> list[dict]:
    """Retrieve broad semantic facts."""
    index = await get_semantic_index()
    embedding = await embed(query)

    agent_filter = Tag("agent_id") == agent_id

    query_obj = VectorQuery(
        vector=embedding,
        vector_field_name="embedding",
        filter_expression=agent_filter,
        num_results=k,
        return_fields=[
            "id",
            "summary",
            "source",
            "confidence",
            "created_at",
            "vector_distance",
        ],
    )

    results = await index.query(query_obj)

    memories = []
    for r in results:
        memories.append(
            {
                "id": r["id"],
                "summary": r["summary"],
                "source": r["source"],
                "confidence": float(r["confidence"]),
                "created_at": float(r["created_at"]),
                "relevance": 1 - float(r["vector_distance"]),
            }
        )

    return memories
