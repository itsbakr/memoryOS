from __future__ import annotations

from .episodic import get_memory_by_id, vector_search_with_scores
from .graph import related_fact_ids

GRAPH_SCORE = 0.2
GRAPH_BOOST = 0.1


async def hybrid_retrieve(
    query: str,
    agent_id: str,
    *,
    k: int = 5,
    min_confidence: float = 0.3,
    active_only: bool = True,
) -> tuple[list, list[dict]]:
    """
    Hybrid retrieval that fuses vector similarity with graph proximity.
    Returns (memories, provenance).
    """
    vector_results = await vector_search_with_scores(
        query,
        agent_id,
        k=k,
        min_confidence=min_confidence,
        active_only=active_only,
    )

    combined: dict[str, dict] = {}
    for item in vector_results:
        mem = item["memory"]
        combined[mem.id] = {
            "memory": mem,
            "score": item["score"],
            "components": {
                "vector_similarity": item["vector_similarity"],
                "live_confidence": item["live_confidence"],
            },
            "sources": ["vector"],
        }

    graph_ids = await related_fact_ids(agent_id, query, limit=k * 3)
    for memory_id in graph_ids:
        if memory_id in combined:
            combined[memory_id]["score"] += GRAPH_BOOST
            combined[memory_id]["components"]["graph_boost"] = GRAPH_BOOST
            combined[memory_id]["sources"].append("graph")
            continue

        mem = await get_memory_by_id(memory_id)
        if not mem:
            continue
        if active_only and not mem.is_active:
            continue
        combined[memory_id] = {
            "memory": mem,
            "score": GRAPH_SCORE,
            "components": {"graph_score": GRAPH_SCORE},
            "sources": ["graph"],
        }

    ranked = sorted(combined.values(), key=lambda item: item["score"], reverse=True)[:k]
    memories = [item["memory"] for item in ranked]
    provenance = [
        {
            "memory_id": item["memory"].id,
            "content": item["memory"].content,
            "category": item["memory"].category or "general",
            "confidence": round(item["memory"].confidence, 3),
            "score": round(item["score"], 4),
            "sources": item["sources"],
            "components": item["components"],
        }
        for item in ranked
    ]

    return memories, provenance
