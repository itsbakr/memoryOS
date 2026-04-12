import pytest
from unittest.mock import AsyncMock

from memory.models import MemoryEntry
import memory.retrieval as retrieval


@pytest.mark.asyncio
@pytest.mark.eval
async def test_hybrid_retrieval_fusion(monkeypatch):
    mem_a = MemoryEntry(
        id="mem-a",
        agent_id="agent-1",
        content="Uses Redis for memory storage",
        layer="episodic",
        source="user_said",
    )
    mem_b = MemoryEntry(
        id="mem-b",
        agent_id="agent-1",
        content="Prefers TypeScript for frontend",
        layer="episodic",
        source="user_said",
    )

    monkeypatch.setattr(
        retrieval,
        "vector_search_with_scores",
        AsyncMock(
            return_value=[
                {
                    "memory": mem_a,
                    "score": 0.6,
                    "vector_similarity": 0.7,
                    "live_confidence": 0.85,
                },
                {
                    "memory": mem_b,
                    "score": 0.4,
                    "vector_similarity": 0.55,
                    "live_confidence": 0.75,
                },
            ]
        ),
    )
    monkeypatch.setattr(
        retrieval, "related_fact_ids", AsyncMock(return_value=["mem-c"])
    )
    mem_c = MemoryEntry(
        id="mem-c",
        agent_id="agent-1",
        content="Graph memory links entities",
        layer="episodic",
        source="user_said",
    )
    monkeypatch.setattr(retrieval, "get_memory_by_id", AsyncMock(return_value=mem_c))

    memories, provenance = await retrieval.hybrid_retrieve(
        "memory architecture", "agent-1", k=3
    )

    assert [m.id for m in memories] == ["mem-a", "mem-b", "mem-c"]
    assert provenance[0]["sources"] == ["vector"]
    assert provenance[-1]["sources"] == ["graph"]
