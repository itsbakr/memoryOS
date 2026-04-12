import pytest
from unittest.mock import AsyncMock
import time

import os

# Ensure dummy OPENAI_API_KEY for tests
os.environ.setdefault("OPENAI_API_KEY", "sk-test-mock-key")

from memory.models import MemoryEntry
import memory.episodic as episodic


@pytest.mark.asyncio
async def test_add_and_retrieve_memories(monkeypatch):
    # Mock openai embeddings
    mock_embeddings = AsyncMock()
    mock_embeddings.create.return_value.data = [AsyncMock(embedding=[0.1] * 1536)]
    monkeypatch.setattr(episodic.openai_client, "embeddings", mock_embeddings)

    # Mock RedisVL AsyncSearchIndex
    mock_index = AsyncMock()
    mock_index.load = AsyncMock()

    # Setup mock query results
    mock_result = {
        "id": "mem-1",
        "content": "test memory",
        "source": "user_said",
        "confidence": "1.0",
        "decay_rate": "0.05",
        "created_at": str(time.time()),
        "last_reinforced": str(time.time()),
        "vector_distance": "0.1",
    }
    mock_index.query.return_value = [mock_result]

    monkeypatch.setattr(episodic, "get_index", AsyncMock(return_value=mock_index))

    # Mock underlying Redis Stream
    mock_redis = AsyncMock()
    monkeypatch.setattr(episodic, "get_redis", AsyncMock(return_value=mock_redis))

    # Test adding
    mem = MemoryEntry(
        agent_id="agent-1", content="test memory", layer="episodic", source="user_said"
    )
    mem_id = await episodic.add_memory(mem)

    assert mem_id == mem.id
    mock_index.load.assert_called_once()
    mock_redis.xadd.assert_called_once()

    # Test retrieving
    results = await episodic.retrieve_memories("test", "agent-1", min_confidence=0.0)
    assert len(results) == 1
    assert results[0].id == "mem-1"
    assert results[0].content == "test memory"
    assert mock_index.query.call_count == 2


@pytest.mark.asyncio
async def test_reinforce_retrieved(monkeypatch):
    mock_index = AsyncMock()
    mock_json_client = AsyncMock()
    mock_index.client = AsyncMock()
    mock_index.client.json = lambda: mock_json_client
    monkeypatch.setattr(episodic, "get_index", AsyncMock(return_value=mock_index))

    mem = MemoryEntry(
        agent_id="agent-1",
        content="test",
        layer="episodic",
        source="user_said",
        confidence=0.5,
    )
    await episodic.reinforce_retrieved([mem])

    assert mem.confidence == 0.7
    assert mock_json_client.set.call_count == 2
