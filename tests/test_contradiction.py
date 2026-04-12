import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-mock-key")

import pytest
from unittest.mock import AsyncMock

from memory.models import ContradictionEvent, MemoryEntry
import memory.contradiction as contradiction


@pytest.mark.asyncio
async def test_check_contradiction_none(monkeypatch):
    monkeypatch.setattr(contradiction, "retrieve_memories", AsyncMock(return_value=[]))
    result = await contradiction.check_contradiction("test", "agent-1")
    assert result is None


@pytest.mark.asyncio
async def test_check_contradiction_found(monkeypatch):
    mem = MemoryEntry(
        agent_id="agent-1", content="old fact", layer="episodic", source="user_said"
    )
    monkeypatch.setattr(
        contradiction, "retrieve_memories", AsyncMock(return_value=[mem])
    )

    mock_openai = AsyncMock()
    mock_openai.chat.completions.create.return_value.choices = [
        AsyncMock(
            message=AsyncMock(
                content='{"is_contradiction": true, "confidence": 0.8, "conflicting_memory_index": 0, "explanation": "conflict"}'
            )
        )
    ]
    monkeypatch.setattr(contradiction, "openai_client", mock_openai)

    mock_redis = AsyncMock()
    monkeypatch.setattr(
        "memory.working.get_redis", AsyncMock(return_value=mock_redis)
    )

    result = await contradiction.check_contradiction("new fact", "agent-1")

    assert result is not None
    assert result.new_fact == "new fact"
    assert result.conflicting_memory_id == mem.id
    mock_redis.setex.assert_called_once()
    mock_redis.lpush.assert_called_once()


@pytest.mark.asyncio
async def test_resolve_contradiction(monkeypatch):
    mock_redis = AsyncMock()
    event = ContradictionEvent(
        agent_id="agent-1",
        new_fact="new fact",
        conflicting_memory_id="mem-old",
        conflicting_memory_content="old fact",
        confidence_score=0.8,
        explanation="conflict",
    )
    mock_redis.get.return_value = event.model_dump_json()
    monkeypatch.setattr(
        "memory.working.get_redis", AsyncMock(return_value=mock_redis)
    )
    monkeypatch.setattr(contradiction, "log_event", AsyncMock())

    old_mem = MemoryEntry(
        id="mem-old",
        agent_id="agent-1",
        content="old fact",
        layer="episodic",
        source="user_said",
        version=2,
    )
    monkeypatch.setattr(contradiction, "get_memory_by_id", AsyncMock(return_value=old_mem))
    monkeypatch.setattr(contradiction, "add_memory", AsyncMock(return_value="mem-new"))
    mock_update = AsyncMock()
    monkeypatch.setattr(contradiction, "update_memory_fields", mock_update)

    await contradiction.resolve_contradiction(event.id, "new fact", "agent-1")

    mock_redis.set.assert_called_once()
    mock_update.assert_called_once()


@pytest.mark.asyncio
async def test_resolve_contradiction_keep_old(monkeypatch):
    mock_redis = AsyncMock()
    event = ContradictionEvent(
        agent_id="agent-1",
        new_fact="new fact",
        conflicting_memory_id="mem-old",
        conflicting_memory_content="old fact",
        confidence_score=0.8,
        explanation="conflict",
    )
    mock_redis.get.return_value = event.model_dump_json()
    monkeypatch.setattr(
        "memory.working.get_redis", AsyncMock(return_value=mock_redis)
    )
    monkeypatch.setattr(contradiction, "log_event", AsyncMock())

    old_mem = MemoryEntry(
        id="mem-old",
        agent_id="agent-1",
        content="old fact",
        layer="episodic",
        source="user_said",
    )
    monkeypatch.setattr(contradiction, "get_memory_by_id", AsyncMock(return_value=old_mem))
    mock_update = AsyncMock()
    monkeypatch.setattr(contradiction, "update_memory_fields", mock_update)

    await contradiction.resolve_contradiction(event.id, "old fact", "agent-1")

    mock_redis.set.assert_called_once()
    mock_update.assert_called_once()
