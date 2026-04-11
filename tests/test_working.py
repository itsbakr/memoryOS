import pytest
import fakeredis.aioredis

from unittest.mock import AsyncMock

import memory.working as working
from memory.models import WorkingMemory


@pytest.mark.asyncio
async def test_working_memory_round_trip(monkeypatch):
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)

    def fake_get_redis(*_args, **_kwargs):
        return fake

    monkeypatch.setattr(working, "get_redis", AsyncMock(return_value=fake))

    payload = WorkingMemory(agent_id="agent-1", task="Plan demo", progress_pct=0.4)
    await working.set_working_memory(payload)

    loaded = await working.get_working_memory("agent-1")
    assert loaded is not None
    assert loaded.task == "Plan demo"
    assert float(loaded.progress_pct) == 0.4

    await working.increment_tool_calls("agent-1")
    updated = await working.get_working_memory("agent-1")
    assert int(updated.tool_calls_made) == 1

    await working.clear_working_memory("agent-1")
    assert await working.get_working_memory("agent-1") is None
