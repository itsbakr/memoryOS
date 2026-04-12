import pytest
import os
import json
from dotenv import load_dotenv
from unittest.mock import AsyncMock

load_dotenv()

from httpx import AsyncClient, ASGITransport
import fakeredis.aioredis as fakeredis

from api.server import app
import api.server as api_server
import memory.episodic as episodic
import memory.retrieval as retrieval
import memory.write_gate as write_gate
from memory.models import ContradictionEvent
from memory.working import get_redis
from memory.models import WorkingMemory

# Set a unique agent_id for E2E tests so we don't clobber real data
TEST_AGENT_ID = "e2e-test-agent"

@pytest.mark.asyncio
async def test_e2e_session_chat_provenance(monkeypatch):
    os.environ.setdefault("OPENAI_API_KEY", "sk-test-mock-key")
    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr("memory.working.get_redis", AsyncMock(return_value=fake_redis))
    monkeypatch.setattr(api_server, "get_redis", AsyncMock(return_value=fake_redis))
    monkeypatch.setattr("memory.audit.get_redis", AsyncMock(return_value=fake_redis))
    monkeypatch.setattr("memory.graph.get_redis", AsyncMock(return_value=fake_redis))

    async def fake_run_agent(agent_id: str, user_message: str, previous_response_id: str | None = None):
        provenance = [
            {
                "memory_id": "mem-provenance",
                "content": "Uses Redis for memory storage",
                "category": "project_decision",
                "confidence": 0.95,
                "score": 0.7,
                "sources": ["vector"],
                "components": {"vector_similarity": 0.8, "live_confidence": 0.9},
            }
        ]
        return "Test reply", "resp-1", None, provenance

    monkeypatch.setattr(api_server, "run_agent", AsyncMock(side_effect=fake_run_agent))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post("/api/sessions")
        assert res.status_code == 200
        session_data = res.json()
        session_id = session_data["session_id"]

        chat_res = await client.post(
            "/api/chat",
            json={"session_id": session_id, "message": "Hello memory"},
        )
        assert chat_res.status_code == 200
        chat_data = chat_res.json()
        assert chat_data["reply"] == "Test reply"
        assert chat_data["provenance"]

        history_res = await client.get(f"/api/chat/history?session_id={session_id}")
        history = history_res.json()["history"]
        assert history[-1]["provenance"][0]["memory_id"] == "mem-provenance"


@pytest.mark.asyncio
async def test_e2e_memory_governance_and_contradiction(monkeypatch):
    os.environ.setdefault("OPENAI_API_KEY", "sk-test-mock-key")
    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr("memory.working.get_redis", AsyncMock(return_value=fake_redis))
    monkeypatch.setattr(api_server, "get_redis", AsyncMock(return_value=fake_redis))
    monkeypatch.setattr("memory.audit.get_redis", AsyncMock(return_value=fake_redis))
    monkeypatch.setattr("memory.graph.get_redis", AsyncMock(return_value=fake_redis))

    stored: dict[str, MemoryEntry] = {}

    async def fake_add_memory(entry: MemoryEntry):
        stored[entry.id] = entry
        return entry.id

    async def fake_retrieve_memories(_query, _agent_id, k=10, min_confidence=0.0, active_only=True):
        items = [m for m in stored.values() if (m.is_active or not active_only)]
        return items[:k]

    async def fake_count_memories(_agent_id):
        return len(stored)

    async def fake_get_memory_by_id(memory_id: str):
        return stored.get(memory_id)

    async def fake_list_memory_ids(_agent_id, limit=500):
        return list(stored.keys())[:limit]

    async def fake_update_fields(memory_id: str, updates: dict):
        mem = stored.get(memory_id)
        if not mem:
            return
        if "is_active" in updates:
            updates["is_active"] = bool(updates["is_active"])
        stored[memory_id] = mem.model_copy(update=updates)

    async def fake_hybrid_retrieve(_query, _agent_id, k=5, min_confidence=0.0, active_only=True):
        items = [m for m in stored.values() if (m.is_active or not active_only)]
        items = items[:k]
        provenance = [
            {
                "memory_id": m.id,
                "content": m.content,
                "category": m.category or "general",
                "confidence": round(m.confidence, 3),
                "score": 0.5,
                "sources": ["vector"],
                "components": {"vector_similarity": 0.5, "live_confidence": m.confidence},
            }
            for m in items
        ]
        return items, provenance

    monkeypatch.setattr(episodic, "add_memory", AsyncMock(side_effect=fake_add_memory))
    monkeypatch.setattr(write_gate, "add_memory", AsyncMock(side_effect=fake_add_memory))
    monkeypatch.setattr(write_gate, "add_fact_to_graph", AsyncMock())
    monkeypatch.setattr(write_gate, "check_contradiction", AsyncMock(return_value=None))
    monkeypatch.setattr(episodic, "retrieve_memories", AsyncMock(side_effect=fake_retrieve_memories))
    monkeypatch.setattr(episodic, "count_memories", AsyncMock(side_effect=fake_count_memories))
    monkeypatch.setattr(episodic, "get_memory_by_id", AsyncMock(side_effect=fake_get_memory_by_id))
    monkeypatch.setattr(episodic, "list_memory_ids", AsyncMock(side_effect=fake_list_memory_ids))
    monkeypatch.setattr(episodic, "update_memory_fields", AsyncMock(side_effect=fake_update_fields))
    monkeypatch.setattr(retrieval, "hybrid_retrieve", AsyncMock(side_effect=fake_hybrid_retrieve))
    monkeypatch.setattr("memory.contradiction.get_memory_by_id", AsyncMock(side_effect=fake_get_memory_by_id))
    monkeypatch.setattr("memory.contradiction.update_memory_fields", AsyncMock(side_effect=fake_update_fields))
    monkeypatch.setattr("memory.contradiction.add_memory", AsyncMock(side_effect=fake_add_memory))
    monkeypatch.setattr(api_server, "retrieve_memories", AsyncMock(side_effect=fake_retrieve_memories))
    monkeypatch.setattr(api_server, "get_memory_by_id", AsyncMock(side_effect=fake_get_memory_by_id))
    monkeypatch.setattr(api_server, "list_memory_ids", AsyncMock(side_effect=fake_list_memory_ids))
    monkeypatch.setattr(api_server, "hybrid_retrieve", AsyncMock(side_effect=fake_hybrid_retrieve))
    monkeypatch.setattr(api_server, "update_memory_fields", AsyncMock(side_effect=fake_update_fields))
    mock_index = AsyncMock()
    mock_json_client = AsyncMock()
    mock_index.client = AsyncMock()
    mock_index.client.json = lambda: mock_json_client
    monkeypatch.setattr(episodic, "get_index", AsyncMock(return_value=mock_index))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        store_res = await client.post(
            "/api/memory/store",
            json={
                "agent_id": TEST_AGENT_ID,
                "content": "We use Redis for memory storage",
                "category": "project_decision",
                "source": "user_said",
            },
        )
        store_data = store_res.json()
        assert store_data["status"] == "stored"
        memory_id = store_data["memory_id"]

        working_res = await client.post(
            "/api/memory/working",
            json={
                "agent_id": TEST_AGENT_ID,
                "task": "Test task",
                "progress_pct": 25,
                "last_action": "Started tests",
            },
        )
        assert working_res.status_code == 200

        stats_res = await client.get(f"/api/memory/stats?agent_id={TEST_AGENT_ID}")
        stats = stats_res.json()
        assert stats["total_memories"] >= 1
        assert stats["working_memory"]["task"] == "Test task"

        export_res = await client.get(f"/api/memory/export?agent_id={TEST_AGENT_ID}")
        export_data = export_res.json()
        assert export_data["count"] >= 1

        graph_res = await client.get(f"/api/memory/graph/stats?agent_id={TEST_AGENT_ID}")
        assert graph_res.status_code == 200

        audit_res = await client.get(f"/api/memory/audit?agent_id={TEST_AGENT_ID}")
        assert audit_res.status_code == 200

        metrics_res = await client.get(f"/api/memory/metrics?agent_id={TEST_AGENT_ID}")
        assert metrics_res.status_code == 200

        search_res = await client.get(
            f"/api/memory/search?agent_id={TEST_AGENT_ID}&query=Redis&limit=5"
        )
        assert search_res.status_code == 200
        assert search_res.json()["results"]

        # Seed a contradiction event and resolve in favor of new fact
        event = ContradictionEvent(
            agent_id=TEST_AGENT_ID,
            new_fact="We use Postgres instead",
            conflicting_memory_id=memory_id,
            conflicting_memory_content="We use Redis for memory storage",
            confidence_score=0.9,
            explanation="conflict",
        )
        await fake_redis.setex(
            f"contradiction:{event.id}", 3600, event.model_dump_json()
        )
        await fake_redis.lpush(f"agent:{TEST_AGENT_ID}:contradictions", event.id)

        resolve_res = await client.post(
            f"/api/contradictions/{event.id}/resolve",
            params={"chosen_fact": event.new_fact, "agent_id": TEST_AGENT_ID},
        )
        assert resolve_res.status_code == 200

        export_after = await client.get(f"/api/memory/export?agent_id={TEST_AGENT_ID}&include_inactive=true")
        memories = export_after.json()["memories"]
        assert any(m["content"] == "We use Postgres instead" for m in memories)
        assert any(m["id"] == memory_id and m["is_active"] is False for m in memories)

        delete_res = await client.delete(
            f"/api/memory/{memory_id}?agent_id={TEST_AGENT_ID}"
        )
        assert delete_res.status_code == 200

        health_res = await client.get("/api/health")
        assert health_res.status_code == 200
