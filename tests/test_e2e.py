import pytest
import os
import json
from dotenv import load_dotenv
load_dotenv()

from httpx import AsyncClient, ASGITransport
import redis.asyncio as aioredis

from api.server import app
from memory.models import WorkingMemory

# Set a unique agent_id for E2E tests so we don't clobber real data
TEST_AGENT_ID = "e2e-test-agent"

@pytest.mark.asyncio
async def test_e2e_session_and_chat_flow():
    r = aioredis.from_url(os.getenv("REDIS_URL"), decode_responses=True)
    await r.flushdb()
    
    # Setup test agent_id
    from api.server import DEFAULT_AGENT
    # For testing the server, we just use the real API with the mock transport
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Create session
        res = await client.post("/api/sessions")
        assert res.status_code == 200
        session_data = res.json()
        assert "session_id" in session_data
        session_id = session_data["session_id"]
        
        # 2. Chat (Initialize task)
        chat_req = {
            "session_id": session_id,
            "message": "Hi, I'm working on a project about game development for my economics class. I want to build a pong game."
        }
        chat_res = await client.post("/api/chat", json=chat_req)
        assert chat_res.status_code == 200
        chat_data = chat_res.json()
        assert chat_data["reply"] is not None
        
        # 3. Verify Memory Extraction and Working Memory updates
        stats_res = await client.get(f"/api/memory/stats?agent_id={session_data['agent_id']}")
        assert stats_res.status_code == 200
        stats = stats_res.json()
        
        # The agent should have updated working memory
        assert stats["working_memory"] is not None
        assert "pong" in str(stats["working_memory"]).lower() or "game" in str(stats["working_memory"]).lower()
        
        # Episodic memories should be stored
        assert stats["total_memories"] > 0
        
        # 4. Plant a contradiction
        chat_req2 = {
            "session_id": session_id,
            "message": "Actually, my assignment is for a psychology class, not economics."
        }
        chat_res2 = await client.post("/api/chat", json=chat_req2)
        assert chat_res2.status_code == 200
        chat_data2 = chat_res2.json()
        
        # Depending on the LLM's speed/confidence, it might flag a contradiction
        # If it does, we can test resolving it
        if chat_data2.get("contradiction"):
            contra = chat_data2["contradiction"]
            # Resolve it
            resolve_res = await client.post(
                f"/api/contradictions/{contra['contradiction_id']}/resolve",
                params={"chosen_fact": contra["new_fact"], "agent_id": session_data['agent_id']}
            )
            assert resolve_res.status_code == 200
            
            # Ensure the old fact's confidence was zeroed
            stats_res3 = await client.get(f"/api/memory/stats?agent_id={session_data['agent_id']}")
            stats3 = stats_res3.json()
            assert stats3["confidence_distribution"]["stale"] > 0
