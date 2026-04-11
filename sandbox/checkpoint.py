import asyncio
import os

from blaxel.core import SandboxInstance
from memory.episodic import retrieve_memories
from memory.models import CheckpointPayload
from memory.working import get_working_memory, set_working_memory
import redis.asyncio as aioredis

SANDBOX_NAME = os.getenv("SANDBOX_NAME", "temporal-memory-agent")
CHECKPOINT_PATH = "/memory/checkpoint.json"
CHECKPOINT_VERSION_KEY = "checkpoint_version"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Fast check if Blaxel is properly configured
HAS_BLAXEL = bool(os.getenv("BL_API_KEY"))


async def get_or_create_sandbox() -> SandboxInstance | None:
    """
    Creates sandbox if not exists, otherwise gets existing.
    Returns None if Blaxel is not configured, falling back to local only.
    """
    if not HAS_BLAXEL:
        return None

    sandbox = await SandboxInstance.create_if_not_exists(
        {
            "name": SANDBOX_NAME,
            "image": "blaxel/python:latest",
            "memory": 2048,
            "region": "us-pdx-1",
            "ttl": "48h",
            "ports": [{"target": 8000}],
        }
    )
    return sandbox


async def checkpoint(agent_id: str) -> str | None:
    """
    Save full agent state to BOTH Redis (fast retrieval) and
    Blaxel sandbox filesystem (persistent, survives Redis flush).
    """
    sandbox = await get_or_create_sandbox()
    working = await get_working_memory(agent_id)

    if not working:
        return None

    recent = await retrieve_memories(
        working.task or "current task", agent_id, k=10, min_confidence=0.2
    )

    from memory.working import get_redis
    r = await get_redis()
    version = int(await r.incr(f"agent:{agent_id}:{CHECKPOINT_VERSION_KEY}"))

    payload = CheckpointPayload(
        agent_id=agent_id,
        working=working,
        recent_episodic=recent,
        checkpoint_version=version,
    )
    checkpoint_json = payload.model_dump_json(indent=2)

    # Always write to local backup for resilience
    os.makedirs(".checkpoints", exist_ok=True)
    with open(f".checkpoints/{agent_id}_latest.json", "w") as f:
        f.write(checkpoint_json)

    if sandbox:
        # Write to Blaxel sandbox filesystem
        try:
            # Check if write is a coroutine
            import inspect
            if inspect.iscoroutinefunction(sandbox.fs.write):
                await sandbox.fs.write(CHECKPOINT_PATH, checkpoint_json)
            else:
                sandbox.fs.write(CHECKPOINT_PATH, checkpoint_json)
            print(f"[CHECKPOINT] v{version} saved to Blaxel sandbox + Redis")
        except Exception as e:
            print(f"[CHECKPOINT] Blaxel write failed: {e}")
    else:
        print(f"[CHECKPOINT] v{version} saved to local file + Redis (Blaxel disabled)")

    return str(version)


async def resume(agent_id: str) -> CheckpointPayload | None:
    """
    Load checkpoint from Blaxel sandbox filesystem, or local fallback.
    Called when agent starts.
    """
    raw = None
    if HAS_BLAXEL:
        try:
            import inspect
            # Blaxel SDK might have .get as coroutine
            if hasattr(SandboxInstance, "get") and inspect.iscoroutinefunction(SandboxInstance.get):
                sandbox = await SandboxInstance.get(SANDBOX_NAME)
            else:
                sandbox = SandboxInstance.get(SANDBOX_NAME)
            
            if inspect.iscoroutinefunction(sandbox.fs.read):
                raw = await sandbox.fs.read(CHECKPOINT_PATH)
            else:
                raw = sandbox.fs.read(CHECKPOINT_PATH)
        except Exception as e:
            print(f"[RESUME] No existing checkpoint found in Blaxel: {e}")

    # Fallback to local
    if not raw and os.path.exists(f".checkpoints/{agent_id}_latest.json"):
        with open(f".checkpoints/{agent_id}_latest.json", "r") as f:
            raw = f.read()

    if not raw:
        return None

    payload = CheckpointPayload.model_validate_json(raw)
    await set_working_memory(payload.working)

    print(f"[RESUME] Restored checkpoint v{payload.checkpoint_version}")
    print(f"[RESUME] Task: {payload.working.task}")
    print(f"[RESUME] Progress: {payload.working.progress_pct}%")
    print(f"[RESUME] Last action: {payload.working.last_action}")

    return payload


async def auto_checkpoint_loop(agent_id: str, interval_seconds: int = 30):
    """Run in background — checkpoints every N seconds."""
    while True:
        await asyncio.sleep(interval_seconds)
        await checkpoint(agent_id)
