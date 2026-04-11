import os

import redis.asyncio as aioredis

from .models import WorkingMemory

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
WORKING_MEMORY_TTL = 86400  # 24 hours


async def get_redis():
    return aioredis.from_url(REDIS_URL, decode_responses=True)


def _key(agent_id: str) -> str:
    return f"agent:{agent_id}:working"


async def set_working_memory(working: WorkingMemory) -> None:
    r = await get_redis()
    # Redis hash cannot store None or nested objects easily without encoding
    mapping = {k: str(v) for k, v in working.model_dump(exclude_none=True).items()}
    await r.hset(_key(working.agent_id), mapping=mapping)
    await r.expire(_key(working.agent_id), WORKING_MEMORY_TTL)
    await r.aclose()


async def get_working_memory(agent_id: str) -> WorkingMemory | None:
    r = await get_redis()
    data = await r.hgetall(_key(agent_id))
    await r.aclose()
    if not data:
        return None
    return WorkingMemory(**data)


async def increment_tool_calls(agent_id: str) -> None:
    r = await get_redis()
    await r.hincrby(_key(agent_id), "tool_calls_made", 1)
    await r.aclose()


async def clear_working_memory(agent_id: str) -> None:
    r = await get_redis()
    await r.delete(_key(agent_id))
    await r.aclose()
