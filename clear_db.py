import asyncio
import redis.asyncio as aioredis
import os
from dotenv import load_dotenv

load_dotenv()

async def clear():
    r = aioredis.from_url(os.getenv("REDIS_URL"))
    await r.flushdb()
    await r.aclose()
    print("Database flushed!")

asyncio.run(clear())
