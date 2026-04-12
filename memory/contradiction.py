import json
import os
import time

import redis.asyncio as aioredis
from openai import AsyncOpenAI

from .decay import reinforce_memory
from .audit import log_event
from .episodic import (
    add_memory,
    get_memory_by_id,
    retrieve_memories,
    update_memory_fields,
)
from .models import ContradictionEvent, MemoryEntry

openai_client = AsyncOpenAI()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

CONTRADICTION_PROMPT = """
You are a contradiction detector for an AI agent's memory system.

New fact being considered for storage:
"{new_fact}"

Existing memories that might conflict:
{existing_memories}

Task: Determine if the new fact DIRECTLY CONTRADICTS any existing memory.
A contradiction means both CANNOT be true simultaneously.
(e.g., "meeting is Thursday" vs "meeting is Wednesday" = contradiction)
(e.g., "user likes Python" vs "user started learning Rust" = NOT contradiction)

Return JSON only:
{{
  "is_contradiction": true/false,
  "confidence": 0.0-1.0,
  "conflicting_memory_index": 0,
  "explanation": "brief explanation"
}}
"""


async def check_contradiction(
    new_fact: str, agent_id: str
) -> ContradictionEvent | None:
    """
    Returns a ContradictionEvent if detected, None otherwise.
    Threshold: 0.75 confidence to surface to user.
    """
    # Search including low-confidence memories (they could still conflict)
    similar = await retrieve_memories(
        new_fact,
        agent_id,
        k=5,
        min_confidence=0.0,
        active_only=True,
    )

    if not similar:
        return None

    existing_str = "\\n".join(
        [
            f"{i}. [{m.source}] {m.content} (confidence: {m.confidence:.2f})"
            for i, m in enumerate(similar)
        ]
    )

    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": CONTRADICTION_PROMPT.format(
                    new_fact=new_fact,
                    existing_memories=existing_str,
                ),
            }
        ],
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content or "{}")

    if not result.get("is_contradiction"):
        return None

    conf = float(result.get("confidence", 0))
    if conf < 0.75:
        return None

    idx = result.get("conflicting_memory_index")
    conflicting = similar[idx] if idx is not None and idx < len(similar) else similar[0]

    event = ContradictionEvent(
        agent_id=agent_id,
        new_fact=new_fact,
        conflicting_memory_id=conflicting.id,
        conflicting_memory_content=conflicting.content,
        confidence_score=conf,
        explanation=result.get("explanation", ""),
    )

    # Store in Redis for dashboard + API resolution
    from memory.working import get_redis
    r = await get_redis()
    await r.setex(
        f"contradiction:{event.id}",
        3600,  # 1 hour TTL
        event.model_dump_json(),
    )
    await r.lpush(f"agent:{agent_id}:contradictions", event.id)
    await r.ltrim(f"agent:{agent_id}:contradictions", 0, 49)  # Keep last 50

    return event


async def resolve_contradiction(event_id: str, chosen_fact: str, agent_id: str) -> None:
    """User or auto-resolver picks which fact wins. Overwrites the losing memory's confidence to 0."""
    from memory.working import get_redis
    r = await get_redis()
    raw = await r.get(f"contradiction:{event_id}")
    if not raw:
        return

    event = ContradictionEvent.model_validate_json(raw)
    if event.resolution != "pending":
        return

    event.resolution = "user_resolved"
    event.chosen_fact = chosen_fact
    event.resolved_at = time.time()

    conflicting = await get_memory_by_id(event.conflicting_memory_id)
    now = time.time()

    if chosen_fact == event.new_fact:
        next_version = (conflicting.version + 1) if conflicting else 1
        new_entry = MemoryEntry(
            agent_id=agent_id,
            content=event.new_fact,
            layer="episodic",
            source="user_said",
            category=conflicting.category if conflicting else "general",
            decay_rate=conflicting.decay_rate if conflicting else 0.05,
            supersedes_id=event.conflicting_memory_id,
            version=next_version,
            valid_from=now,
        )
        new_id = await add_memory(new_entry)
        event.chosen_memory_id = new_id
        event.superseded_memory_id = event.conflicting_memory_id

        if conflicting:
            await update_memory_fields(
                event.conflicting_memory_id,
                {
                    "is_active": 0,
                    "valid_to": now,
                    "superseded_by_id": new_id,
                    "confidence": 0.0,
                },
            )
        await log_event(
            agent_id,
            "contradiction_resolved",
            {"event_id": event.id, "chosen": "new", "memory_id": new_id},
        )
    else:
        if conflicting:
            reinforce_memory(conflicting)
            await update_memory_fields(
                event.conflicting_memory_id,
                {
                    "last_reinforced": now,
                    "confidence": conflicting.confidence,
                },
            )
            event.chosen_memory_id = event.conflicting_memory_id
        await log_event(
            agent_id,
            "contradiction_resolved",
            {"event_id": event.id, "chosen": "old", "memory_id": event.conflicting_memory_id},
        )

    await r.set(f"contradiction:{event_id}", event.model_dump_json())
