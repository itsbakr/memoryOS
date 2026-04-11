import json

from openai import AsyncOpenAI

from .decay import DECAY_RATES
from .models import MemoryEntry

client = AsyncOpenAI()

EXTRACTION_PROMPT = """
You are a memory extraction system for an AI agent.
Given a conversation turn, extract ATOMIC FACTS worth remembering.

Rules:
- Each fact must be a single, standalone statement (max 100 chars)
- Only extract facts that matter beyond this conversation turn
- Label each fact with source: user_said | agent_inferred | tool_result
- Ignore small talk, filler, and transient state
- Return JSON object ONLY, with a "facts" array. No explanation.

Output format EXACTLY like this:
{"facts": [{"content": "User prefers morning meetings", "source": "user_said"}]}

Conversation turn:
{conversation_turn}

Current task context:
{task_context}
"""


async def extract_facts(
    conversation_turn: str, task_context: str, agent_id: str
) -> list[MemoryEntry]:
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": EXTRACTION_PROMPT.format(
                        conversation_turn=conversation_turn,
                        task_context=task_context,
                    ),
                }
            ],
            response_format={"type": "json_object"},
        )

        content = (response.choices[0].message.content or "{}").strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            raw = json.loads(content)
        except json.JSONDecodeError:
            return []

        if isinstance(raw, list):
            facts = raw
        else:
            facts = raw.get("facts", [])

        return [
            MemoryEntry(
                agent_id=agent_id,
                content=f["content"],
                layer="episodic",
                source=f["source"],
                decay_rate=DECAY_RATES.get(f["source"], 0.05),
            )
            for f in facts
            if f.get("content")
        ]
    except Exception as e:
        print(f"[Extractor Error] {e}")
        return []
