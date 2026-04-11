import json

from openai import AsyncOpenAI

from .categories import CATEGORY_DECAY_RATES, layer_for
from .models import MemoryEntry

client = AsyncOpenAI()

EXTRACTION_PROMPT = """
You are a memory extraction system for a developer AI agent (Claude Code / Cursor).
Given a conversation turn, extract ATOMIC FACTS worth remembering long-term.

=== CATEGORY DEFINITIONS ===
Classify every fact into exactly ONE of these categories:

  personal_context   — Who the developer is: timezone, team size, role, project stage,
                       hackathon/deadline info, company context.
                       Signal words: "I am", "we are", "I work", "my team", "deadline"

  user_preference    — How the developer likes to work: language/framework choices,
                       code style, tooling preferences, communication style.
                       Signal words: "I prefer", "I always", "I never", "I like", "I hate",
                       "don't use", "instead of", "rather than"

  project_decision   — Architectural or technical choices made with rationale.
                       Signal words: "we decided", "we chose", "we're using X because",
                       "we went with", "we're not using X", "we dropped", "we switched"

  workflow_pattern   — Repeatable commands, scripts, deploy flows, test commands,
                       git workflows, CI/CD steps the developer uses.
                       Signal words: shell commands, "to deploy", "to test", "to run",
                       "the command is", "I use X to"

  codebase_knowledge — Current facts about the code: file structure, bugs known,
                       TODOs, what's implemented vs not, recent changes.
                       Signal words: "the file", "in server.py", "line N", "the function",
                       "currently broken", "not yet implemented", "we have"

  task_context       — What the developer is actively working on RIGHT NOW.
                       Signal words: "I'm working on", "current task", "next step",
                       "today I'm", "right now", "in progress"

  general            — Anything important that doesn't fit above.

=== EXTRACTION RULES ===
- Each fact must be a SINGLE, STANDALONE statement (max 120 chars)
- Preserve rationale when given: "chose Redis over Postgres for low-latency reads"
  is better than "uses Redis"
- Only extract facts that matter BEYOND this single conversation turn
- Ignore small talk, greetings, filler words, and purely transient state
- Source classification:
    user_said       — developer explicitly stated it
    agent_inferred  — you inferred it from context (mark these carefully)
    tool_result     — came from a tool/command output
- Return a JSON object with a "facts" array. NO explanation, NO markdown fences.

=== OUTPUT FORMAT ===
{"facts": [
  {"content": "...", "category": "...", "source": "user_said|agent_inferred|tool_result"},
  ...
]}

=== CONVERSATION TURN ===
{conversation_turn}

=== CURRENT TASK CONTEXT ===
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

        facts = raw.get("facts", []) if isinstance(raw, dict) else raw

        entries = []
        for f in facts:
            if not f.get("content"):
                continue
            category = f.get("category", "general")
            if category not in CATEGORY_DECAY_RATES:
                category = "general"
            decay = CATEGORY_DECAY_RATES[category]
            layer = layer_for(category)
            # task_context category maps to working layer but we store in episodic
            # so it participates in vector search; working memory is set separately
            if layer == "working":
                layer = "episodic"
            entries.append(
                MemoryEntry(
                    agent_id=agent_id,
                    content=f["content"],
                    layer=layer,
                    source=f.get("source", "agent_inferred"),
                    category=category,
                    decay_rate=decay,
                )
            )
        return entries

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Extractor Error] {e}")
        return []
