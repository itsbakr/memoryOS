import asyncio
import json

from openai import AsyncOpenAI
from sandbox.checkpoint import auto_checkpoint_loop, checkpoint, resume
from memory.episodic import retrieve_memories
from memory.working import get_working_memory
from memory.extractor import extract_facts
from agent.tools import MEMORY_TOOLS, handle_tool_call

openai_client = AsyncOpenAI()

SYSTEM_PROMPT = """You are a long-running AI agent with persistent memory.

You have access to memory tools:
- store_memory: save important facts before you forget them
- retrieve_memory: search your memory for relevant context
- update_task_progress: track where you are in a task
- resolve_contradiction: resolve conflicting information

CRITICAL RULES:
1. ALWAYS call retrieve_memory at the start of each task to load relevant context
2. ALWAYS call store_memory when the user shares important information
3. ALWAYS call update_task_progress when you start a new conversation topic, or complete a subtask. Treat the user's current goal as the "task".
4. If store_memory returns contradiction_detected, STOP and ask the user
   to clarify before proceeding. Show them both facts. Do NOT guess.
5. You persist across sessions. When you start, you may already have
   context from previous sessions — always check.

Be direct. Tell the user when you're remembering something, when you're
detecting a conflict, and when you're resuming from a previous session."""


async def run_agent(
    agent_id: str, user_message: str, previous_response_id: str | None = None
):
    """
    Single turn of the Codex agent loop.
    Returns (response_text, response_id, contradiction_event_or_None)
    """

    # Auto-extract facts on user input (for deterministic capture outside of tool calls)
    working = await get_working_memory(agent_id)
    task_context = working.task if working else "no active task"
    extracted_facts = await extract_facts(user_message, task_context, agent_id)
    for fact in extracted_facts:
        from memory.episodic import add_memory

        await add_memory(fact)

    # Inject relevant memory context into the message
    memories = await retrieve_memories(user_message, agent_id, k=5)
    memory_context = ""
    if memories:
        memory_lines = "\\n".join(
            [f"- [{m.source}, conf={m.confidence:.2f}] {m.content}" for m in memories]
        )
        memory_context = f"\\n\\n[MEMORY CONTEXT]\\n{memory_lines}"

    working_context = ""
    if working:
        working_context = f"\\n\\n[CURRENT TASK]\\n{working.task} ({working.progress_pct}% complete)\\nLast action: {working.last_action}"

    enriched_message = user_message + memory_context + working_context

    response = await openai_client.responses.create(
        model="gpt-4o-mini",  # Fallback model; Codex-mini-latest if available
        instructions=SYSTEM_PROMPT,
        input=enriched_message,
        tools=MEMORY_TOOLS,
        previous_response_id=previous_response_id,
    )

    contradiction_event = None

    while any(getattr(item, "type", None) == "function_call" for item in response.output):
        tool_outputs = []

        for item in response.output:
            if getattr(item, "type", None) == "function_call":
                tc = item
                args = json.loads(tc.arguments)
                result = await handle_tool_call(tc.name, args, agent_id)

                if result.get("status") == "contradiction_detected":
                    contradiction_event = result

                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": json.dumps(result)
                })

        response = await openai_client.responses.create(
            model="gpt-4o-mini",
            instructions=SYSTEM_PROMPT,
            input=tool_outputs,
            tools=MEMORY_TOOLS,
            previous_response_id=response.id
        )

    output_text = ""
    for item in response.output:
        if getattr(item, "type", None) == "message" and getattr(item, "content", None):
            output_text += item.content[0].text

    return output_text, response.id, contradiction_event


async def interactive_session(agent_id: str = "demo-agent"):
    """Full interactive session with checkpoint loop."""

    existing = await resume(agent_id)
    if existing:
        print(f"\\n[SYSTEM] Resuming from checkpoint v{existing.checkpoint_version}")
        print(f"[SYSTEM] Previous task: {existing.working.task}")
        print(f"[SYSTEM] Progress: {existing.working.progress_pct}%\\n")

    # Start background checkpoint loop
    asyncio.create_task(auto_checkpoint_loop(agent_id, interval_seconds=30))

    previous_response_id = None
    print("Temporal Memory OS — Type 'quit' to exit, 'memory' to see stats\\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            await checkpoint(agent_id)
            break
        if user_input.lower() == "memory":
            memories = await retrieve_memories(
                "everything", agent_id, k=20, min_confidence=0.0
            )
            print(f"\\n[MEMORY DUMP] {len(memories)} memories:")
            for m in memories:
                print(f"  [{m.confidence:.2f}] {m.content}")
            print()
            continue

        response, previous_response_id, contradiction = await run_agent(
            agent_id, user_input, previous_response_id
        )

        if contradiction:
            print("\\n[CONTRADICTION DETECTED]")
            print(f"  New: {contradiction['new_fact']}")
            print(f"  Conflicts with: {contradiction['conflicts_with']}")
            print(f"  {contradiction['explanation']}")
            print("  Which is correct? (type 'new' or 'old')")
            choice = input(">>> ").strip().lower()
            chosen = (
                contradiction["new_fact"]
                if choice == "new"
                else contradiction["conflicts_with"]
            )
            from memory.contradiction import resolve_contradiction

            await resolve_contradiction(
                contradiction["contradiction_id"], chosen, agent_id
            )
            print(f"[SYSTEM] Resolved. Storing: '{chosen}'\\n")

        print(f"\\nAgent: {response}\\n")
