from memory.contradiction import check_contradiction, resolve_contradiction
from memory.episodic import add_memory, retrieve_memories
from memory.models import MemoryEntry, WorkingMemory
from memory.working import set_working_memory
import time

# Tool definitions in OpenAI format
MEMORY_TOOLS = [
    {
        "type": "function",
        "name": "store_memory",
        "description": "Store a new fact in episodic memory. Call this when the user says something important or when you learn something worth remembering.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The atomic fact to store (max 150 chars)",
                },
                "source": {
                    "type": "string",
                    "enum": ["user_said", "agent_inferred", "tool_result"],
                },
            },
            "required": ["content", "source"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "retrieve_memory",
        "description": "Search episodic memory for relevant facts. Call this before answering questions about past context.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What you're looking for",
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum confidence threshold (0.0-1.0). Default 0.3. Use 0.0 to see stale memories.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "update_task_progress",
        "description": "Update working memory with current task state. Call after completing each subtask.",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "subtask": {"type": "string"},
                "progress_pct": {"type": "number"},
                "last_action": {"type": "string"},
            },
            "required": ["task", "progress_pct", "last_action"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "resolve_contradiction",
        "description": "Called when a contradiction has been surfaced and user has chosen.",
        "parameters": {
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
                "chosen_fact": {"type": "string"},
            },
            "required": ["event_id", "chosen_fact"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "run_sandbox_command",
        "description": "Execute a shell command inside an isolated Blaxel Sandbox environment. Use this to run scripts, fetch URLs via curl, or perform real-world actions for the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                }
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
]


async def handle_tool_call(tool_name: str, tool_args: dict, agent_id: str) -> dict:
    """Dispatch tool calls from Codex to memory functions."""

    if tool_name == "store_memory":
        # FIRST: check for contradiction before storing
        contradiction = await check_contradiction(tool_args["content"], agent_id)
        if contradiction:
            return {
                "status": "contradiction_detected",
                "contradiction_id": contradiction.id,
                "new_fact": contradiction.new_fact,
                "conflicts_with": contradiction.conflicting_memory_content,
                "confidence_score": contradiction.confidence_score,
                "explanation": contradiction.explanation,
                "action_required": "Ask user which is correct before storing.",
            }

        mem = MemoryEntry(
            agent_id=agent_id,
            content=tool_args["content"],
            layer="episodic",
            source=tool_args["source"],
        )
        mem_id = await add_memory(mem)
        return {"status": "stored", "memory_id": mem_id}

    elif tool_name == "retrieve_memory":
        memories = await retrieve_memories(
            query=tool_args["query"],
            agent_id=agent_id,
            min_confidence=tool_args.get("min_confidence", 0.3),
        )
        return {
            "memories": [
                {
                    "content": m.content,
                    "source": m.source,
                    "confidence": round(m.confidence, 2),
                    "age_hours": round((time.time() - m.created_at) / 3600, 1),
                }
                for m in memories
            ],
            "count": len(memories),
        }

    elif tool_name == "update_task_progress":
        working = WorkingMemory(agent_id=agent_id, **tool_args)
        await set_working_memory(working)
        return {"status": "updated"}

    elif tool_name == "resolve_contradiction":
        await resolve_contradiction(
            tool_args["event_id"], tool_args["chosen_fact"], agent_id
        )
        return {"status": "resolved"}

    elif tool_name == "run_sandbox_command":
        from sandbox.checkpoint import get_or_create_sandbox
        try:
            sandbox = await get_or_create_sandbox()
            if not sandbox:
                return {"error": "Blaxel sandbox not configured in this environment"}
            
            # Process exec might be synchronous or async depending on the SDK version, check safely
            import inspect
            process = sandbox.process.exec({"command": tool_args["command"]})
            if inspect.iscoroutine(process):
                process = await process
                
            wait_call = sandbox.process.wait(process.pid, max_wait=15000, interval=500)
            if inspect.iscoroutine(wait_call):
                process_result = await wait_call
            else:
                process_result = wait_call
                
            return {
                "status": process_result.status,
                "stdout": process_result.stdout,
                "stderr": process_result.stderr
            }
        except Exception as e:
            return {"error": str(e)}

    return {"error": f"Unknown tool: {tool_name}"}
