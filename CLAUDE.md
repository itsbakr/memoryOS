# memoryOS — Claude Code Instructions

> This project IS the memory system. You are working inside memoryOS itself.
> The memoryOS MCP server is connected and running. Use it for ALL memory operations.

---

## CRITICAL: Use memoryOS MCP Tools — NOT File-Based Memory

**Do NOT write to MEMORY.md, user_profile.md, or any markdown memory files.**
**Do NOT read from ~/.claude/projects/*/memory/ files.**

Instead, use these MCP tools which are already connected:

| When you want to... | Use this tool |
|---|---|
| Remember something the developer said | `remember(content, category)` |
| Look up past context | `recall(query)` |
| Know who you're talking to | `get_my_profile()` |
| Know the project state | `get_project_context()` |
| Track current work | `update_task(task, progress_pct)` |

**Why memoryOS is better than file memory:**
- Temporal confidence decay — stale facts are automatically down-weighted
- Contradiction detection — you'll be warned before overwriting conflicting facts
- Vector semantic search — retrieve by meaning, not keyword
- Visible in the live dashboard at http://localhost:8000

---

## Session Start Protocol

At the start of EVERY new conversation, run these two tools first:

1. `get_my_profile()` — load who the developer is and their preferences
2. `get_project_context()` — load project decisions and codebase state

Only after loading context should you respond. If both return empty, ask the
developer to introduce themselves and then call `remember()` with what they share.

---

## Memory Rules

- **Names, identity, background** → `remember(..., category="personal_context")`
- **Coding style, tool preferences** → `remember(..., category="user_preference")`
- **Architecture / tech decisions** → `remember(..., category="project_decision")`
- **Deploy scripts, test commands** → `remember(..., category="workflow_pattern")`
- **Current file/code state** → `remember(..., category="codebase_knowledge")`
- **Active task right now** → `update_task(task=..., progress_pct=...)`

Always use `source="user_said"` when the developer told you directly.

---

## About This Project

**memoryOS** is a persistent memory layer for AI coding agents built by Ahmed Bakr.

Stack: FastAPI + Redis (RedisVL vector search) + OpenAI embeddings + React/Vite dashboard.

Key features:
- Three memory layers: working, episodic, semantic
- Temporal confidence decay (exponential per category)
- Contradiction detection with user resolution
- Blaxel sandbox checkpoints for session recovery
- MCP server exposing memory tools to Claude Code (this file's purpose)

Backend runs at: `http://localhost:8000`
Dashboard: `http://localhost:8000` (React frontend in `frontend/dist`)
Start: `uvicorn api.server:app --reload --port 8000 --env-file .env`
