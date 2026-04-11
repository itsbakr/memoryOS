#!/usr/bin/env python3
"""
memoryOS MCP Server
-------------------
Exposes memoryOS as an MCP (Model Context Protocol) server over stdin/stdout
so Claude Code agents can actively read and write persistent developer memory.

Usage
-----
Add to your Claude Code MCP config (~/.claude/settings.json or project
.claude/settings.json):

  {
    "mcpServers": {
      "memoryOS": {
        "command": "python",
        "args": ["/path/to/memoryOS/mcp_server.py"],
        "env": {
          "MEMORY_OS_URL": "http://localhost:8000",
          "MEMORY_OS_AGENT_ID": "claude-code"
        }
      }
    }
  }

Tools exposed
-------------
  remember          — store a fact with a developer category
  recall            — semantic search over stored memories
  get_my_profile    — user preferences + personal context
  get_project_context — project decisions + codebase knowledge
  update_task       — set current working task in working memory

The server proxies all calls to the memoryOS FastAPI backend, so the backend
must be running (uvicorn api.server:app --port 8000) for tools to work.
"""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

MEMORY_OS_URL = os.getenv("MEMORY_OS_URL", "http://localhost:8000")
AGENT_ID = os.getenv("MEMORY_OS_AGENT_ID", "claude-code")

# ---------------------------------------------------------------------------
# MCP protocol helpers
# ---------------------------------------------------------------------------

def _write(obj: dict) -> None:
    line = json.dumps(obj)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _read() -> dict | None:
    line = sys.stdin.readline()
    if not line:
        return None
    return json.loads(line.strip())


def _http_get(path: str, params: dict | None = None) -> dict:
    url = MEMORY_OS_URL + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        return {"error": str(e)}


def _http_post(path: str, body: dict) -> dict:
    url = MEMORY_OS_URL + path
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "remember",
        "description": (
            "PRIMARY MEMORY TOOL — use this instead of writing files. "
            "Stores one or more facts in memoryOS with temporal decay and vector search. "
            "Call this whenever the developer shares their name, preferences, project info, "
            "decisions, workflow patterns, or anything worth keeping across sessions. "
            "If a message contains multiple distinct pieces of information, break them down "
            "and supply them as separate items in the facts array. "
            "NEVER write to MEMORY.md or user_profile.md — use this tool instead."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "description": "An array of 1 or more atomic facts to store.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The atomic fact to remember (max 120 chars). Be specific and self-contained.",
                            },
                            "category": {
                                "type": "string",
                                "enum": [
                                    "personal_context",
                                    "user_preference",
                                    "project_decision",
                                    "workflow_pattern",
                                    "codebase_knowledge",
                                    "task_context",
                                    "general",
                                ],
                                "description": (
                                    "personal_context: who the developer is (timezone, team, role). "
                                    "user_preference: how they like to work (frameworks, code style). "
                                    "project_decision: architectural choices and rationale. "
                                    "workflow_pattern: commands, scripts, deploy flows. "
                                    "codebase_knowledge: current state of the codebase. "
                                    "task_context: what they're actively working on right now. "
                                    "general: anything else important."
                                ),
                            },
                            "source": {
                                "type": "string",
                                "enum": ["user_said", "agent_inferred", "tool_result"],
                                "description": "How this fact was obtained.",
                                "default": "user_said",
                            }
                        },
                        "required": ["content", "category"]
                    }
                }
            },
            "required": ["facts"],
        },
    },
    {
        "name": "recall",
        "description": (
            "PRIMARY RECALL TOOL — use this instead of reading MEMORY.md files. "
            "Semantic vector search over all stored memories with live confidence scores. "
            "Call this at the START of every session and whenever you need context. "
            "NEVER read MEMORY.md or user_profile.md — use this tool instead."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query — what are you trying to remember?",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of results (default 8, max 20).",
                    "default": 8,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_my_profile",
        "description": (
            "Load the developer's full persistent profile from memoryOS: who they are, "
            "their coding preferences, communication style, and personal context. "
            "ALWAYS call this at the start of every new session BEFORE asking who the user is. "
            "This replaces reading user_profile.md — the data here includes confidence scores "
            "and temporal decay so stale facts are automatically down-weighted."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_project_context",
        "description": (
            "Retrieve all known project decisions, architecture choices, and "
            "current codebase knowledge. Call when starting work on a project "
            "to avoid re-explaining context."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "update_task",
        "description": (
            "Update the current working task in memory. Call when starting a new "
            "subtask or completing one so context survives session restarts."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Short description of the current task.",
                },
                "progress_pct": {
                    "type": "number",
                    "description": "Completion percentage 0-100.",
                    "default": 0,
                },
                "last_action": {
                    "type": "string",
                    "description": "What was the last action taken?",
                },
            },
            "required": ["task"],
        },
    },
]


def _tool_remember(args: dict) -> str:
    facts = args.get("facts", [])
    if not facts:
        return "Error: facts array is empty."
        
    results = []
    success_count = 0
    
    for f in facts:
        content = f.get("content")
        if not content:
            continue
            
        body = {
            "agent_id": AGENT_ID,
            "content": content,
            "category": f.get("category", "general"),
            "source": f.get("source", "user_said"),
        }
        resp = _http_post("/api/memory/store", body)
        
        if "error" in resp:
            results.append(f"Failed to store: '{content[:30]}...' ({resp['error']})")
        else:
            success_count += 1
            cat = resp.get("category", "general")
            results.append(f"Stored [{cat}]: {content}")
            
    summary = f"Remembered {success_count}/{len(facts)} facts.\n"
    return summary + "\n".join(results)


def _tool_recall(args: dict) -> str:
    query = args["query"]
    limit = min(int(args.get("limit", 8)), 20)
    resp = _http_get("/api/memory/search", {"agent_id": AGENT_ID, "query": query, "limit": limit})
    if "error" in resp:
        return f"Error searching memory: {resp['error']}"
    results = resp.get("results", [])
    if not results:
        return "No relevant memories found."
    lines = []
    for m in results:
        conf = m.get("confidence", 0)
        cat = m.get("category", "")
        age = m.get("age_hours", 0)
        age_str = f"{age:.0f}h ago" if age >= 1 else "< 1h ago"
        lines.append(f"- (conf={conf:.2f}, {age_str} [{cat}]) {m['content']}")
    return "\n".join(lines)


def _tool_get_my_profile(args: dict) -> str:
    resp = _http_get("/api/context/snapshot", {"agent_id": AGENT_ID})
    if "error" in resp:
        return f"Error fetching profile: {resp['error']}"
    profile = resp.get("profile", [])
    working = resp.get("working_memory")
    lines = ["=== Developer Profile ==="]
    if not profile:
        lines.append("No profile memories stored yet.")
    for m in profile:
        lines.append(f"- [{m['category']}] {m['content']}  (conf={m['confidence']:.2f})")
    if working and working.get("task"):
        lines.append(f"\nCurrent task: {working['task']} ({working.get('progress_pct', 0):.0f}% complete)")
        if working.get("last_action"):
            lines.append(f"Last action: {working['last_action']}")
    return "\n".join(lines)


def _tool_get_project_context(args: dict) -> str:
    resp = _http_get("/api/context/snapshot", {"agent_id": AGENT_ID})
    if "error" in resp:
        return f"Error fetching project context: {resp['error']}"
    project = resp.get("project", [])
    workflow = resp.get("workflow", [])
    lines = ["=== Project Context ==="]
    if not project and not workflow:
        lines.append("No project context stored yet.")
    if project:
        lines.append("\nDecisions & Architecture:")
        for m in project:
            lines.append(f"- [{m['category']}] {m['content']}  (conf={m['confidence']:.2f}, {m['age_hours']:.0f}h ago)")
    if workflow:
        lines.append("\nWorkflow Patterns:")
        for m in workflow:
            lines.append(f"- {m['content']}  (conf={m['confidence']:.2f})")
    return "\n".join(lines)


def _tool_update_task(args: dict) -> str:
    body = {
        "agent_id": AGENT_ID,
        "task": args["task"],
        "progress_pct": float(args.get("progress_pct", 0)),
        "last_action": args.get("last_action", ""),
    }
    resp = _http_post("/api/memory/working", body)
    if "error" in resp:
        # Store as a task_context memory instead as fallback
        ingest = _http_post("/api/context/ingest", {
            "agent_id": AGENT_ID,
            "transcript": f"Current task: {args['task']}",
            "task_context": "task_context",
        })
        return f"Task noted (via memory): {args['task']}"
    return f"Task updated: {args['task']} ({args.get('progress_pct', 0):.0f}% complete)"


TOOL_HANDLERS = {
    "remember": _tool_remember,
    "recall": _tool_recall,
    "get_my_profile": _tool_get_my_profile,
    "get_project_context": _tool_get_project_context,
    "update_task": _tool_update_task,
}


# ---------------------------------------------------------------------------
# MCP message dispatch
# ---------------------------------------------------------------------------

def handle_message(msg: dict) -> dict | None:
    method = msg.get("method", "")
    msg_id = msg.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "memoryOS",
                    "version": "1.0.0",
                },
                "capabilities": {
                    "tools": {},
                },
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": TOOLS},
        }

    if method == "tools/call":
        params = msg.get("params", {})
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        handler = TOOL_HANDLERS.get(tool_name)
        if handler is None:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }

        try:
            result_text = handler(tool_args)
        except Exception as e:
            result_text = f"Tool error: {e}"

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [{"type": "text", "text": result_text}],
                "isError": False,
            },
        }

    if method == "notifications/initialized":
        return None  # no response needed for notifications

    # Unknown method — return standard error
    if msg_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    while True:
        try:
            msg = _read()
        except (EOFError, json.JSONDecodeError):
            break
        if msg is None:
            break
        response = handle_message(msg)
        if response is not None:
            _write(response)


if __name__ == "__main__":
    main()
