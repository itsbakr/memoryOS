#!/usr/bin/env bash
# post-conversation.sh
# Triggered by Cursor's stop event (agent completes a task).
# Syncs CLAUDE.md so memories accumulated during this session are
# immediately available in the next one.
#
# Also calls /api/context/ingest if LAST_CONVERSATION_FILE is set,
# allowing external tools to pass a transcript path for bulk extraction.
#
# Fails open: errors do not block anything.

set -euo pipefail

input=$(cat)

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SYNC_SCRIPT="$SCRIPT_DIR/scripts/sync_claude_md.py"

MEMORY_OS_URL="${MEMORY_OS_URL:-http://localhost:8000}"
MEMORY_OS_AGENT_ID="${MEMORY_OS_AGENT_ID:-claude-code}"

# Optional: if a transcript file path was exported, ingest it first
if [ -n "${LAST_CONVERSATION_FILE:-}" ] && [ -f "$LAST_CONVERSATION_FILE" ]; then
    TRANSCRIPT=$(cat "$LAST_CONVERSATION_FILE")
    PAYLOAD=$(python3 -c "
import json, sys
transcript = sys.stdin.read()
print(json.dumps({
    'agent_id': '$MEMORY_OS_AGENT_ID',
    'transcript': transcript,
    'task_context': 'post-session harvest'
}))
" <<< "$TRANSCRIPT" 2>/dev/null || echo "")

    if [ -n "$PAYLOAD" ]; then
        curl -s -X POST \
            -H "Content-Type: application/json" \
            -d "$PAYLOAD" \
            "$MEMORY_OS_URL/api/context/ingest" \
            >/tmp/memoryos-ingest.log 2>&1 || true
    fi
fi

# Sync CLAUDE.md with latest snapshot
if command -v python3 &>/dev/null && [ -f "$SYNC_SCRIPT" ]; then
    python3 "$SYNC_SCRIPT" \
        --url "$MEMORY_OS_URL" \
        --agent-id "$MEMORY_OS_AGENT_ID" \
        --project "$SCRIPT_DIR" \
        >/tmp/memoryos-post-sync.log 2>&1 || true
fi

# No output needed for stop event — just exit cleanly
exit 0
