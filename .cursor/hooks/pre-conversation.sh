#!/usr/bin/env bash
# pre-conversation.sh
# Triggered by Cursor's sessionStart event.
# Refreshes CLAUDE.md with the latest memory snapshot so Claude always
# opens each session with up-to-date context — no manual re-explaining needed.
#
# Fails open: if the backend is down the session still starts normally.

set -euo pipefail

# Read stdin (hook input JSON) — we don't use it but must consume it
input=$(cat)

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SYNC_SCRIPT="$SCRIPT_DIR/scripts/sync_claude_md.py"

MEMORY_OS_URL="${MEMORY_OS_URL:-http://localhost:8000}"
MEMORY_OS_AGENT_ID="${MEMORY_OS_AGENT_ID:-claude-code}"

# Run sync in background so it doesn't block session startup
if command -v python3 &>/dev/null && [ -f "$SYNC_SCRIPT" ]; then
    python3 "$SYNC_SCRIPT" \
        --url "$MEMORY_OS_URL" \
        --agent-id "$MEMORY_OS_AGENT_ID" \
        --project "$SCRIPT_DIR" \
        >/tmp/memoryos-pre-sync.log 2>&1 &
fi

# Always allow the session to proceed
echo '{"permission": "allow"}'
exit 0
