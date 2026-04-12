from __future__ import annotations

import re
from typing import Iterable, Literal, Optional

from .categories import CATEGORY_DECAY_RATES, layer_for
from .contradiction import check_contradiction
from .episodic import add_memory
from .graph import add_fact_to_graph
from .models import MemoryEntry
from .policies import retention_expires_at
from .audit import log_event

MAX_CONTENT_LEN = 200
WHITESPACE_RE = re.compile(r"\s+")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")


def _contains_sensitive(content: str) -> bool:
    return bool(EMAIL_RE.search(content) or PHONE_RE.search(content))

ConflictPolicy = Literal["skip", "surface", "store"]


def _normalize_content(content: str) -> str:
    normalized = WHITESPACE_RE.sub(" ", content.strip())
    if len(normalized) > MAX_CONTENT_LEN:
        normalized = normalized[:MAX_CONTENT_LEN].rstrip()
    return normalized


def _normalize_entry(entry: MemoryEntry) -> Optional[MemoryEntry]:
    if not entry.content or not entry.content.strip():
        return None

    category = entry.category or "general"
    if category not in CATEGORY_DECAY_RATES:
        category = "general"

    decay_rate = CATEGORY_DECAY_RATES[category]
    layer = layer_for(category)
    if layer == "working":
        layer = "episodic"
    expires_at = retention_expires_at(category, entry.created_at)

    return entry.model_copy(
        update={
            "content": _normalize_content(entry.content),
            "category": category,
            "decay_rate": decay_rate,
            "layer": layer,
            "expires_at": expires_at,
        }
    )


async def write_memory_entries(
    entries: Iterable[MemoryEntry],
    *,
    conflict_policy: ConflictPolicy = "skip",
) -> list[dict]:
    """
    Centralized, deterministic write gate for all memory writes.
    Returns structured results for auditing and UI.
    """
    results: list[dict] = []

    for entry in entries:
        normalized = _normalize_entry(entry)
        if not normalized:
            results.append(
                {
                    "status": "skipped",
                    "reason": "empty_content",
                    "content": entry.content,
                }
            )
            await log_event(
                entry.agent_id,
                "memory_skipped",
                {"reason": "empty_content"},
            )
            continue
        if _contains_sensitive(normalized.content):
            results.append(
                {
                    "status": "skipped",
                    "reason": "sensitive_content",
                    "content": normalized.content,
                }
            )
            await log_event(
                normalized.agent_id,
                "memory_skipped",
                {"reason": "sensitive_content", "content": normalized.content},
            )
            continue

        contradiction = await check_contradiction(normalized.content, normalized.agent_id)
        if contradiction:
            payload = {
                "status": "contradiction_detected",
                "reason": "conflict_check",
                "contradiction_id": contradiction.id,
                "new_fact": contradiction.new_fact,
                "conflicts_with": contradiction.conflicting_memory_content,
                "confidence_score": contradiction.confidence_score,
                "explanation": contradiction.explanation,
            }
            results.append(payload)
            await log_event(
                normalized.agent_id,
                "memory_conflict",
                {
                    "contradiction_id": contradiction.id,
                    "new_fact": contradiction.new_fact,
                    "conflicts_with": contradiction.conflicting_memory_content,
                },
            )
            if conflict_policy in ("skip", "surface"):
                continue

        memory_id = await add_memory(normalized)
        try:
            await add_fact_to_graph(normalized.agent_id, memory_id, normalized.content)
        except Exception:
            pass
        await log_event(
            normalized.agent_id,
            "memory_stored",
            {
                "memory_id": memory_id,
                "category": normalized.category or "general",
            },
        )
        results.append(
            {
                "status": "stored",
                "memory_id": memory_id,
                "content": normalized.content,
                "category": normalized.category or "general",
            }
        )

    return results
