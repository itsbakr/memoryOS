from __future__ import annotations

from .categories import CATEGORY_DECAY_RATES

# Retention TTLs in hours (None = no expiry)
RETENTION_TTL_HOURS: dict[str, float | None] = {
    "personal_context": None,
    "user_preference": 24 * 180,  # ~6 months
    "project_decision": 24 * 365,  # 1 year
    "workflow_pattern": 24 * 120,  # 4 months
    "codebase_knowledge": 24 * 30,  # 1 month
    "task_context": 24 * 3,  # 3 days
    "general": 24 * 90,  # 3 months
}


def retention_ttl_hours(category: str) -> float | None:
    if category in RETENTION_TTL_HOURS:
        return RETENTION_TTL_HOURS[category]
    return RETENTION_TTL_HOURS.get("general")


def retention_expires_at(category: str, created_at: float) -> float | None:
    ttl = retention_ttl_hours(category)
    if ttl is None:
        return None
    return created_at + (ttl * 3600)
