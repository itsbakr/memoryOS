from __future__ import annotations

from typing import Literal, Optional
from uuid import uuid4
import time

from pydantic import BaseModel, Field

try:
    from redisvl.schema import IndexSchema
except Exception:  # pragma: no cover - handled in tests via dependency install
    IndexSchema = None


class MemoryEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    content: str  # atomic extracted fact, max 200 chars
    layer: Literal["working", "episodic", "semantic"]
    confidence: float = 1.0  # 0.0-1.0, starts at 1.0
    source: Literal["user_said", "agent_inferred", "tool_result"]
    # Developer memory category (see memory/categories.py for full ontology).
    # None means legacy/uncategorised entry — treated as "general".
    category: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    last_reinforced: float = Field(default_factory=time.time)
    # DECAY RATES (per hour):
    # personal_context: 0.0005 (permanent — identity facts)
    # user_preference:  0.001  (very slow — weeks to decay meaningfully)
    # project_decision: 0.002  (slow — months)
    # workflow_pattern: 0.005  (moderate — weeks)
    # codebase_knowledge: 0.01 (days — code changes)
    # task_context:     0.1    (fast — hours, current work)
    decay_rate: float = 0.05
    embedding: Optional[list[float]] = None


class WorkingMemory(BaseModel):
    agent_id: str
    task: str
    subtask: Optional[str] = None
    progress_pct: float = 0.0
    last_action: Optional[str] = None
    last_checkpoint: float = Field(default_factory=time.time)
    tool_calls_made: int = 0
    context_summary: Optional[str] = None


class ContradictionEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    new_fact: str
    conflicting_memory_id: str
    conflicting_memory_content: str
    confidence_score: float  # how confident we are it's a contradiction
    explanation: str
    resolution: Literal["pending", "user_resolved", "auto_resolved"] = "pending"
    chosen_fact: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    resolved_at: Optional[float] = None


class CheckpointPayload(BaseModel):
    agent_id: str
    working: WorkingMemory
    recent_episodic: list[MemoryEntry]  # last 10 for quick resume context
    checkpoint_version: int
    created_at: float = Field(default_factory=time.time)


if IndexSchema:
    EPISODIC_SCHEMA = IndexSchema.from_dict(
        {
            "index": {
                "name": "episodic_memory",
                "prefix": "mem:episodic",
                "storage_type": "json",
            },
            "fields": [
                {"name": "agent_id", "type": "tag"},
                {"name": "content", "type": "text"},
                {"name": "source", "type": "tag"},
                {"name": "category", "type": "tag"},
                {"name": "confidence", "type": "numeric"},
                {"name": "decay_rate", "type": "numeric"},
                {"name": "created_at", "type": "numeric"},
                {"name": "last_reinforced", "type": "numeric"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": 1536,
                        "distance_metric": "cosine",
                        "algorithm": "hnsw",
                        "datatype": "float32",
                    },
                },
            ],
        }
    )

    SEMANTIC_SCHEMA = IndexSchema.from_dict(
        {
            "index": {
                "name": "semantic_memory",
                "prefix": "mem:semantic",
                "storage_type": "json",
            },
            "fields": [
                {"name": "agent_id", "type": "tag"},
                {"name": "summary", "type": "text"},
                {"name": "source", "type": "tag"},
                {"name": "confidence", "type": "numeric"},
                {"name": "created_at", "type": "numeric"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": 1536,
                        "distance_metric": "cosine",
                        "algorithm": "hnsw",
                        "datatype": "float32",
                    },
                },
            ],
        }
    )
else:  # pragma: no cover - handled in tests via dependency install
    EPISODIC_SCHEMA = None
    SEMANTIC_SCHEMA = None
