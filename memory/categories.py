"""
Developer Memory Ontology
-------------------------
Defines categories for developer-workflow-specific memory, with tuned decay
rates and layer mappings so Claude Code sessions have persistent identity.

Category → Layer mapping:
  personal_context  → semantic   (long-lived identity facts)
  user_preference   → semantic   (coding style, tool preferences)
  project_decision  → episodic   (architectural choices + rationale)
  workflow_pattern  → episodic   (commands, scripts, processes)
  codebase_knowledge→ episodic   (current state of files/code)
  task_context      → working    (active task / in-progress items)
"""

from __future__ import annotations

CATEGORY_DECAY_RATES: dict[str, float] = {
    # Permanent identity facts — essentially never decay
    "personal_context": 0.0005,
    # Coding preferences — very slow (weeks before meaningful decay)
    "user_preference": 0.001,
    # Architecture / tech decisions — slow (months)
    "project_decision": 0.002,
    # Workflow scripts, commands, deploy patterns — moderate (weeks)
    "workflow_pattern": 0.005,
    # File-level code observations — faster (days; code changes)
    "codebase_knowledge": 0.01,
    # Active task / in-progress work — fast (hours)
    "task_context": 0.1,
    # Fallback for uncategorised facts
    "general": 0.05,
}

CATEGORY_LAYERS: dict[str, str] = {
    "personal_context": "semantic",
    "user_preference": "semantic",
    "project_decision": "episodic",
    "workflow_pattern": "episodic",
    "codebase_knowledge": "episodic",
    "task_context": "working",
    "general": "episodic",
}

# Human-readable labels for the dashboard
CATEGORY_LABELS: dict[str, str] = {
    "personal_context": "Personal Context",
    "user_preference": "Preference",
    "project_decision": "Project Decision",
    "workflow_pattern": "Workflow Pattern",
    "codebase_knowledge": "Codebase Knowledge",
    "task_context": "Active Task",
    "general": "General",
}

# Categories shown in the "Developer Profile" panel
PROFILE_CATEGORIES = {"personal_context", "user_preference"}

# Categories shown in the "Project Context" panel
PROJECT_CATEGORIES = {"project_decision", "codebase_knowledge"}

# Categories shown in the "How I Work" panel
WORKFLOW_CATEGORIES = {"workflow_pattern"}

ALL_CATEGORIES = set(CATEGORY_DECAY_RATES.keys())


def decay_rate_for(category: str) -> float:
    return CATEGORY_DECAY_RATES.get(category, CATEGORY_DECAY_RATES["general"])


def layer_for(category: str) -> str:
    return CATEGORY_LAYERS.get(category, "episodic")
