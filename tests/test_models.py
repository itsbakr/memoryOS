import time

from memory.models import (
    CheckpointPayload,
    ContradictionEvent,
    MemoryEntry,
    WorkingMemory,
    EPISODIC_SCHEMA,
    SEMANTIC_SCHEMA,
)


def test_memory_entry_defaults():
    entry = MemoryEntry(
        agent_id="agent-1",
        content="User prefers morning meetings",
        layer="episodic",
        source="user_said",
    )
    assert entry.id
    assert entry.confidence == 1.0
    assert entry.decay_rate == 0.05
    assert entry.created_at <= time.time()
    assert entry.last_reinforced <= time.time()


def test_working_memory_defaults():
    working = WorkingMemory(agent_id="agent-1", task="Plan demo")
    assert working.progress_pct == 0.0
    assert working.tool_calls_made == 0
    assert working.last_checkpoint <= time.time()


def test_contradiction_event_defaults():
    event = ContradictionEvent(
        agent_id="agent-1",
        new_fact="Standup is Thursday",
        conflicting_memory_id="mem-123",
        conflicting_memory_content="Standup is Wednesday",
        confidence_score=0.9,
        explanation="Dates conflict",
    )
    assert event.resolution == "pending"
    assert event.created_at <= time.time()


def test_checkpoint_payload_structure():
    working = WorkingMemory(agent_id="agent-1", task="Plan demo")
    entry = MemoryEntry(
        agent_id="agent-1",
        content="User prefers morning meetings",
        layer="episodic",
        source="user_said",
    )
    payload = CheckpointPayload(
        agent_id="agent-1",
        working=working,
        recent_episodic=[entry],
        checkpoint_version=1,
    )
    assert payload.working.task == "Plan demo"
    assert payload.recent_episodic[0].content.startswith("User")


def test_index_schemas_available():
    assert EPISODIC_SCHEMA is not None
    assert SEMANTIC_SCHEMA is not None
