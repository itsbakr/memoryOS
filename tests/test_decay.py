import time

import pytest

from memory.decay import calculate_current_confidence, reinforce_memory, should_prune
from memory.models import MemoryEntry


def test_calculate_current_confidence():
    now = time.time()
    mem = MemoryEntry(
        agent_id="agent-1",
        content="Tool data point",
        layer="episodic",
        source="tool_result",
        confidence=1.0,
        decay_rate=0.1,
        last_reinforced=now - 2 * 3600,
    )
    conf = calculate_current_confidence(mem, now=now)
    assert conf == pytest.approx(0.8187, rel=1e-3)


def test_reinforce_memory_caps():
    mem = MemoryEntry(
        agent_id="agent-1",
        content="Test",
        layer="episodic",
        source="user_said",
        confidence=0.95,
    )
    reinforced = reinforce_memory(mem)
    assert reinforced.confidence == 1.0
    assert reinforced.last_reinforced <= time.time()


def test_should_prune_threshold():
    now = time.time()
    mem = MemoryEntry(
        agent_id="agent-1",
        content="Stale data",
        layer="episodic",
        source="tool_result",
        confidence=0.1,
        decay_rate=0.1,
        last_reinforced=now - 72 * 3600,
    )
    assert should_prune(mem, threshold=0.05) is True
