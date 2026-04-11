import math
import time

from .models import MemoryEntry

DECAY_RATES = {
    "user_said": 0.001,  # user preferences — very slow
    "agent_inferred": 0.05,  # agent reasoning — moderate
    "tool_result": 0.1,  # tool outputs — fast (data goes stale)
}


def calculate_current_confidence(
    memory: MemoryEntry, now: float | None = None
) -> float:
    """
    Exponential decay: conf(t) = initial_conf * e^(-decay_rate * hours_elapsed)

    A confidence of 0.3 means the memory is still used but weighted low.
    A confidence of 0.1 means it's effectively stale — filter it out.
    """
    if now is None:
        now = time.time()
    hours_elapsed = (now - memory.last_reinforced) / 3600
    decayed = memory.confidence * math.exp(-memory.decay_rate * hours_elapsed)
    return round(max(0.0, min(1.0, decayed)), 4)


def reinforce_memory(memory: MemoryEntry) -> MemoryEntry:
    """Called when a memory is retrieved and used — refreshes confidence."""
    memory.confidence = min(1.0, memory.confidence + 0.2)
    memory.last_reinforced = time.time()
    return memory


def should_prune(memory: MemoryEntry, threshold: float = 0.05) -> bool:
    """Returns True if memory should be removed from active retrieval."""
    return calculate_current_confidence(memory) < threshold
