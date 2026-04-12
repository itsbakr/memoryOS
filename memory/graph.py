from __future__ import annotations

import itertools
import re

from .working import get_redis

ENTITY_RE = re.compile(r"[A-Za-z0-9_#@]+")
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
    "about",
    "have",
    "has",
    "were",
    "what",
    "when",
    "where",
    "which",
    "their",
    "there",
    "over",
    "under",
    "then",
    "than",
    "them",
    "they",
    "you",
    "our",
    "but",
    "are",
}


def _nodes_key(agent_id: str) -> str:
    return f"graph:{agent_id}:nodes"


def _facts_key(agent_id: str) -> str:
    return f"graph:{agent_id}:facts"


def _stats_key(agent_id: str) -> str:
    return f"graph:{agent_id}:stats"


def _edges_key(agent_id: str, node: str) -> str:
    return f"graph:{agent_id}:edges:{node}"


def _node_facts_key(agent_id: str, node: str) -> str:
    return f"graph:{agent_id}:facts:{node}"


def _fact_nodes_key(agent_id: str, memory_id: str) -> str:
    return f"graph:{agent_id}:fact:{memory_id}"


def extract_entities(text: str) -> list[str]:
    tokens = [t.lower() for t in ENTITY_RE.findall(text)]
    entities = [t for t in tokens if len(t) > 2 and t not in STOPWORDS]
    return list(dict.fromkeys(entities))


async def add_fact_to_graph(agent_id: str, memory_id: str, content: str) -> list[str]:
    entities = extract_entities(content)
    if not entities:
        return []

    r = await get_redis()
    nodes_key = _nodes_key(agent_id)
    stats_key = _stats_key(agent_id)

    new_nodes = await r.sadd(nodes_key, *entities)
    if new_nodes:
        await r.hincrby(stats_key, "nodes", new_nodes)

    new_fact = await r.sadd(_facts_key(agent_id), memory_id)
    if new_fact:
        await r.hincrby(stats_key, "facts", 1)

    await r.sadd(_fact_nodes_key(agent_id, memory_id), *entities)

    edges_added = 0
    for left, right in itertools.combinations(entities, 2):
        edges_added += await r.sadd(_edges_key(agent_id, left), right)
        edges_added += await r.sadd(_edges_key(agent_id, right), left)

    if edges_added:
        await r.hincrby(stats_key, "edges", edges_added)

    for entity in entities:
        await r.sadd(_node_facts_key(agent_id, entity), memory_id)

    return entities


async def related_fact_ids(agent_id: str, query: str, limit: int = 20) -> list[str]:
    entities = extract_entities(query)
    if not entities:
        return []

    r = await get_redis()
    fact_ids: set[str] = set()

    for entity in entities:
        fact_ids.update(await r.smembers(_node_facts_key(agent_id, entity)))
        neighbors = await r.smembers(_edges_key(agent_id, entity))
        for neighbor in list(neighbors)[:5]:
            fact_ids.update(await r.smembers(_node_facts_key(agent_id, neighbor)))
        if len(fact_ids) >= limit:
            break

    return list(fact_ids)[:limit]


async def graph_stats(agent_id: str) -> dict:
    r = await get_redis()
    stats = await r.hgetall(_stats_key(agent_id))
    return {
        "nodes": int(stats.get("nodes", 0)),
        "edges": int(stats.get("edges", 0)),
        "facts": int(stats.get("facts", 0)),
    }
