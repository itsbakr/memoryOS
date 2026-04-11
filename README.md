# Temporal Memory OS

### Persistent memory for long-running agents.

Standard LLMs forget everything between sessions, and most vector stores retrieve stale information indefinitely. Temporal Memory OS solves this by making agents remember the RIGHT things, surface conflicts before they poison reasoning, and never lose their place mid-task.

## Core Innovations

1. **Temporal Confidence Decay**: Not all facts are equal forever. Our system applies an exponential half-life to memories based on their source. Tool outputs decay quickly; core user preferences decay slowly. When a memory's confidence drops below 0.1, it is safely filtered out of the agent's context window.
2. **Explicit Contradiction Surfacing**: Simple RAG systems silently resolve conflicts by retrieving conflicting chunks. Our system flags direct contradictions with high confidence (e.g., "Meeting is Thursday" vs "Meeting is Wednesday") and bubbles them up. The agent explicitly asks the user which is correct before polluting the memory store.
3. **Blaxel Session Recovery**: We persist the agent's working memory and task state directly into a Blaxel sandbox filesystem alongside the fast Redis store. If the agent is interrupted or the process restarts, it resumes from the exact task state and progress percentage with zero context loss.

## The Architecture (Three-Tier Memory)

1. **Working Memory (`memory/working.py`)**: Short-lived task context, progress percentage, and current active task tracking (backed by Redis Hashes).
2. **Episodic Memory (`memory/episodic.py`)**: Individual events, facts, and observations stored with vector embeddings and an explicit chronological timeline (backed by RedisVL + OpenAI Embeddings).
3. **Semantic Memory (`memory/semantic.py`)**: Abstracted, summarized knowledge about the user and environment (backed by RedisVL).

## Setup & Installation

1. Copy `.env.example` to `.env` and fill in:
   - `OPENAI_API_KEY` (Required for Embeddings and GPT-4o-mini)
   - `REDIS_URL` (Redis Cloud free tier works great)
   - `BL_API_KEY` & `BL_WORKSPACE` (Optional: Blaxel credentials for sandbox checkpointing. Falls back to local `.checkpoints/` if omitted)
2. Install dependencies via virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

### 1. The Interactive Dashboard
A full OpenAI-style chat UI that displays your session threads, live memory states, confidence decay, and token savings in real-time.
```bash
uvicorn api.server:app --reload --port 8000
```
Open `http://localhost:8000` in your browser.

### 2. The Demo Scripts
Run the automated end-to-end scenarios to test Contradiction, Session Recovery, and Confidence Decay live.
```bash
python tests/demo.py
```

### 3. Interactive Agent Session
Chat directly with the memory-powered agent from your terminal.
```bash
python main.py
```
*Try typing: "My favorite color is blue." Then: "Actually, it's green." to trigger the contradiction resolution flow!*

## Testing

Comprehensive tests are included for models, decay engine, Redis serialization, and the contradiction pipelines.
```bash
pytest tests/
```