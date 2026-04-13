# MemoryOS

### The Long-Term Memory Layer for Personal AI & Coding Agents

Most LLM agents suffer from "goldfish memory"—they forget everything between sessions, or worse, they retrieve stale, contradictory information from a vector database indefinitely. 

**MemoryOS** solves this by mimicking how the human brain actually works. It makes agents remember the *right* things, allows different types of memories to fade at different speeds, surfaces contradictions before they poison reasoning, and ensures you never lose your place mid-task. 

Designed specifically for personal AI assistants and coding agents (like Claude Code and Cursor), MemoryOS gives your AI a persistent identity and understanding of *you*.

---

## 🧠 Core Innovations: How It Mimics the Human Brain

### 1. Temporal Confidence Decay (The Ebbinghaus Curve)
Not all facts are equal forever. Our system applies an exponential half-life to memories based on their cognitive category. 
- **Task Context** (e.g., "I'm debugging the auth flow") decays quickly (hours).
- **Workflow Patterns** (e.g., "Run `npm run build:staging`") decay moderately (weeks).
- **Personal Preferences** (e.g., "I prefer functional React components") are essentially permanent. 
When a memory's confidence drops below a threshold, it is safely filtered out of the agent's active context window—just like a human forgetting trivial daily tasks while retaining core skills.

### 2. Deterministic Write Gating & Explicit Contradiction Resolution
Simple RAG systems blindly append new chunks, silently creating conflicts (e.g., "Meeting is Thursday" vs. "Meeting is Wednesday").
MemoryOS routes all writes through a **Deterministic Write Gate** that checks for semantic near-duplicates and logical contradictions *before* writing. If a direct contradiction is flagged with high confidence, the agent pauses to explicitly ask the user which is correct. The resolution is fully transactional, maintaining a complete temporal lineage of the corrected facts.

### 3. Hybrid Retrieval: Vector + Graph
MemoryOS fuses two distinct retrieval systems for human-like recall:
- **Vector Search (Semantic):** "This concept feels related." (Backed by RedisVL).
- **Graph Search (Relational):** "These entities are connected." (Backed by lightweight Redis Sets mapping entity co-occurrence).
The hybrid planner fuses both scores, returning highly contextualized results with full provenance (explainability), ensuring the agent understands *how* things are connected, not just that they share keywords.

### 4. Zero-Context-Loss Session Recovery
Working memory (current task, progress percentage, last action) is persisted in Redis. If the agent is interrupted or the process restarts, it resumes from the exact task state and progress percentage.

---

## 🏗️ Architecture

```mermaid
flowchart LR
    %% Data Flow
    userMsg[User Message] --> extract[LLM Fact Extraction]
    extract --> writeGate[Deterministic Write Gate]
    writeGate --> contra[Contradiction Engine]
    
    %% Branching Logic
    contra -->|No Conflict| version[Temporal Versioning]
    contra -->|Conflict Detected| resolver[User Resolution]
    resolver --> version
    
    %% Storage
    version --> episodic[(Episodic Vector Store)]
    version --> graph[(Entity Graph Store)]
    
    %% Retrieval
    query[Agent Query] --> planner[Hybrid Retrieval Planner]
    planner --> episodic
    planner --> graph
    
    %% Ranking and Output
    episodic --> rank[Fusion & Re-rank]
    graph --> rank
    rank --> context[Prompt Context\nwith Provenance]
    context --> agent[Agent Response]
    
    %% Styling
    classDef process fill:#2f2f2f,stroke:#424242,color:#fff
    classDef storage fill:#064e3b,stroke:#065f46,color:#34d399
    classDef highlight fill:#4f46e5,stroke:#4338ca,color:#fff
    
    class extract,writeGate,contra,resolver,planner,rank,context process;
    class episodic,graph storage;
    class userMsg,agent,query highlight;
```

1. **Working Memory (`memory/working.py`)**: Short-lived task context, progress tracking, and active execution state.
2. **Episodic Memory (`memory/episodic.py`)**: Individual events, facts, and observations stored with vector embeddings, chronological versioning, and live confidence scores.
3. **Graph Memory (`memory/graph.py`)**: Relational entity maps for multi-hop recall.
4. **Governance & Audit (`memory/policies.py`, `memory/audit.py`)**: Enforces retention TTLs, handles PII sensitivity, and logs operations for complete observability.

---

## 🚀 Setup & Installation

1. Copy `.env.example` to `.env` and fill in:
   - `OPENAI_API_KEY` (Required for Embeddings and GPT-4o-mini extraction/contradiction logic)
   - `REDIS_URL` (Redis Cloud free tier works perfectly)

2. Install dependencies via a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### 1. The Interactive UI Dashboard
A full OpenAI-style chat interface that displays your session threads, live working memory states, confidence decay curves, and token savings in real-time. It includes a dedicated **Developer Profile** panel to audit exactly what the agent knows about you.
```bash
uvicorn api.server:app --reload --port 8000
```
Open `http://localhost:8000` in your browser.

### 2. Interactive Terminal Agent
Chat directly with the memory-powered agent from your terminal.
```bash
python main.py
```
*Try typing: "My favorite language is Python." Then: "Actually, I prefer Rust now." to trigger the explicit contradiction resolution flow!*

### 3. Cursor & Claude Code Integration
MemoryOS is designed to act as the context engine for coding agents. It exposes MCP endpoints (`/api/context/snapshot`, `/api/memory/search`, `/api/memory/store`) that dynamically inject your Developer Profile, Project Decisions, and Workflow Patterns directly into your agent's context window.

---

## 🧪 Testing & Evaluation

MemoryOS includes a rigorous suite of unit, integration, and end-to-end tests, alongside benchmark-style evaluation harnesses for hybrid retrieval accuracy.

```bash
# Run the fast unit/integration suite
pytest tests/ -m "not eval"

# Run the operation-level evaluation benchmarks
pytest tests/ -m "eval"
```