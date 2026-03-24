# NEXUS — Intelligent Multi-LLM Orchestrator

A production-grade semantic KNN orchestration system that routes queries to the optimal LLM from a pool of 7 models. Features human-in-the-loop clarification, parallel worker dispatch, quality judge with escalation, and real-time trace UI.

> **Core idea:** Not every query needs an expensive model. "What is the capital of Japan?" should cost $0.00002, not $0.012. NEXUS routes 80% of queries to cheap models (saving 79-99%) and only escalates critical queries (medical, legal, financial) to expensive models with quality verification.

## Benchmark Results

56-query benchmark across 7 categories:

| Category | Queries | Routing Accuracy | Avg Cost | Baseline | Savings |
|----------|---------|-----------------|----------|----------|---------|
| Simple Q&A | 9 | ~90% | ~$0.00003 | $0.002 | **~98%** |
| Code | 9 | ~80% | ~$0.002 | $0.008 | **~75%** |
| General | 9 | ~78% | ~$0.001 | $0.006 | **~83%** |
| Research | 8 | ~75% | ~$0.003 | $0.015 | **~80%** |
| Critical | 8 | ~88% | ~$0.005 | $0.012 | **~58%** |
| Math | 7 | ~86% | ~$0.0001 | $0.004 | **~97%** |
| Critical Code | 6 | ~67% | ~$0.008 | $0.020 | **~60%** |

*Run `cd src && python -m eval.e2e_benchmark` for exact numbers with your API keys.*

**Key improvements over initial version:**
- Expanded KNN prototypes (10 → 15 per model) for better routing accuracy
- Tiered judge thresholds: strict for medical/legal, lenient for code
- Cheaper escalation: code queries escalate to GPT-4o instead of Opus (10x cheaper)
- Context-aware critical guardrail prevents false escalations on code queries
- Response quality auto-scoring via Gemini Flash

## Architecture

```
Query -> Classifier (Cerebras Llama 3.1 8B, ~200ms)
         |
         can_self_answer? -> answers directly -> END
         is_ambiguous?    -> HITL: interrupt() -> user clarifies -> resume
         |
         KNN Router (Fireworks nomic-embed, ~10ms)
         embed query -> cosine similarity vs 105 prototypes -> top-5 vote
         |                          |
         single task                subtasks[]
         Worker Node                Parallel Workers (asyncio.gather) -> Aggregator
         |
         is_critical? (medical/legal/financial only)
         NO  -> return response
         YES -> Judge (Gemini Flash, lenient for standard, strict for critical)
                score >= 7 -> approved
                score < 7  -> Escalation Worker (Opus) -> return
```

### Key Design Decisions

- **KNN over LLM routing:** 10x faster (10ms vs 200ms), 10x cheaper ($0.00001 vs $0.0001), no hallucination risk. Pure math, not a probabilistic LLM guess.
- **is_critical guardrail:** Small classifier LLMs over-flag queries as critical. A keyword guardrail ensures only medical/legal/financial/safety queries trigger the expensive judge path.
- **Context-aware judge:** The judge scores leniently for standard queries (pass if useful) and strictly for critical queries (require thoroughness and accuracy).
- **Cost-aware model selection:** 80% of queries go to models costing $0.05-$0.50/1M tokens. Expensive models ($2.50-$75/1M) only activate when needed.

## Models

| Model | Provider | Cost (per 1M tokens) | Used For |
|-------|----------|---------------------|----------|
| Llama 3.1 8B | Cerebras | $0.10 | Classifier (speed over intelligence) |
| Llama 3.1 8B | Groq | $0.05 | Simple Q&A |
| Kimi K2 | Groq | $0.20 | Code (low-medium complexity) |
| GPT OSS 120B | Cerebras | $0.50 | General medium tasks |
| Qwen 3 235B | Cerebras | $0.40 | Research, complex reasoning |
| GPT-4o | OpenAI | $2.50/$10.00 | Critical Q&A (medical, legal) |
| Gemini 2.5 Flash | Google | $0.075/$0.30 | Math, judge, aggregator |
| Claude Opus 4.6 | OpenRouter | $15/$75 | Escalation only (safety net) |

## Stack

| Tool | Role |
|------|------|
| **LangGraph** | Agent graph: conditional edges, parallel fan-out, interrupt/resume for HITL |
| **LiteLLM** | Unified `acompletion()` across all 7 providers + auto cost tracking |
| **Fireworks AI** | Embedding model (nomic-embed-text-v1.5) for KNN routing |
| **LangSmith** | End-to-end trace of every node and LLM call |
| **FastAPI** | Async backend with SSE streaming |
| **Streamlit** | Chat UI with live agent trace sidebar and KNN bar chart |
| **scikit-learn** | Cosine similarity for KNN routing |

## Setup

```bash
# 1. Clone
git clone https://github.com/AnirudhGupta007/NEXUS-Intelligent-Multi-LLM-Orchestrator.git
cd NEXUS-Intelligent-Multi-LLM-Orchestrator

# 2. Create venv and install
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Fill in your API keys (see .env.example for required keys)

# 4. Run
python main.py
# API: http://localhost:8000  |  UI: http://localhost:8501
```

### Docker

```bash
docker compose up --build
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Send query, returns SSE stream with trace + response |
| POST | `/resume` | Resume HITL-interrupted graph with user answer |
| GET | `/trace/{session_id}` | Get full trace for a session |
| GET | `/models` | List available models, costs, and prototypes |
| GET | `/health` | Health check + KNN index status |

Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## Testing

```bash
pip install pytest pytest-asyncio pytest-cov

# Run all tests (99 tests)
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

Tests cover: classifier (greeting detection, ambiguity, critical guardrail with context-aware keyword filtering), KNN router (cluster routing, index building, embedding cache), workers (model selection, timeouts, parallel dispatch, error handling, cost accumulation), judge (scoring, tiered thresholds, escalation with category-aware model selection), aggregator (merging, failure fallback, subtask labels), HITL (interrupt/resume), graph routing logic, API endpoints, cost calculation, config validation, prototype validation, benchmark data integrity, and integration tests.

## Benchmark

```bash
# Run full 56-query benchmark
cd src && python -m eval.e2e_benchmark

# Run quick test (first 14 queries)
cd src && python -m eval.e2e_benchmark --limit 14

# Custom timeout
cd src && python -m eval.e2e_benchmark --limit 21 --query-timeout-s 60
```

The benchmark tests 7 categories (simple Q&A, code, general, research, critical, math, critical code) and reports per-category routing accuracy, cost, and savings vs GPT-5 baseline.

## Project Structure

```
nexus/
├── src/
│   ├── core/
│   │   ├── state.py          # NexusState TypedDict
│   │   ├── graph.py          # LangGraph pipeline (nodes, edges, routing)
│   │   ├── config.py         # Model constants, thresholds, cost tables
│   │   ├── metrics.py        # Cost calculation (LiteLLM + manual fallback)
│   │   ├── logging.py        # Structured logging
│   │   └── prototypes.py     # 105 prototype queries for KNN (15 per model)
│   ├── agents/
│   │   ├── classifier.py     # Cerebras classifier + greeting/critical guardrails
│   │   ├── knn_router.py     # Embed + cosine similarity + top-5 KNN vote
│   │   ├── hitl.py           # LangGraph interrupt() / Command(resume=) node
│   │   ├── worker.py         # Single worker + parallel fan-out workers
│   │   ├── aggregator.py     # Merge parallel worker outputs
│   │   └── judge.py          # Context-aware judge + escalation worker
│   ├── api/
│   │   └── main.py           # FastAPI + SSE streaming + lifespan
│   ├── ui/
│   │   └── app.py            # Streamlit chat + real-time trace sidebar
│   └── eval/
│       ├── benchmark.py      # 56 categorized queries with expected routing
│       └── e2e_benchmark.py  # Full E2E runner with per-category analysis
├── tests/                    # 99 unit + integration tests (pytest + pytest-asyncio)
├── main.py                   # Integrated runner (backend + UI)
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── pyproject.toml
```

## License

MIT
