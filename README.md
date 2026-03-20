# NEXUS — Intelligent Multi-LLM Orchestrator

A production-grade semantic KNN orchestration system that routes queries to the optimal LLM from a pool of 7 models. Features human-in-the-loop clarification, parallel worker dispatch, quality judge with escalation, and real-time trace UI.

## Architecture

```
Query -> Classifier (Cerebras Llama 3.1 8B, ~200ms)
         |
         can_self_answer? -> Llama answers directly -> END
         is_ambiguous?    -> HITL: interrupt() -> user replies -> resume -> continue
         |
         KNN Router
         embed query -> cosine similarity vs 70 prototypes -> top-5 KNN vote -> best model
         |                          |
         single task                subtasks[]
         Worker Node                Parallel Workers (asyncio.gather) -> Aggregator
         |
         is_critical?
         NO  -> set_final -> return response
         YES -> Judge (Gemini Flash)
                score >= 7 -> return
                score < 7  -> Escalation Worker (Opus) -> return
         |
         FastAPI astream_events -> SSE -> Streamlit
         LangSmith traces everything automatically
```

## Models

| Model | Provider | Used For |
|-------|----------|----------|
| Llama 3.1 8B | Cerebras | Classifier + trivial self-answers |
| Llama 3.1 8B | Groq | Simple Q&A |
| Kimi K2 | Groq | Code (low-medium complexity) |
| GPT OSS 120B | Cerebras | General medium tasks |
| Qwen 3 235B | Cerebras | Research, complex reasoning |
| GPT-4o | OpenAI | Critical Q&A, factual research |
| Gemini 2.5 Flash | Google | Math, aggregator, judge |
| Claude Opus 4.6 | OpenRouter | Critical code (escalation only) |

## KNN Routing

Queries are embedded using `text-embedding-3-small` and compared against 70 prototype examples (10 per model category). Top-5 nearest neighbors vote on the best model via majority vote. Cost: ~$0.00001/query.

## Stack

| Tool | Role |
|------|------|
| **LangGraph** | Agent graph: nodes, conditional edges, parallel Send API, interrupt/resume |
| **LiteLLM** | Single `acompletion()` call for all 7 models + auto cost tracking |
| **LangSmith** | Traces every node/LLM call automatically |
| **FastAPI** | Async backend, SSE streaming |
| **Streamlit** | Chat UI + live agent trace sidebar with KNN bar chart |
| **scikit-learn** | `cosine_similarity` for KNN routing |

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
# Fill in your API keys (see .env.example for details)

# 4. Run
python main.py
```

Or run backend and UI separately:
```bash
# Terminal 1 — Backend
uvicorn src.api.main:app --reload --port 8000 --app-dir .

# Terminal 2 — UI
cd src && streamlit run ui/app.py
```

### Docker

```bash
docker compose up --build
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Send query, returns SSE stream |
| POST | `/resume` | Resume HITL-interrupted graph |
| GET | `/trace/{session_id}` | Get full trace for a session |
| GET | `/models` | List available models and costs |
| GET | `/health` | Health check + KNN index status |

Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## Testing

```bash
# Install dev dependencies
pip install pytest pytest-asyncio pytest-cov

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## Benchmark

Run the 56-query benchmark to evaluate routing accuracy and cost savings:

```bash
cd src && python -m eval.e2e_benchmark --limit 10
```

## Project Structure

```
nexus/
├── src/
│   ├── core/
│   │   ├── state.py          # NexusState TypedDict
│   │   ├── graph.py          # Full LangGraph pipeline
│   │   ├── config.py         # Model constants, thresholds
│   │   ├── metrics.py        # Cost calculation (LiteLLM + fallback)
│   │   ├── logging.py        # Structured logging
│   │   └── prototypes.py     # 70 example queries for KNN
│   ├── agents/
│   │   ├── classifier.py     # Llama classifier node
│   │   ├── knn_router.py     # Embed + cosine + KNN vote
│   │   ├── hitl.py           # interrupt() / resume node
│   │   ├── worker.py         # Single + parallel workers
│   │   ├── aggregator.py     # Merge parallel outputs
│   │   └── judge.py          # Judge + escalation worker
│   ├── api/
│   │   └── main.py           # FastAPI + SSE streaming
│   ├── ui/
│   │   └── app.py            # Streamlit chat + trace UI
│   └── eval/
│       ├── benchmark.py      # 56-query test suite with expected routing
│       └── e2e_benchmark.py  # Full E2E benchmark runner
├── tests/                    # Unit tests (pytest)
├── main.py                   # Integrated runner (backend + UI)
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── pyproject.toml
```

## License

MIT
