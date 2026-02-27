# NEXUS — Intelligent Multi-LLM Orchestrator

A general-purpose AI assistant that routes every query to the optimal LLM using semantic KNN similarity. Live trace of every agent decision. FastAPI backend + Streamlit UI.

## Architecture

```
Query -> Classifier (Cerebras Llama 3.1 8B, ~200ms)
         |
         can_self_answer? -> Llama answers directly -> END
         is_ambiguous?    -> HITL: interrupt() -> user replies -> resume -> continue
         |
         KNN Router
         embed query -> cosine similarity vs all prototypes -> top-5 KNN vote -> best model
         |                          |
         single task                subtasks[]
         Worker Node                Parallel Workers (asyncio.gather) -> Aggregator
         |
         is_critical?
         NO  -> set_final -> return response
         YES -> Judge (Gemini Flash)
                score >= 7 -> return
                score < 7  -> Escalation Worker (failure context injected) -> return
         |
         FastAPI astream_events -> SSE -> Streamlit
         LangSmith traces everything automatically
```

## Models

| Model | Provider | Used For |
|-------|----------|----------|
| Llama 3.1 8B | Cerebras | Classifier + trivial self-answers |
| Llama 3.1 8B | Groq | Simple Q&A |
| Kimi K2 | Groq | Code low-medium |
| GPT OSS 120B | Groq | General medium tasks |
| Qwen 3 235B | Cerebras | Research, complex reasoning |
| GPT-4o | OpenAI | Critical Q&A, factual research |
| Gemini 2.5 Flash | Google | Math, aggregator, judge |
| Opus 4.6 | OpenRouter | Critical code only |

## KNN Routing

Queries are embedded using `text-embedding-3-small` and compared against 70 prototype examples (10 per model). Top-5 nearest neighbors vote on the best model. Cost: ~$0.00001/query.

## Stack

| Tool | Role |
|------|------|
| **LangGraph** | Agent graph: nodes, conditional edges, parallel Send API, interrupt/resume |
| **LiteLLM** | Single `acompletion()` call for all 7 models + auto cost tracking |
| **LangSmith** | Traces every node/LLM call automatically |
| **FastAPI** | Async backend, SSE streaming |
| **Streamlit** | Chat UI + live agent trace sidebar with KNN bar chart |
| **scikit-learn** | `cosine_similarity` for KNN |

## Setup

```bash
# 1. Clone
git clone https://github.com/anirudh-tipstat/Heyvision-CRM.git
cd Heyvision-CRM
git checkout dev

# 2. Create venv and install
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .

# 3. Fill .env with API keys
cp .env.example .env
# Add: OPENAI_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY, CEREBRAS_API_KEY, OPENROUTER_API_KEY
# Add: LANGSMITH_TRACING=true, LANGSMITH_API_KEY, LANGSMITH_PROJECT
# 4. Run backend (terminal 1)
uvicorn api.main:app --reload --port 8000

# 5. Run UI (terminal 2)
streamlit run ui/app.py
```

Or use the integrated runner:
```bash
python main.py
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Send query, returns SSE stream |
| POST | `/resume` | Resume HITL-interrupted graph |
| GET | `/trace/{session_id}` | Get full trace for a session |
| GET | `/models` | List available models and costs |
| GET | `/health` | Health check |

Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## Project Structure

```
nexus/
├── src/
│   ├── core/
│   │   ├── state.py          # NexusState TypedDict
│   │   ├── graph.py          # Full LangGraph pipeline
│   │   ├── config.py         # Model constants, thresholds
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
│       └── benchmark.py      # 60-query test suite
├── main.py                   # Integrated runner
├── .env
└── pyproject.toml
```

## License

MIT
