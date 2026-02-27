import os

# === Model Constants (Build Spec: 7 Models) ===

# Classifier (Cerebras — ultra-low latency)
MODEL_CLASSIFIER = "cerebras/llama3.1-8b"

# Worker Models (selected by KNN router)
MODEL_LLAMA_GROQ = "groq/llama-3.1-8b-instant"          # Simple Q&A
MODEL_KIMI_K2 = "groq/moonshotai/kimi-k2-instruct-0905"  # Code low-medium (updated)
MODEL_GPT_OSS = "cerebras/gpt-oss-120b"                  # General medium tasks (updated to Cerebras)
MODEL_QWEN_235B = "cerebras/qwen-3-235b-a22b-instruct-2507" # Research, complex reasoning (updated to Cerebras)
MODEL_GPT4O = "openai/gpt-4o"                            # Critical Q&A, factual research
MODEL_GEMINI_FLASH = "gemini/gemini-2.5-flash"           # Math, aggregator, judge
MODEL_OPUS = "openrouter/anthropic/claude-opus-4.6"      # Critical code only

# Special-purpose models
JUDGE_MODEL = MODEL_GEMINI_FLASH
AGGREGATOR_MODEL = MODEL_GEMINI_FLASH
MODEL_EMBED = "text-embedding-3-small"

# Thresholds
JUDGE_THRESHOLD = 7.0
MAX_ESCALATIONS = 1
KNN_K_VALUE = 5  # top-5 KNN vote

# GPT-5 baseline cost per query (for savings calculation).
# Override via env when you have your own measured baseline for your workload.
GPT5_BASELINE_COST = float(os.getenv("GPT5_BASELINE_COST", "0.012"))

# Costs per 1M tokens (USD) — approximate
MODEL_COSTS = {
    MODEL_CLASSIFIER: {"input": 0.10, "output": 0.10},
    MODEL_LLAMA_GROQ: {"input": 0.05, "output": 0.08},
    MODEL_KIMI_K2: {"input": 0.20, "output": 0.20},
    MODEL_GPT_OSS: {"input": 0.50, "output": 0.50},
    MODEL_QWEN_235B: {"input": 0.40, "output": 0.40},
    MODEL_GPT4O: {"input": 2.50, "output": 10.00},
    MODEL_GEMINI_FLASH: {"input": 0.075, "output": 0.30},
    MODEL_OPUS: {"input": 15.00, "output": 75.00},
}
