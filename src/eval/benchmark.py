import asyncio
import time
import json
import uuid
from langgraph.types import Command
from core.graph import nexus_graph
from core.config import GPT5_BASELINE_COST

# 60 realistic queries with expected model routing (approx 8-9 per category)
QUERIES = [
    # Simple Q&A — expect Llama Groq
    {"q": "Hello, how are you?", "expected": "groq/llama-3.1-8b-instant"},
    {"q": "What is the capital of Japan?", "expected": "groq/llama-3.1-8b-instant"},
    {"q": "Who invented the telephone?", "expected": "groq/llama-3.1-8b-instant"},
    {"q": "What does DNA stand for?", "expected": "groq/llama-3.1-8b-instant"},
    {"q": "How many days are in a leap year?", "expected": "groq/llama-3.1-8b-instant"},
    {"q": "What is the largest ocean on Earth?", "expected": "groq/llama-3.1-8b-instant"},
    {"q": "Who painted the Mona Lisa?", "expected": "groq/llama-3.1-8b-instant"},
    {"q": "What is the speed of light?", "expected": "groq/llama-3.1-8b-instant"},
    {"q": "Define entropy in one sentence.", "expected": "groq/llama-3.1-8b-instant"},

    # Code — expect Kimi K2
    {"q": "Write a Python function to merge two sorted arrays.", "expected": "groq/kimi-k2"},
    {"q": "Implement a stack using linked lists in Java.", "expected": "groq/kimi-k2"},
    {"q": "Debug this TypeScript function that returns undefined.", "expected": "groq/kimi-k2"},
    {"q": "Write unit tests for a REST API endpoint using pytest.", "expected": "groq/kimi-k2"},
    {"q": "Create a React hook for debounced search.", "expected": "groq/kimi-k2"},
    {"q": "Implement the observer pattern in Python.", "expected": "groq/kimi-k2"},
    {"q": "Write a SQL query to find the second highest salary.", "expected": "groq/kimi-k2"},
    {"q": "Convert this Python 2 code to Python 3.", "expected": "groq/kimi-k2"},
    {"q": "Write a basic web scraper in Python using BeautifulSoup.", "expected": "groq/kimi-k2"},

    # General — expect GPT OSS 120B
    {"q": "Compare the pros and cons of electric vs hydrogen cars.", "expected": "groq/gpt-oss-120b"},
    {"q": "Explain how blockchain consensus mechanisms work.", "expected": "groq/gpt-oss-120b"},
    {"q": "What are the key differences between agile and waterfall?", "expected": "groq/gpt-oss-120b"},
    {"q": "Explain the model-view-controller architecture pattern.", "expected": "groq/gpt-oss-120b"},
    {"q": "Describe how DNS resolution works step by step.", "expected": "groq/gpt-oss-120b"},
    {"q": "Compare Docker and virtual machines.", "expected": "groq/gpt-oss-120b"},
    {"q": "What are the trade-offs of eventual consistency?", "expected": "groq/gpt-oss-120b"},
    {"q": "How does HTTPS encryption work?", "expected": "groq/gpt-oss-120b"},
    {"q": "Explain the difference between threads and processes.", "expected": "groq/gpt-oss-120b"},

    # Research — expect Qwen 235B
    {"q": "Analyze the economic impact of AI on the labor market in 2025.", "expected": "cerebras/qwen3-235b"},
    {"q": "Write a comprehensive overview of mRNA vaccine technology.", "expected": "cerebras/qwen3-235b"},
    {"q": "Explain the geopolitical implications of semiconductor supply chains.", "expected": "cerebras/qwen3-235b"},
    {"q": "Provide a deep analysis of central bank digital currencies.", "expected": "cerebras/qwen3-235b"},
    {"q": "Summarize the evolution of machine learning from 1950 to present.", "expected": "cerebras/qwen3-235b"},
    {"q": "Analyze the sociological effects of social media on Gen Z.", "expected": "cerebras/qwen3-235b"},
    {"q": "Write a literature review on attention mechanisms in NLP.", "expected": "cerebras/qwen3-235b"},
    {"q": "Compare the education systems of Finland, Japan, and the US.", "expected": "cerebras/qwen3-235b"},

    # Critical — expect GPT-4o
    {"q": "Is it safe to combine metformin and alcohol?", "expected": "openai/gpt-4o"},
    {"q": "What are my rights if wrongfully terminated in Texas?", "expected": "openai/gpt-4o"},
    {"q": "Explain the tax implications of selling inherited property.", "expected": "openai/gpt-4o"},
    {"q": "What are the warning signs of a stroke?", "expected": "openai/gpt-4o"},
    {"q": "Review this non-compete clause for enforceability issues.", "expected": "openai/gpt-4o"},
    {"q": "What are the fiduciary duties of a board of directors?", "expected": "openai/gpt-4o"},
    {"q": "Explain HIPAA compliance requirements for a web app.", "expected": "openai/gpt-4o"},
    {"q": "What are the side effects of long-term statin use?", "expected": "openai/gpt-4o"},

    # Math — expect Gemini Flash
    {"q": "Solve for x: 5x^2 - 3x + 2 = 0.", "expected": "gemini/gemini-2.5-flash"},
    {"q": "What is the derivative of ln(x^2 + 1)?", "expected": "gemini/gemini-2.5-flash"},
    {"q": "Calculate compound interest: $5000 at 4% for 10 years.", "expected": "gemini/gemini-2.5-flash"},
    {"q": "Convert 250 kilometers per hour to miles per hour.", "expected": "gemini/gemini-2.5-flash"},
    {"q": "Find the eigenvalues of matrix [[2,1],[1,2]].", "expected": "gemini/gemini-2.5-flash"},
    {"q": "Integrate x*e^x dx.", "expected": "gemini/gemini-2.5-flash"},
    {"q": "Compute the probability of rolling sum 7 with two dice.", "expected": "gemini/gemini-2.5-flash"},

    # Critical code — expect Opus
    {"q": "Write production JWT auth middleware with refresh tokens for Express.", "expected": "openrouter/anthropic/claude-opus-4.6"},
    {"q": "Design a scalable event-driven payment processing system.", "expected": "openrouter/anthropic/claude-opus-4.6"},
    {"q": "Perform a security audit on this REST API implementation.", "expected": "openrouter/anthropic/claude-opus-4.6"},
    {"q": "Implement distributed locking with Redis and proper failover.", "expected": "openrouter/anthropic/claude-opus-4.6"},
    {"q": "Write a complete CI/CD pipeline for Kubernetes deployment.", "expected": "openrouter/anthropic/claude-opus-4.6"},
    {"q": "Design the database schema for encrypted real-time messaging.", "expected": "openrouter/anthropic/claude-opus-4.6"},
]


async def run_benchmark():
    print(f"Running {len(QUERIES)}-query benchmark...")

    results = []
    total_cost = 0.0
    critical_queries = 0
    escalations = 0
    latency_total = 0.0
    correct_routing = 0
    knn_confidence_total = 0.0

    for idx, item in enumerate(QUERIES):
        query = item["q"]
        expected = item["expected"]

        session_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": session_id}}

        start_time = time.time()
        initial_state = {"query": query}

        try:
            final_state = None
            async for event in nexus_graph.astream(initial_state, config=config, stream_mode="values"):
                final_state = event

            # If HITL was triggered, resume with default answer
            state_snapshot = nexus_graph.get_state(config)
            if state_snapshot.next and "hitl" in state_snapshot.next:
                async for event in nexus_graph.astream(
                    Command(resume="please proceed with best guess"), config=config, stream_mode="values"
                ):
                    final_state = event

            end_time = time.time()
            latency = end_time - start_time
            latency_total += latency

            final_memory = nexus_graph.get_state(config).values

            selected_models = final_memory.get("selected_models", [])
            routed_model = selected_models[0] if selected_models else "unknown"
            is_critical = final_memory.get("is_critical", False)
            escalated = final_memory.get("escalation_count", 0) > 0
            query_cost = final_memory.get("total_cost", 0.0)
            knn_scores = final_memory.get("knn_scores", {})
            top_knn = max(knn_scores.values()) if knn_scores else 0.0

            if routed_model == expected:
                correct_routing += 1

            if is_critical:
                critical_queries += 1

            if escalated:
                escalations += 1

            total_cost += query_cost
            knn_confidence_total += top_knn

            results.append({
                "id": idx,
                "query": query,
                "expected": expected,
                "routed": routed_model,
                "correct": routed_model == expected,
                "critical": is_critical,
                "escalated": escalated,
                "latency_s": round(latency, 2),
                "cost_usd": round(query_cost, 6),
                "knn_top_score": round(top_knn, 4),
            })

            status = "✅" if routed_model == expected else "❌"
            print(f"[{idx+1}/{len(QUERIES)}] {status} routed={routed_model} expected={expected} latency={latency:.1f}s")

        except Exception as e:
            print(f"[{idx+1}/{len(QUERIES)}] ❌ Failed: {e}")

    # Summary
    n = len(QUERIES)
    avg_cost = total_cost / n if n > 0 else 0
    baseline_cost = GPT5_BASELINE_COST * n
    saved = baseline_cost - total_cost
    saved_pct = (saved / baseline_cost * 100) if baseline_cost > 0 else 0

    summary = {
        "total_queries": n,
        "routing_accuracy": f"{(correct_routing / n) * 100:.1f}%",
        "avg_cost_per_query": f"${avg_cost:.6f}",
        "total_cost": f"${total_cost:.6f}",
        "vs_gpt5_baseline": f"${baseline_cost:.4f}",
        "cost_saved": f"${saved:.4f} ({saved_pct:.1f}%)",
        "judge_triggered": f"{critical_queries}/{n}",
        "escalations": f"{escalations}/{n}",
        "avg_knn_confidence": f"{knn_confidence_total / n:.4f}" if n > 0 else "N/A",
        "avg_latency": f"{latency_total / n:.2f}s" if n > 0 else "N/A",
    }

    print(f"\n{'='*50}")
    print("BENCHMARK SUMMARY")
    print("="*50)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    final_output = {"summary": summary, "results": results}

    with open("eval/results.json", "w") as f:
        json.dump(final_output, f, indent=4)

    print("\nResults saved to eval/results.json")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
