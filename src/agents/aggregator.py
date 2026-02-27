import time
import litellm
from core.state import NexusState, TraceEntry
from core.config import AGGREGATOR_MODEL
from core.metrics import calculate_cost


async def aggregator_node(state: NexusState) -> dict:
    """Merges multiple parallel worker outputs into a single cohesive response."""
    query = state.get("enriched_query") or state.get("query", "")
    worker_responses = state.get("worker_responses", [])
    subtasks = state.get("subtasks", [])

    # Format context with subtask labels
    parts = []
    for i, w in enumerate(worker_responses):
        label = subtasks[i] if i < len(subtasks) else f"Agent {i+1}"
        parts.append(f"[{label}]: {w.get('response', '')}")
    combined_context = "\n\n".join(parts)

    try:
        start = time.time()
        response = await litellm.acompletion(
            model=AGGREGATOR_MODEL,
            messages=[
                {"role": "system", "content": "Merge these agent responses. No redundancy. Preserve all insights."},
                {"role": "user", "content": f"Query: {query}\n\nAgent responses:\n{combined_context}"},
            ],
        )
        latency_ms = (time.time() - start) * 1000
        aggregated_content = response.choices[0].message.content
        cost = calculate_cost(AGGREGATOR_MODEL, response)

    except Exception as e:
        aggregated_content = f"Error during aggregation: {str(e)}\n\nRaw outputs:\n{combined_context}"
        cost = 0.0
        latency_ms = 0.0

    trace_entry: TraceEntry = {
        "node": "aggregator",
        "action": "merged",
        "detail": f"Merged {len(worker_responses)} responses using {AGGREGATOR_MODEL}",
        "timestamp": time.time(),
    }

    return {
        "aggregated_response": aggregated_content,
        "trace": [trace_entry],
        "total_cost": state.get("total_cost", 0.0) + cost,
        "total_latency": state.get("total_latency", 0.0) + (latency_ms / 1000),
    }
