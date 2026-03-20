import json
import time
import litellm
from core.state import NexusState, TraceEntry
from core.config import JUDGE_MODEL, JUDGE_THRESHOLD, MAX_ESCALATIONS, MODEL_OPUS
from core.metrics import calculate_cost

async def judge_node(state: NexusState) -> dict:
    """Evaluate response quality and approve or trigger escalation."""
    
    query = state.get("enriched_query") or state.get("query", "")
    
    # Evaluate previously generated content
    if state.get("aggregated_response"):
        response_to_evaluate = state.get("aggregated_response")
    elif state.get("worker_responses"):
        worker_responses = state.get("worker_responses", [])
        response_to_evaluate = worker_responses[-1].get("response", "") if worker_responses else "No response."
    else:
        response_to_evaluate = "No response generated."
        
    is_critical = state.get("is_critical", False)
    strictness = "STRICT" if is_critical else "LENIENT"
    context = (
        "This is a CRITICAL query (medical/legal/financial). Be strict — accuracy is paramount. "
        "Only pass responses that are thorough, accurate, and safe."
        if is_critical else
        "This is a standard query. Be lenient — if the response reasonably answers the question, "
        "give it a passing score. Minor gaps are acceptable. Only reject clearly wrong or empty responses."
    )

    prompt = f"""Evaluate the agent's response. Mode: {strictness}.
{context}

Return ONLY JSON with these keys:
- score: float (0-10). For standard queries, a reasonable answer should score 7+. For critical queries, only thorough answers score 7+.
- dimensions: dict with 'accuracy', 'completeness', 'reasoning_depth' (each 0-10)
- failure_reason: string (why it failed, or empty if it passes)
- retry_instruction: string (how to improve, or empty)
- escalate_to: string (model for retry, or empty)

Original Query: {query}
Agent Response: {response_to_evaluate}
"""
    
    try:
        start = time.time()
        response = await litellm.acompletion(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": f"You are a response quality judge. Score generously for standard queries (pass if the answer is useful). Score strictly for critical queries (medical/legal/financial). Threshold: {JUDGE_THRESHOLD}. If score < threshold, provide failure_reason and retry_instruction."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        latency_ms = (time.time() - start) * 1000
        content = response.choices[0].message.content
        result = json.loads(content)
        cost = calculate_cost(JUDGE_MODEL, response)
            
    except Exception as e:
        # Failsafe fallback 
        result = {
            "score": 0.0,
            "failure_reason": f"Evaluation system error: {str(e)}",
            "retry_instruction": "Retry with a highly intelligent model.",
            "escalate_to": MODEL_OPUS
        }
        cost = 0.0
        latency_ms = 0.0
        
    score = result.get("score", 0.0)
    passed = score >= JUDGE_THRESHOLD
    
    trace_entry: TraceEntry = {
        "node": "judge",
        "action": "approved" if passed else "rejected",
        "detail": f"Score {score:.1f}. " + (f"Passed." if passed else f"Reason: {result.get('failure_reason')}"),
        "timestamp": time.time()
    }
    
    output = {
        "judge_score": score,
        "judge_feedback": result.get("failure_reason", ""),
        "trace": [trace_entry],
        "total_cost": state.get("total_cost", 0.0) + cost,
        "total_latency": state.get("total_latency", 0.0) + (latency_ms / 1000),
    }
    
    if passed:
        return output
    else:
        # Reject and trigger escalation
        output["escalation_model"] = result.get("escalate_to", MODEL_OPUS)
        output["escalation_instruction"] = result.get("retry_instruction", "")
        return output

async def escalation_worker_node(state: NexusState) -> dict:
    """Invoked when the judge fails a response. Uses designated more capable model."""
    
    target_query = state.get("enriched_query") or state.get("query", "")
    escalation_instruction = state.get("escalation_instruction", "")
    escalation_model = state.get("escalation_model", MODEL_OPUS)
    
    prompt = f"Previous attempt failed: {escalation_instruction}. Fix this specifically and address the query below.\n\nQuery: {target_query}"
    
    try:
        start = time.time()
        response = await litellm.acompletion(
            model=escalation_model,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.time() - start) * 1000
        output_content = response.choices[0].message.content
        actual_model = response.model if hasattr(response, 'model') else escalation_model
        cost = calculate_cost(actual_model, response)

    except Exception as e:
        output_content = f"Escalation failed entirely: {str(e)}"
        actual_model = escalation_model
        cost = 0.0
        latency_ms = 0.0
        
    trace_entry: TraceEntry = {
        "node": "escalation_worker",
        "action": "escalated_response",
        "detail": f"Used {actual_model} after judge rejection.",
        "timestamp": time.time()
    }
    
    return {
        "final_response": output_content,
        "escalation_count": state.get("escalation_count", 0) + 1,
        "trace": [trace_entry],
        "total_cost": state.get("total_cost", 0.0) + cost,
        "total_latency": state.get("total_latency", 0.0) + (latency_ms / 1000),
    }
