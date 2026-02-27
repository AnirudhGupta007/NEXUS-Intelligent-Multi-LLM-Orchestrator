import json
import time
import re
import litellm
from core.state import NexusState, TraceEntry
from core.config import MODEL_CLASSIFIER
from core.metrics import calculate_cost


def _is_greeting_or_smalltalk(query: str) -> bool:
    q = query.strip().lower()
    if not q:
        return False

    q = re.sub(r"\s+", " ", q)
    exact = {
        "hi",
        "hello",
        "hey",
        "yo",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "how are you doing",
        "whats up",
        "what's up",
    }
    if q in exact:
        return True

    return q.startswith("hi ") or q.startswith("hello ") or q.startswith("hey ")


async def classifier_node(state: NexusState) -> dict:
    """Classifies the incoming query using Cerebras Llama 3.1 8B.
    Returns: can_self_answer, is_critical, is_ambiguous, clarifying_question,
             self_answer, subtasks[].
    NOTE: does NOT output task_type or model — KNN router handles that.
    """
    query = state.get("query", "")
    conversation_turns = state.get("conversation_turns", 0)

    prompt = f"""Analyze the incoming query and return ONLY valid JSON with these keys:
- can_self_answer: bool (true if it's a simple greeting or trivial question you can answer immediately)
- self_answer: string or null (your direct answer if can_self_answer is true)
- is_ambiguous: bool (true if the query lacks sufficient detail and needs human clarification)
- clarifying_question: string or null (the question to ask the user if is_ambiguous is true)
- is_critical: bool (true if this involves sensitive data, important logic, medical, legal, or financial topics)
- subtasks: list of strings (break into distinct subtasks if the query has multiple parts, otherwise empty list)

Return ONLY JSON. No markdown, no explanation.

Query: {query}"""

    try:
        start = time.time()
        response = await litellm.acompletion(
            model=MODEL_CLASSIFIER,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        latency_ms = (time.time() - start) * 1000
        content = response.choices[0].message.content
        result = json.loads(content)

        # Track cost
        cost = calculate_cost(MODEL_CLASSIFIER, response)

    except Exception as e:
        # Fallback defaults if LLM fails
        result = {
            "can_self_answer": False,
            "self_answer": None,
            "is_ambiguous": False,
            "clarifying_question": None,
            "is_critical": False,
            "subtasks": [],
        }
        cost = 0.0
        latency_ms = 0.0

    # If conversation_turns > 0, force is_ambiguous=false (already clarified)
    is_ambiguous = result.get("is_ambiguous", False)
    if conversation_turns > 0:
        is_ambiguous = False

    can_self_answer = result.get("can_self_answer", False)
    # Guardrail: only allow early exit for clear greeting/small-talk.
    if can_self_answer and not _is_greeting_or_smalltalk(query):
        can_self_answer = False
    
    subtasks = result.get("subtasks", [])
    if not isinstance(subtasks, list):
        subtasks = []

    trace_entry: TraceEntry = {
        "node": "classifier",
        "action": "classified",
        "detail": f"self={can_self_answer} ambiguous={is_ambiguous} critical={result.get('is_critical', False)} subtasks={len(subtasks)}",
        "timestamp": time.time(),
    }

    output = {
        "can_self_answer": can_self_answer,
        "is_ambiguous": is_ambiguous,
        "is_critical": result.get("is_critical", False),
        "clarifying_question": result.get("clarifying_question") or "",
        "subtasks": subtasks,
        "original_query": query,
        "trace": [trace_entry],
        "total_cost": state.get("total_cost", 0.0) + cost,
        "total_latency": state.get("total_latency", 0.0) + (latency_ms / 1000),
    }

    # If can_self_answer, set final_response directly
    if can_self_answer and result.get("self_answer"):
        output["final_response"] = result["self_answer"]

    return output
