"""Integration tests — mock all LLM calls, run queries through the full LangGraph pipeline."""

import json
import pytest
import numpy as np
from unittest.mock import patch, AsyncMock, MagicMock
from langgraph.types import Command

from core.graph import nexus_graph
from tests.conftest import make_mock_completion


def _mock_knn_index():
    """Build a mock KNN index for integration tests."""
    from core.config import (
        MODEL_LLAMA_GROQ, MODEL_KIMI_K2, MODEL_GPT_OSS,
        MODEL_QWEN_235B, MODEL_GPT4O, MODEL_GEMINI_FLASH, MODEL_OPUS,
    )
    models = [MODEL_LLAMA_GROQ, MODEL_KIMI_K2, MODEL_GPT_OSS, MODEL_QWEN_235B, MODEL_GPT4O, MODEL_GEMINI_FLASH, MODEL_OPUS]
    np.random.seed(42)
    all_vectors = []
    all_labels = []
    for i, model in enumerate(models):
        center = np.zeros(1536)
        center[i * 200: (i + 1) * 200] = 1.0
        for _ in range(15):
            vec = center + np.random.randn(1536) * 0.1
            vec = vec / np.linalg.norm(vec)
            all_vectors.append(vec)
            all_labels.append(model)
    return {"all_vectors": np.array(all_vectors), "all_labels": all_labels}


def _mock_embed_for_model_idx(idx):
    """Return an embedding vector close to model cluster `idx`."""
    vec = np.zeros(1536)
    vec[idx * 200: (idx + 1) * 200] = 1.0
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


class TestSimpleQueryPipeline:
    @pytest.mark.asyncio
    async def test_greeting_exits_early(self):
        """Greeting should be self-answered without routing."""
        classifier_resp = make_mock_completion(json.dumps({
            "can_self_answer": True,
            "self_answer": "Hello! How can I help?",
            "is_ambiguous": False,
            "clarifying_question": None,
            "is_critical": False,
            "subtasks": [],
        }))

        with patch("agents.classifier.litellm.acompletion", new_callable=AsyncMock, return_value=classifier_resp):
            with patch("agents.classifier.calculate_cost", return_value=0.00001):
                config = {"configurable": {"thread_id": "test-greeting"}}
                final_state = None
                async for event in nexus_graph.astream(
                    {"query": "hello", "conversation_turns": 0, "total_cost": 0.0, "total_latency": 0.0,
                     "trace": [], "worker_responses": [], "knn_scores": {}, "selected_models": [], "escalation_count": 0},
                    config=config, stream_mode="values",
                ):
                    final_state = event

        assert final_state is not None
        assert final_state.get("can_self_answer") is True
        assert final_state.get("final_response") == "Hello! How can I help?"


class TestNonCriticalQueryPipeline:
    @pytest.mark.asyncio
    async def test_simple_qa_routes_to_worker(self):
        """Non-critical query should classify -> KNN route -> worker -> set_final."""
        import agents.knn_router as knn_mod
        original_index = knn_mod.KNN_INDEX
        knn_mod.KNN_INDEX = _mock_knn_index()

        classifier_resp = make_mock_completion(json.dumps({
            "can_self_answer": False, "self_answer": None,
            "is_ambiguous": False, "clarifying_question": None,
            "is_critical": False, "subtasks": [],
        }))
        worker_resp = make_mock_completion("Tokyo is the capital of Japan.")
        # Embed returns vector close to Llama cluster (idx 0)
        mock_embed = MagicMock()
        mock_embed.data = [{"embedding": _mock_embed_for_model_idx(0)}]

        try:
            with patch("agents.classifier.litellm.acompletion", new_callable=AsyncMock, return_value=classifier_resp):
                with patch("agents.classifier.calculate_cost", return_value=0.00001):
                    with patch("agents.knn_router.litellm.aembedding", new_callable=AsyncMock, return_value=mock_embed):
                        with patch("agents.worker.litellm.acompletion", new_callable=AsyncMock, return_value=worker_resp):
                            with patch("agents.worker.calculate_cost", return_value=0.00005):
                                config = {"configurable": {"thread_id": "test-simple-qa"}}
                                final_state = None
                                async for event in nexus_graph.astream(
                                    {"query": "What is the capital of Japan?", "conversation_turns": 0,
                                     "total_cost": 0.0, "total_latency": 0.0,
                                     "trace": [], "worker_responses": [], "knn_scores": {},
                                     "selected_models": [], "escalation_count": 0},
                                    config=config, stream_mode="values",
                                ):
                                    final_state = event

            assert final_state is not None
            assert final_state.get("final_response") is not None
            assert final_state.get("is_critical") is False
        finally:
            knn_mod.KNN_INDEX = original_index


class TestCriticalQueryRouting:
    def test_critical_flag_triggers_judge_route(self):
        """Verify that is_critical=True sends the query to judge after worker."""
        from core.graph import route_from_worker
        assert route_from_worker({"is_critical": True}) == "judge"

    @pytest.mark.asyncio
    async def test_classifier_sets_critical_for_medical_query(self):
        """Verify classifier correctly flags medical queries as critical."""
        from agents.classifier import classifier_node

        classifier_resp = make_mock_completion(json.dumps({
            "can_self_answer": False, "self_answer": None,
            "is_ambiguous": False, "clarifying_question": None,
            "is_critical": True, "subtasks": [],
        }))
        with patch("agents.classifier.litellm.acompletion", new_callable=AsyncMock, return_value=classifier_resp):
            with patch("agents.classifier.calculate_cost", return_value=0.00001):
                result = await classifier_node({
                    "query": "Is mixing drugs safe?",
                    "conversation_turns": 0,
                    "total_cost": 0.0,
                    "total_latency": 0.0,
                })
        assert result["is_critical"] is True

    @pytest.mark.asyncio
    async def test_judge_approves_good_critical_response(self):
        """Verify judge approves a high-scoring critical response."""
        from agents.judge import judge_node

        judge_resp = make_mock_completion(json.dumps({
            "score": 8.5, "dimensions": {"accuracy": 9, "completeness": 8, "reasoning_depth": 8},
            "failure_reason": "", "retry_instruction": "", "escalate_to": "",
        }))
        with patch("agents.judge.litellm.acompletion", new_callable=AsyncMock, return_value=judge_resp):
            with patch("agents.judge.calculate_cost", return_value=0.0003):
                result = await judge_node({
                    "query": "Is mixing drugs safe?",
                    "is_critical": True,
                    "worker_responses": [{"response": "Do not mix drugs without consulting a doctor."}],
                    "total_cost": 0.001,
                    "total_latency": 1.0,
                    "escalation_count": 0,
                })
        assert result["judge_score"] == 8.5
        assert "escalation_model" not in result
