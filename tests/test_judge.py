import json
import pytest
from unittest.mock import patch, AsyncMock
from agents.judge import judge_node, escalation_worker_node
from core.config import MODEL_OPUS, MODEL_GPT4O, ESCALATION_MODELS
from tests.conftest import make_mock_completion


class TestJudgeNode:
    @pytest.mark.asyncio
    async def test_passes_high_score(self):
        mock_resp = make_mock_completion(json.dumps({"score": 8.5, "dimensions": {}, "failure_reason": "", "retry_instruction": "", "escalate_to": ""}))
        with patch("agents.judge.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            with patch("agents.judge.calculate_cost", return_value=0.0003):
                result = await judge_node({"query": "test", "worker_responses": [{"response": "good answer"}], "total_cost": 0.0, "total_latency": 0.0, "escalation_count": 0})
        assert result["judge_score"] == 8.5
        assert "escalation_model" not in result

    @pytest.mark.asyncio
    async def test_rejects_low_score(self):
        mock_resp = make_mock_completion(json.dumps({"score": 4.0, "dimensions": {}, "failure_reason": "Too shallow", "retry_instruction": "Add depth", "escalate_to": MODEL_OPUS}))
        with patch("agents.judge.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            with patch("agents.judge.calculate_cost", return_value=0.0003):
                result = await judge_node({"query": "test", "worker_responses": [{"response": "bad"}], "total_cost": 0.0, "total_latency": 0.0, "escalation_count": 0})
        assert result["judge_score"] == 4.0
        assert result["escalation_model"] == MODEL_OPUS

    @pytest.mark.asyncio
    async def test_llm_failure_escalates(self):
        with patch("agents.judge.litellm.acompletion", new_callable=AsyncMock, side_effect=Exception("fail")):
            result = await judge_node({"query": "test", "worker_responses": [{"response": "x"}], "total_cost": 0.0, "total_latency": 0.0, "escalation_count": 0})
        assert result["judge_score"] == 0.0
        # Default escalation for non-critical queries is now GPT-4o
        assert result["escalation_model"] == ESCALATION_MODELS["default"]

    @pytest.mark.asyncio
    async def test_llm_failure_escalates_critical_to_opus(self):
        """Critical (medical/legal) queries should escalate to Opus on judge failure."""
        with patch("agents.judge.litellm.acompletion", new_callable=AsyncMock, side_effect=Exception("fail")):
            result = await judge_node({"query": "Is it safe to take this drug with alcohol?", "is_critical": True, "worker_responses": [{"response": "x"}], "total_cost": 0.0, "total_latency": 0.0, "escalation_count": 0})
        assert result["escalation_model"] == MODEL_OPUS


class TestEscalationWorker:
    @pytest.mark.asyncio
    async def test_escalation_works(self):
        mock_resp = make_mock_completion("Better answer")
        with patch("agents.judge.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            with patch("agents.judge.calculate_cost", return_value=0.01):
                result = await escalation_worker_node({"query": "test", "escalation_model": MODEL_OPUS, "escalation_instruction": "improve", "escalation_count": 0, "total_cost": 0.0, "total_latency": 0.0})
        assert result["final_response"] == "Better answer"
        assert result["escalation_count"] == 1
