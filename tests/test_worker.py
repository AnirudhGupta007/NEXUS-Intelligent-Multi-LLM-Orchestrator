import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from agents.worker import worker_node, parallel_worker_node
from tests.conftest import make_mock_completion


class TestWorkerNode:
    @pytest.mark.asyncio
    async def test_calls_selected_model(self):
        mock_resp = make_mock_completion("Quicksort here...")
        with patch("agents.worker.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            with patch("agents.worker.calculate_cost", return_value=0.0002):
                result = await worker_node({"query": "quicksort", "selected_models": ["groq/moonshotai/kimi-k2-instruct-0905"], "total_cost": 0.0, "total_latency": 0.0})
        assert result["worker_responses"][0]["response"] == "Quicksort here..."
        assert result["worker_responses"][0]["model"] == "groq/moonshotai/kimi-k2-instruct-0905"

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        with patch("agents.worker.asyncio.wait_for", side_effect=asyncio.TimeoutError):
            result = await worker_node({"query": "test", "selected_models": ["test-model"], "total_cost": 0.0, "total_latency": 0.0})
        assert result["worker_responses"][0]["response"] == "[timeout]"

    @pytest.mark.asyncio
    async def test_trace_recorded(self):
        mock_resp = make_mock_completion("response")
        with patch("agents.worker.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            with patch("agents.worker.calculate_cost", return_value=0.0001):
                result = await worker_node({"query": "test", "selected_models": ["m"], "total_cost": 0.0, "total_latency": 0.0})
        assert result["trace"][0]["node"] == "worker"


class TestParallelWorkerNode:
    @pytest.mark.asyncio
    async def test_dispatches_all_subtasks(self):
        mock_resp = make_mock_completion("subtask response")
        with patch("agents.worker.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            with patch("agents.worker.calculate_cost", return_value=0.0001):
                result = await parallel_worker_node({"query": "complex", "subtasks": ["a", "b", "c"], "selected_models": ["m1", "m2", "m3"], "total_cost": 0.0, "total_latency": 0.0})
        assert len(result["worker_responses"]) == 3
        assert result["trace"][0]["node"] == "parallel_workers"
