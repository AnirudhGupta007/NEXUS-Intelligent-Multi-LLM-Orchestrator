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


    @pytest.mark.asyncio
    async def test_worker_fallback_model(self):
        """Missing selected_models uses default Llama."""
        mock_resp = make_mock_completion("fallback response")
        with patch("agents.worker.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            with patch("agents.worker.calculate_cost", return_value=0.0001):
                result = await worker_node({"query": "test", "selected_models": [], "total_cost": 0.0, "total_latency": 0.0})
        assert result["worker_responses"][0]["model"] == "groq/llama-3.1-8b-instant"

    @pytest.mark.asyncio
    async def test_worker_error_handling(self):
        """Generic exception returns error string in response."""
        with patch("agents.worker.litellm.acompletion", new_callable=AsyncMock, side_effect=ValueError("bad model")):
            result = await worker_node({"query": "test", "selected_models": ["bad-model"], "total_cost": 0.0, "total_latency": 0.0})
        assert "Error" in result["worker_responses"][0]["response"]
        assert result["worker_responses"][0]["cost_usd"] == 0.0


class TestParallelWorkerNode:
    @pytest.mark.asyncio
    async def test_dispatches_all_subtasks(self):
        mock_resp = make_mock_completion("subtask response")
        with patch("agents.worker.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            with patch("agents.worker.calculate_cost", return_value=0.0001):
                result = await parallel_worker_node({"query": "complex", "subtasks": ["a", "b", "c"], "selected_models": ["m1", "m2", "m3"], "total_cost": 0.0, "total_latency": 0.0})
        assert len(result["worker_responses"]) == 3
        assert result["trace"][0]["node"] == "parallel_workers"

    @pytest.mark.asyncio
    async def test_parallel_worker_cost_accumulation(self):
        """Total cost should be sum of all subtask costs."""
        mock_resp = make_mock_completion("response")
        with patch("agents.worker.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            with patch("agents.worker.calculate_cost", return_value=0.001):
                result = await parallel_worker_node({
                    "query": "complex", "subtasks": ["a", "b", "c"],
                    "selected_models": ["m1", "m2", "m3"],
                    "total_cost": 0.0, "total_latency": 0.0,
                })
        # 3 subtasks * 0.001 each = 0.003
        assert result["total_cost"] == pytest.approx(0.003, abs=0.0005)

    @pytest.mark.asyncio
    async def test_parallel_worker_handles_exception(self):
        """One subtask exception should not break other subtasks."""
        call_count = 0

        async def flaky_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("model overloaded")
            return make_mock_completion("success")

        with patch("agents.worker.litellm.acompletion", side_effect=flaky_completion):
            with patch("agents.worker.calculate_cost", return_value=0.0001):
                result = await parallel_worker_node({
                    "query": "test", "subtasks": ["a", "b", "c"],
                    "selected_models": ["m1", "m2", "m3"],
                    "total_cost": 0.0, "total_latency": 0.0,
                })
        assert len(result["worker_responses"]) == 3
        # At least one should have succeeded
        success_count = sum(1 for r in result["worker_responses"] if "Error" not in r["response"])
        assert success_count >= 2
