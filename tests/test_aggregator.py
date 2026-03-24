import pytest
from unittest.mock import patch, AsyncMock
from agents.aggregator import aggregator_node
from tests.conftest import make_mock_completion


class TestAggregatorNode:
    @pytest.mark.asyncio
    async def test_merges_multiple_responses(self):
        mock_resp = make_mock_completion("Merged answer combining all insights.")
        with patch("agents.aggregator.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            with patch("agents.aggregator.calculate_cost", return_value=0.0002):
                result = await aggregator_node({
                    "query": "complex question",
                    "worker_responses": [
                        {"response": "Part A answer"},
                        {"response": "Part B answer"},
                        {"response": "Part C answer"},
                    ],
                    "subtasks": ["Part A", "Part B", "Part C"],
                    "total_cost": 0.001,
                    "total_latency": 1.0,
                })
        assert result["aggregated_response"] == "Merged answer combining all insights."
        assert result["trace"][0]["node"] == "aggregator"

    @pytest.mark.asyncio
    async def test_handles_empty_responses(self):
        mock_resp = make_mock_completion("No data to merge.")
        with patch("agents.aggregator.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            with patch("agents.aggregator.calculate_cost", return_value=0.0001):
                result = await aggregator_node({
                    "query": "test",
                    "worker_responses": [],
                    "subtasks": [],
                    "total_cost": 0.0,
                    "total_latency": 0.0,
                })
        assert "aggregated_response" in result

    @pytest.mark.asyncio
    async def test_llm_failure_returns_raw(self):
        with patch("agents.aggregator.litellm.acompletion", new_callable=AsyncMock, side_effect=Exception("API down")):
            result = await aggregator_node({
                "query": "test",
                "worker_responses": [{"response": "raw output 1"}, {"response": "raw output 2"}],
                "subtasks": ["task1", "task2"],
                "total_cost": 0.0,
                "total_latency": 0.0,
            })
        assert "raw output 1" in result["aggregated_response"]
        assert "raw output 2" in result["aggregated_response"]

    @pytest.mark.asyncio
    async def test_cost_and_latency_tracked(self):
        mock_resp = make_mock_completion("merged")
        with patch("agents.aggregator.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            with patch("agents.aggregator.calculate_cost", return_value=0.0005):
                result = await aggregator_node({
                    "query": "test",
                    "worker_responses": [{"response": "a"}],
                    "subtasks": ["t1"],
                    "total_cost": 0.001,
                    "total_latency": 2.0,
                })
        assert result["total_cost"] > 0.001
        assert result["total_latency"] >= 2.0

    @pytest.mark.asyncio
    async def test_subtask_labels_in_prompt(self):
        """Verify subtask names are passed as labels to the LLM prompt."""
        captured_messages = []

        async def capture_call(**kwargs):
            captured_messages.append(kwargs["messages"])
            return make_mock_completion("merged")

        with patch("agents.aggregator.litellm.acompletion", side_effect=capture_call):
            with patch("agents.aggregator.calculate_cost", return_value=0.0):
                await aggregator_node({
                    "query": "test",
                    "worker_responses": [{"response": "ans1"}, {"response": "ans2"}],
                    "subtasks": ["Design DB", "Write API"],
                    "total_cost": 0.0,
                    "total_latency": 0.0,
                })
        user_msg = captured_messages[0][1]["content"]
        assert "Design DB" in user_msg
        assert "Write API" in user_msg
