import json
import pytest
from unittest.mock import patch, AsyncMock
from agents.classifier import classifier_node, _is_greeting_or_smalltalk


class TestGreetingDetection:
    def test_basic_greetings(self):
        assert _is_greeting_or_smalltalk("hello") is True
        assert _is_greeting_or_smalltalk("hi") is True
        assert _is_greeting_or_smalltalk("hey") is True
        assert _is_greeting_or_smalltalk("good morning") is True

    def test_greeting_with_prefix(self):
        assert _is_greeting_or_smalltalk("hi there") is True
        assert _is_greeting_or_smalltalk("hello world") is True

    def test_non_greetings(self):
        assert _is_greeting_or_smalltalk("write quicksort in python") is False
        assert _is_greeting_or_smalltalk("what is quantum computing?") is False
        assert _is_greeting_or_smalltalk("") is False

    def test_case_insensitive(self):
        assert _is_greeting_or_smalltalk("Hello") is True
        assert _is_greeting_or_smalltalk("HI") is True


class TestClassifierNode:
    @pytest.mark.asyncio
    async def test_greeting_self_answers(self):
        from tests.conftest import make_mock_completion
        mock_resp = make_mock_completion(json.dumps({
            "can_self_answer": True, "self_answer": "Hello! How can I help?",
            "is_ambiguous": False, "clarifying_question": None, "is_critical": False, "subtasks": [],
        }))
        with patch("agents.classifier.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            with patch("agents.classifier.calculate_cost", return_value=0.0001):
                result = await classifier_node({"query": "hello", "conversation_turns": 0, "total_cost": 0.0, "total_latency": 0.0})
        assert result["can_self_answer"] is True
        assert result["final_response"] == "Hello! How can I help?"

    @pytest.mark.asyncio
    async def test_complex_query_not_self_answered(self):
        from tests.conftest import make_mock_completion
        mock_resp = make_mock_completion(json.dumps({
            "can_self_answer": True, "self_answer": "Here...",
            "is_ambiguous": False, "clarifying_question": None, "is_critical": False, "subtasks": [],
        }))
        with patch("agents.classifier.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            with patch("agents.classifier.calculate_cost", return_value=0.0001):
                result = await classifier_node({"query": "write quicksort in python", "conversation_turns": 0, "total_cost": 0.0, "total_latency": 0.0})
        assert result["can_self_answer"] is False

    @pytest.mark.asyncio
    async def test_ambiguous_query_detected(self):
        from tests.conftest import make_mock_completion
        mock_resp = make_mock_completion(json.dumps({
            "can_self_answer": False, "self_answer": None,
            "is_ambiguous": True, "clarifying_question": "Which language?", "is_critical": False, "subtasks": [],
        }))
        with patch("agents.classifier.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            with patch("agents.classifier.calculate_cost", return_value=0.0001):
                result = await classifier_node({"query": "sort this", "conversation_turns": 0, "total_cost": 0.0, "total_latency": 0.0})
        assert result["is_ambiguous"] is True
        assert result["clarifying_question"] == "Which language?"

    @pytest.mark.asyncio
    async def test_ambiguous_disabled_after_clarification(self):
        from tests.conftest import make_mock_completion
        mock_resp = make_mock_completion(json.dumps({
            "can_self_answer": False, "self_answer": None,
            "is_ambiguous": True, "clarifying_question": "What?", "is_critical": False, "subtasks": [],
        }))
        with patch("agents.classifier.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            with patch("agents.classifier.calculate_cost", return_value=0.0001):
                result = await classifier_node({"query": "sort", "conversation_turns": 1, "total_cost": 0.0, "total_latency": 0.0})
        assert result["is_ambiguous"] is False

    @pytest.mark.asyncio
    async def test_critical_query_flagged(self):
        from tests.conftest import make_mock_completion
        mock_resp = make_mock_completion(json.dumps({
            "can_self_answer": False, "self_answer": None,
            "is_ambiguous": False, "clarifying_question": None, "is_critical": True, "subtasks": [],
        }))
        with patch("agents.classifier.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            with patch("agents.classifier.calculate_cost", return_value=0.0001):
                result = await classifier_node({"query": "Is mixing drugs safe?", "conversation_turns": 0, "total_cost": 0.0, "total_latency": 0.0})
        assert result["is_critical"] is True

    @pytest.mark.asyncio
    async def test_llm_failure_fallback(self):
        with patch("agents.classifier.litellm.acompletion", new_callable=AsyncMock, side_effect=Exception("API down")):
            result = await classifier_node({"query": "test", "conversation_turns": 0, "total_cost": 0.0, "total_latency": 0.0})
        assert result["can_self_answer"] is False
        assert result["is_ambiguous"] is False

    @pytest.mark.asyncio
    async def test_trace_entry_added(self):
        from tests.conftest import make_mock_completion
        mock_resp = make_mock_completion(json.dumps({
            "can_self_answer": False, "self_answer": None,
            "is_ambiguous": False, "clarifying_question": None, "is_critical": False, "subtasks": [],
        }))
        with patch("agents.classifier.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            with patch("agents.classifier.calculate_cost", return_value=0.0001):
                result = await classifier_node({"query": "test", "conversation_turns": 0, "total_cost": 0.0, "total_latency": 0.0})
        assert len(result["trace"]) == 1
        assert result["trace"][0]["node"] == "classifier"
