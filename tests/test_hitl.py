import pytest
from unittest.mock import patch
from agents.hitl import hitl_node


class TestHITLNode:
    def test_uses_clarifying_question(self):
        with patch("agents.hitl.interrupt", return_value="I want Python"):
            result = hitl_node({"query": "sort this", "clarifying_question": "Which language?", "conversation_turns": 0, "total_cost": 0.0, "total_latency": 0.0})
        assert result["user_clarification"] == "I want Python"
        assert "[Clarified: I want Python]" in result["enriched_query"]
        assert result["conversation_turns"] == 1

    def test_fallback_question(self):
        with patch("agents.hitl.interrupt", return_value="more details"):
            result = hitl_node({"query": "do thing", "clarifying_question": "", "conversation_turns": 0, "total_cost": 0.0, "total_latency": 0.0})
        assert "ambiguous" in result["clarifying_question"]

    def test_none_answer_default(self):
        with patch("agents.hitl.interrupt", return_value=None):
            result = hitl_node({"query": "sort", "clarifying_question": "Language?", "conversation_turns": 0, "total_cost": 0.0, "total_latency": 0.0})
        assert result["user_clarification"] == "please proceed with best guess"

    def test_preserves_metrics(self):
        with patch("agents.hitl.interrupt", return_value="answer"):
            result = hitl_node({"query": "t", "clarifying_question": "Q?", "conversation_turns": 0, "total_cost": 0.005, "total_latency": 1.5})
        assert result["total_cost"] == 0.005
        assert result["total_latency"] == 1.5
