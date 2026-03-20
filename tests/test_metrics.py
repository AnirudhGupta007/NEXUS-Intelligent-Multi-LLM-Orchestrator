import pytest
from unittest.mock import MagicMock, patch
from core.metrics import calculate_cost
from core.config import MODEL_GPT4O, MODEL_LLAMA_GROQ


class TestCalculateCost:
    def test_litellm_cost_used(self):
        with patch("core.metrics.litellm.completion_cost", return_value=0.005):
            assert calculate_cost(MODEL_GPT4O, MagicMock()) == 0.005

    def test_fallback_manual(self):
        mock = MagicMock()
        mock.usage.prompt_tokens = 1000
        mock.usage.completion_tokens = 500
        with patch("core.metrics.litellm.completion_cost", return_value=0):
            cost = calculate_cost(MODEL_GPT4O, mock)
        assert abs(cost - 0.0075) < 0.0001

    def test_zero_on_no_usage(self):
        mock = MagicMock()
        mock.usage = None
        with patch("core.metrics.litellm.completion_cost", return_value=0):
            assert calculate_cost("unknown", mock) == 0.0

    def test_exception_fallback(self):
        mock = MagicMock()
        mock.usage.prompt_tokens = 100
        mock.usage.completion_tokens = 100
        with patch("core.metrics.litellm.completion_cost", side_effect=Exception("err")):
            cost = calculate_cost(MODEL_LLAMA_GROQ, mock)
        assert cost > 0
