import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def make_mock_completion(content, model="test-model", prompt_tokens=50, completion_tokens=100):
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    mock.model = model
    mock.usage = MagicMock()
    mock.usage.prompt_tokens = prompt_tokens
    mock.usage.completion_tokens = completion_tokens
    return mock


@pytest.fixture
def mock_knn_index():
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
        center[i * 200 : (i + 1) * 200] = 1.0
        for _ in range(10):
            vec = center + np.random.randn(1536) * 0.1
            vec = vec / np.linalg.norm(vec)
            all_vectors.append(vec)
            all_labels.append(model)
    return {"all_vectors": np.array(all_vectors), "all_labels": all_labels}
