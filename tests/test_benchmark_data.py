from eval.benchmark import QUERIES
from core.config import (
    MODEL_LLAMA_GROQ, MODEL_KIMI_K2, MODEL_GPT_OSS,
    MODEL_QWEN_235B, MODEL_GPT4O, MODEL_GEMINI_FLASH, MODEL_OPUS,
    MODEL_COSTS, CATEGORY_BASELINES,
)


ALL_MODELS = [
    MODEL_LLAMA_GROQ, MODEL_KIMI_K2, MODEL_GPT_OSS,
    MODEL_QWEN_235B, MODEL_GPT4O, MODEL_GEMINI_FLASH, MODEL_OPUS,
]


class TestBenchmarkQueries:
    def test_all_queries_have_expected_model(self):
        for item in QUERIES:
            assert "expected" in item, f"Missing expected model: {item['q']}"
            assert item["expected"] in ALL_MODELS, f"Unknown model {item['expected']} for: {item['q']}"

    def test_all_queries_have_category(self):
        for item in QUERIES:
            assert "category" in item, f"Missing category: {item['q']}"
            assert isinstance(item["category"], str) and len(item["category"]) > 0

    def test_all_queries_have_text(self):
        for item in QUERIES:
            assert "q" in item
            assert isinstance(item["q"], str) and len(item["q"].strip()) > 0

    def test_categories_cover_all_models(self):
        models_in_benchmark = {item["expected"] for item in QUERIES}
        for model in ALL_MODELS:
            assert model in models_in_benchmark, f"No benchmark queries target {model}"

    def test_no_duplicate_queries(self):
        queries = [item["q"] for item in QUERIES]
        assert len(queries) == len(set(queries)), "Duplicate benchmark queries found"

    def test_category_baselines_exist(self):
        categories = {item["category"] for item in QUERIES}
        for cat in categories:
            assert cat in CATEGORY_BASELINES, f"Missing baseline for category: {cat}"

    def test_reasonable_query_count(self):
        assert len(QUERIES) >= 50, f"Only {len(QUERIES)} queries — need at least 50 for meaningful benchmarks"

    def test_minimum_queries_per_category(self):
        from collections import Counter
        counts = Counter(item["category"] for item in QUERIES)
        for cat, count in counts.items():
            assert count >= 5, f"Category {cat} has only {count} queries, need at least 5"
