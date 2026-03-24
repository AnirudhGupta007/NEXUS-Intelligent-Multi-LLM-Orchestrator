from core.config import (
    MODEL_CLASSIFIER, MODEL_LLAMA_GROQ, MODEL_KIMI_K2, MODEL_GPT_OSS,
    MODEL_QWEN_235B, MODEL_GPT4O, MODEL_GEMINI_FLASH, MODEL_OPUS,
    MODEL_COSTS, CATEGORY_BASELINES, JUDGE_THRESHOLD, JUDGE_THRESHOLDS,
    KNN_K_VALUE, MAX_ESCALATIONS, ESCALATION_MODELS,
)
from core.prototypes import MODEL_PROTOTYPES


ALL_WORKER_MODELS = [
    MODEL_LLAMA_GROQ, MODEL_KIMI_K2, MODEL_GPT_OSS,
    MODEL_QWEN_235B, MODEL_GPT4O, MODEL_GEMINI_FLASH, MODEL_OPUS,
]


class TestModelCosts:
    def test_all_models_have_costs(self):
        for model in [MODEL_CLASSIFIER] + ALL_WORKER_MODELS:
            assert model in MODEL_COSTS, f"Missing cost entry for {model}"

    def test_costs_have_input_and_output(self):
        for model, costs in MODEL_COSTS.items():
            assert "input" in costs, f"{model} missing input cost"
            assert "output" in costs, f"{model} missing output cost"
            assert costs["input"] >= 0
            assert costs["output"] >= 0


class TestCategoryBaselines:
    def test_all_benchmark_categories_have_baselines(self):
        from eval.benchmark import QUERIES
        categories = {q["category"] for q in QUERIES}
        for cat in categories:
            assert cat in CATEGORY_BASELINES, f"Missing baseline for category: {cat}"

    def test_baselines_are_positive(self):
        for cat, cost in CATEGORY_BASELINES.items():
            assert cost > 0, f"Baseline for {cat} must be positive"


class TestThresholds:
    def test_judge_threshold_in_range(self):
        assert 0 <= JUDGE_THRESHOLD <= 10

    def test_tiered_thresholds_in_range(self):
        for cat, threshold in JUDGE_THRESHOLDS.items():
            assert 0 <= threshold <= 10, f"Threshold for {cat} out of range"

    def test_knn_k_value_positive(self):
        assert KNN_K_VALUE > 0

    def test_max_escalations_positive(self):
        assert MAX_ESCALATIONS >= 1


class TestEscalationModels:
    def test_escalation_models_exist_in_costs(self):
        for cat, model in ESCALATION_MODELS.items():
            assert model in MODEL_COSTS, f"Escalation model {model} for {cat} not in MODEL_COSTS"


class TestPrototypeModelConsistency:
    def test_all_worker_models_have_prototypes(self):
        for model in ALL_WORKER_MODELS:
            assert model in MODEL_PROTOTYPES, f"Missing prototypes for {model}"

    def test_prototype_models_match_config(self):
        for model in MODEL_PROTOTYPES:
            assert model in MODEL_COSTS, f"Prototype model {model} not in MODEL_COSTS"
