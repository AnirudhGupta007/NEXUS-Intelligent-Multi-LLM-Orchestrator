from core.prototypes import MODEL_PROTOTYPES


class TestPrototypes:
    def test_all_models_have_prototypes(self):
        assert len(MODEL_PROTOTYPES) == 7

    def test_minimum_prototypes_per_model(self):
        for model, queries in MODEL_PROTOTYPES.items():
            assert len(queries) >= 10, f"{model} has only {len(queries)} prototypes, need at least 10"

    def test_expanded_prototypes_per_model(self):
        for model, queries in MODEL_PROTOTYPES.items():
            assert len(queries) == 15, f"{model} has {len(queries)} prototypes, expected 15"

    def test_no_duplicate_prototypes_within_model(self):
        for model, queries in MODEL_PROTOTYPES.items():
            assert len(queries) == len(set(queries)), f"Duplicate prototypes in {model}"

    def test_no_duplicate_prototypes_across_models(self):
        all_queries = []
        for queries in MODEL_PROTOTYPES.values():
            all_queries.extend(queries)
        assert len(all_queries) == len(set(all_queries)), "Duplicate prototypes found across models"

    def test_total_prototype_count(self):
        total = sum(len(q) for q in MODEL_PROTOTYPES.values())
        assert total == 7 * 15  # 105 total

    def test_prototypes_are_non_empty_strings(self):
        for model, queries in MODEL_PROTOTYPES.items():
            for q in queries:
                assert isinstance(q, str) and len(q.strip()) > 0, f"Empty prototype in {model}"
