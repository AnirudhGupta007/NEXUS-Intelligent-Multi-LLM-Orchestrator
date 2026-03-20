import pytest
from core.graph import route_from_classifier, route_from_knn, route_from_worker, route_from_aggregator, route_from_judge, set_final
from langgraph.graph import END


class TestRouteFromClassifier:
    def test_self_answer_exits(self):
        assert route_from_classifier({"can_self_answer": True, "is_ambiguous": False}) == END
    def test_ambiguous_goes_to_hitl(self):
        assert route_from_classifier({"can_self_answer": False, "is_ambiguous": True}) == "hitl"
    def test_normal_goes_to_router(self):
        assert route_from_classifier({"can_self_answer": False, "is_ambiguous": False}) == "knn_router"


class TestRouteFromKNN:
    def test_subtasks_trigger_parallel(self):
        assert route_from_knn({"subtasks": ["t1", "t2"]}) == "parallel_worker"
    def test_no_subtasks_trigger_single(self):
        assert route_from_knn({"subtasks": []}) == "worker"
    def test_missing_subtasks(self):
        assert route_from_knn({}) == "worker"


class TestRouteFromWorker:
    def test_critical_goes_to_judge(self):
        assert route_from_worker({"is_critical": True}) == "judge"
    def test_non_critical_goes_to_final(self):
        assert route_from_worker({"is_critical": False}) == "set_final"


class TestRouteFromAggregator:
    def test_critical_goes_to_judge(self):
        assert route_from_aggregator({"is_critical": True}) == "judge"
    def test_non_critical_goes_to_final(self):
        assert route_from_aggregator({"is_critical": False}) == "set_final"


class TestRouteFromJudge:
    def test_escalation_on_failure(self):
        assert route_from_judge({"escalation_model": "opus", "escalation_count": 0}) == "escalation_worker"
    def test_no_escalation_at_limit(self):
        assert route_from_judge({"escalation_model": "opus", "escalation_count": 1}) == "set_final"
    def test_passed_goes_to_final(self):
        assert route_from_judge({"escalation_count": 0}) == "set_final"


class TestSetFinal:
    def test_aggregated_preferred(self):
        result = set_final({"aggregated_response": "merged", "worker_responses": [{"response": "x"}], "total_cost": 0.005, "total_latency": 2.0, "selected_models": ["a"], "escalation_count": 0, "escalation_model": None})
        assert result["final_response"] == "merged"
    def test_worker_fallback(self):
        result = set_final({"aggregated_response": None, "worker_responses": [{"response": "direct"}], "total_cost": 0.0, "total_latency": 0.0, "selected_models": [], "escalation_count": 0, "escalation_model": None})
        assert result["final_response"] == "direct"
    def test_no_response(self):
        result = set_final({"aggregated_response": None, "worker_responses": [], "final_response": None, "total_cost": 0.0, "total_latency": 0.0, "selected_models": [], "escalation_count": 0, "escalation_model": None})
        assert result["final_response"] == "No response generated."
