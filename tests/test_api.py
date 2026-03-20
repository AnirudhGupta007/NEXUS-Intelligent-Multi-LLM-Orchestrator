import pytest
import numpy as np
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    import agents.knn_router as knn_mod
    np.random.seed(42)
    knn_mod.KNN_INDEX = {"all_vectors": np.random.randn(70, 1536), "all_labels": ["groq/llama-3.1-8b-instant"] * 70}

    # Patch the lifespan to skip real KNN index build
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def mock_lifespan(app):
        yield

    import api.main as api_module
    original_lifespan = api_module.app.router.lifespan_context
    api_module.app.router.lifespan_context = mock_lifespan

    with TestClient(api_module.app, raise_server_exceptions=False) as c:
        yield c

    api_module.app.router.lifespan_context = original_lifespan
    knn_mod.KNN_INDEX = None


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["knn_index_loaded"] is True


class TestModelsEndpoint:
    def test_models_list(self, client):
        resp = client.get("/models")
        assert resp.status_code == 200
        assert len(resp.json()["prototypes"]) == 7
        assert resp.json()["baseline_model"] == "gpt-5"


class TestResumeEndpoint:
    def test_missing_session(self, client):
        resp = client.post("/resume", json={"session_id": "nope", "answer": "test"})
        assert resp.json()["error"] == "Session not found"
