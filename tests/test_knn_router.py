import pytest
import numpy as np
from unittest.mock import patch, AsyncMock, MagicMock
from core.config import MODEL_LLAMA_GROQ, MODEL_KIMI_K2
from agents.knn_router import knn_router_node, semantic_route, build_knn_index


class TestSemanticRoute:
    @pytest.mark.asyncio
    async def test_routes_to_correct_cluster(self, mock_knn_index):
        query_vec = np.zeros(1536)
        query_vec[0:200] = 1.0
        query_vec = query_vec / np.linalg.norm(query_vec)
        mock_embed = MagicMock()
        mock_embed.data = [{"embedding": query_vec.tolist()}]
        with patch("agents.knn_router.litellm.aembedding", new_callable=AsyncMock, return_value=mock_embed):
            model, scores = await semantic_route("hello", mock_knn_index)
        assert model == MODEL_LLAMA_GROQ
        assert isinstance(scores, dict)

    @pytest.mark.asyncio
    async def test_routes_code_to_kimi(self, mock_knn_index):
        query_vec = np.zeros(1536)
        query_vec[200:400] = 1.0
        query_vec = query_vec / np.linalg.norm(query_vec)
        mock_embed = MagicMock()
        mock_embed.data = [{"embedding": query_vec.tolist()}]
        with patch("agents.knn_router.litellm.aembedding", new_callable=AsyncMock, return_value=mock_embed):
            model, _ = await semantic_route("write quicksort", mock_knn_index)
        assert model == MODEL_KIMI_K2


class TestKNNRouterNode:
    @pytest.mark.asyncio
    async def test_single_query_routing(self, mock_knn_index):
        import agents.knn_router as knn_mod
        original = knn_mod.KNN_INDEX
        knn_mod.KNN_INDEX = mock_knn_index
        query_vec = np.zeros(1536)
        query_vec[0:200] = 1.0
        query_vec = query_vec / np.linalg.norm(query_vec)
        mock_embed = MagicMock()
        mock_embed.data = [{"embedding": query_vec.tolist()}]
        try:
            with patch("agents.knn_router.litellm.aembedding", new_callable=AsyncMock, return_value=mock_embed):
                result = await knn_router_node({"query": "hello", "subtasks": [], "total_cost": 0.0, "total_latency": 0.0})
            assert len(result["selected_models"]) == 1
            assert result["trace"][0]["node"] == "knn_router"
        finally:
            knn_mod.KNN_INDEX = original

    @pytest.mark.asyncio
    async def test_no_index_returns_error(self):
        import agents.knn_router as knn_mod
        original = knn_mod.KNN_INDEX
        knn_mod.KNN_INDEX = None
        try:
            result = await knn_router_node({"query": "test", "subtasks": [], "total_cost": 0.0, "total_latency": 0.0})
            assert "error" in result
        finally:
            knn_mod.KNN_INDEX = original


class TestBuildKNNIndex:
    @pytest.mark.asyncio
    async def test_builds_correct_shape(self):
        from core.prototypes import MODEL_PROTOTYPES
        total = sum(len(v) for v in MODEL_PROTOTYPES.values())
        async def mock_embed_fn(model, input):
            mock = MagicMock()
            mock.data = [{"embedding": np.random.randn(1536).tolist()} for _ in input]
            return mock
        with patch("agents.knn_router.litellm.aembedding", side_effect=mock_embed_fn):
            index = await build_knn_index()
        assert index["all_vectors"].shape[0] == total
        assert index["all_vectors"].shape[1] == 1536
