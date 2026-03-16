"""
Test suite for RecommenderAPI.

Tests cover:
  - Health endpoints
  - Movies endpoints (list, get, recommend)
  - Music endpoints (list, get, recommend)
  - Input validation (422 errors)
  - 503 behavior when models are not loaded
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.ml.model_manager import ModelManager
from app.ml.recommender import RecommenderEngine


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_fake_svd(user_ids, item_ids):
    """Create a minimal SVD mock that satisfies the engine."""
    svd = MagicMock()
    svd.predict.side_effect = lambda uid, iid: MagicMock(est=3.5)

    # Build fake trainset internals
    trainset = MagicMock()
    trainset._raw2inner_id_items = {str(iid): i for i, iid in enumerate(item_ids)}
    trainset._raw2inner_id_users = {str(uid): i for i, uid in enumerate(user_ids)}
    svd.trainset = trainset

    # qi: item factor matrix (shape: n_items x n_factors)
    n = len(item_ids)
    svd.qi = np.random.rand(n, 20).astype(np.float32)

    return svd


def _make_fake_model(domain: str) -> dict:
    user_ids = list(range(1, 11))
    item_ids = list(range(1, 21))
    n_users, n_items = len(user_ids), len(item_ids)

    matrix = np.random.rand(n_users, n_items).astype(np.float32)

    if domain == "movies":
        metadata = {
            iid: {"title": f"Movie {iid}", "genres": ["Action", "Drama"]}
            for iid in item_ids
        }
    else:
        metadata = {
            iid: {"title": f"Artist {iid}", "tags": ["rock", "pop"]}
            for iid in item_ids
        }

    return {
        "svd": _make_fake_svd(user_ids, item_ids),
        "interaction_matrix": matrix,
        "user_ids": user_ids,
        "item_ids": item_ids,
        "item_metadata": metadata,
        "domain": domain,
        "algorithm": "SVD",
        "dataset": "Test Dataset",
        "num_users": n_users,
        "num_items": n_items,
        "trained_at": "2024-01-01T00:00:00",
        "metrics": {"rmse": 0.9, "mae": 0.7},
    }


@pytest.fixture(autouse=True)
def load_fake_models():
    """Inject fake models into ModelManager before each test."""
    ModelManager._registry = {
        "movies": _make_fake_model("movies"),
        "music": _make_fake_model("music"),
    }
    yield
    ModelManager.clear_all()


@pytest.fixture
def client():
    return TestClient(app)


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["models_loaded"]["movies"] is True
        assert body["models_loaded"]["music"] is True

    def test_health_models_not_loaded(self, client):
        ModelManager.clear_all()
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["models_loaded"]["movies"] is False
        assert body["models_loaded"]["music"] is False

    def test_model_info_movies(self, client):
        r = client.get("/models/movies")
        assert r.status_code == 200
        body = r.json()
        assert body["domain"] == "movies"
        assert body["num_users"] == 10
        assert body["num_items"] == 20

    def test_model_info_music(self, client):
        r = client.get("/models/music")
        assert r.status_code == 200
        assert r.json()["domain"] == "music"

    def test_model_info_not_found(self, client):
        r = client.get("/models/unknown")
        assert r.status_code == 404


# ── Movies ────────────────────────────────────────────────────────────────────

class TestMovies:
    def test_list_movies(self, client):
        r = client.get("/movies/")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list)
        assert len(body) > 0
        assert "movie_id" in body[0]
        assert "title" in body[0]

    def test_list_movies_pagination(self, client):
        r = client.get("/movies/?skip=0&limit=5")
        assert r.status_code == 200
        assert len(r.json()) <= 5

    def test_get_movie_by_id(self, client):
        r = client.get("/movies/1")
        assert r.status_code == 200
        assert r.json()["movie_id"] == 1

    def test_get_movie_not_found(self, client):
        r = client.get("/movies/99999")
        assert r.status_code == 404

    def test_recommend_user_based(self, client):
        r = client.post("/movies/recommend", json={"user_id": 1, "strategy": "user_based", "top_n": 5})
        assert r.status_code == 200
        body = r.json()
        assert body["strategy"] == "user_based"
        assert body["total"] <= 5
        assert len(body["recommendations"]) <= 5

    def test_recommend_item_based(self, client):
        r = client.post("/movies/recommend", json={"movie_id": 1, "strategy": "item_based", "top_n": 5})
        assert r.status_code == 200
        assert r.json()["strategy"] == "item_based"

    def test_recommend_hybrid(self, client):
        r = client.post("/movies/recommend", json={"user_id": 1, "strategy": "hybrid", "top_n": 10})
        assert r.status_code == 200
        assert r.json()["strategy"] == "hybrid"

    def test_recommend_missing_ids(self, client):
        r = client.post("/movies/recommend", json={"top_n": 5})
        assert r.status_code == 422

    def test_recommend_item_based_missing_movie_id(self, client):
        r = client.post("/movies/recommend", json={"user_id": 1, "strategy": "item_based"})
        assert r.status_code == 422

    def test_recommend_top_n_limit(self, client):
        r = client.post("/movies/recommend", json={"user_id": 1, "top_n": 200})
        assert r.status_code == 422

    def test_recommend_503_when_no_model(self, client):
        ModelManager._registry.pop("movies", None)
        r = client.post("/movies/recommend", json={"user_id": 1})
        assert r.status_code == 503

    def test_recommendation_score_range(self, client):
        r = client.post("/movies/recommend", json={"user_id": 1, "strategy": "hybrid", "top_n": 10})
        for rec in r.json()["recommendations"]:
            assert 0.0 <= rec["score"] <= 1.0


# ── Music ─────────────────────────────────────────────────────────────────────

class TestMusic:
    def test_list_artists(self, client):
        r = client.get("/music/")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list)
        assert "artist_id" in body[0]

    def test_get_artist_by_id(self, client):
        r = client.get("/music/1")
        assert r.status_code == 200
        assert r.json()["artist_id"] == 1

    def test_get_artist_not_found(self, client):
        r = client.get("/music/99999")
        assert r.status_code == 404

    def test_recommend_user_based(self, client):
        r = client.post("/music/recommend", json={"user_id": 1, "strategy": "user_based", "top_n": 5})
        assert r.status_code == 200
        body = r.json()
        assert body["strategy"] == "user_based"

    def test_recommend_item_based(self, client):
        r = client.post("/music/recommend", json={"artist_id": 1, "strategy": "item_based", "top_n": 5})
        assert r.status_code == 200

    def test_recommend_hybrid(self, client):
        r = client.post("/music/recommend", json={"user_id": 2, "strategy": "hybrid", "top_n": 8})
        assert r.status_code == 200

    def test_recommend_missing_ids(self, client):
        r = client.post("/music/recommend", json={"top_n": 5})
        assert r.status_code == 422

    def test_recommend_503_when_no_model(self, client):
        ModelManager._registry.pop("music", None)
        r = client.post("/music/recommend", json={"user_id": 1})
        assert r.status_code == 503


# ── RecommenderEngine unit tests ──────────────────────────────────────────────

class TestRecommenderEngine:
    @pytest.fixture
    def engine(self):
        model = _make_fake_model("movies")
        return RecommenderEngine(
            svd_model=model["svd"],
            interaction_matrix=model["interaction_matrix"],
            item_ids=model["item_ids"],
            user_ids=model["user_ids"],
            item_metadata=model["item_metadata"],
        )

    def test_recommend_for_user_returns_list(self, engine):
        recs = engine.recommend_for_user(user_id=1, top_n=5)
        assert isinstance(recs, list)
        assert len(recs) <= 5

    def test_recommend_for_unknown_user_fallback(self, engine):
        # Unknown user falls back to popular items — should not raise
        recs = engine.recommend_for_user(user_id=9999, top_n=5)
        assert isinstance(recs, list)

    def test_recommend_similar_items(self, engine):
        recs = engine.recommend_similar_items(item_id=1, top_n=5)
        assert isinstance(recs, list)

    def test_recommend_hybrid(self, engine):
        recs = engine.recommend_hybrid(user_id=1, top_n=5)
        assert isinstance(recs, list)
        assert len(recs) <= 5

    def test_scores_are_normalized(self, engine):
        recs = engine.recommend_hybrid(user_id=1, top_n=10)
        for r in recs:
            assert 0.0 <= r["score"] <= 1.0, f"Score out of range: {r['score']}"
