"""
Core recommendation engine.

Supports:
  - user_based:  SVD predictions for unseen items per user
  - item_based:  cosine similarity between item vectors
  - hybrid:      weighted combination of both
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RecommenderEngine:
    """
    Wraps a trained Surprise SVD model and the interaction matrix
    to provide user-based, item-based, and hybrid recommendations.
    """

    def __init__(self, svd_model, interaction_matrix: np.ndarray,
                 item_ids: list, user_ids: list, item_metadata: dict):
        self.svd = svd_model
        self.interaction_matrix = interaction_matrix          # shape (n_users, n_items)
        self.item_ids = item_ids                              # list of raw item ids
        self.user_ids = user_ids                              # list of raw user ids
        self.item_metadata = item_metadata                    # {item_id: {title, genres/tags, ...}}
        self._item_similarity: Optional[np.ndarray] = None   # computed lazily

    # ── Item similarity matrix (cosine on item vectors from SVD) ─────────────

    def _get_item_similarity(self) -> np.ndarray:
        if self._item_similarity is None:
            item_factors = np.array([
                self.svd.qi[self.svd.trainset.to_inner_iid(str(iid))]
                for iid in self.item_ids
                if str(iid) in self.svd.trainset._raw2inner_id_items
            ])
            self._item_similarity = cosine_similarity(item_factors)
        return self._item_similarity

    # ── User-based ────────────────────────────────────────────────────────────

    def recommend_for_user(self, user_id: int, top_n: int = 10) -> list[dict]:
        """Predict ratings for unseen items and return top-N."""
        if user_id not in self.user_ids:
            logger.warning(f"Unknown user_id={user_id}, falling back to popular items")
            return self._popular_items(top_n)

        user_idx = self.user_ids.index(user_id)
        seen = set(np.where(self.interaction_matrix[user_idx] > 0)[0])

        predictions = []
        for idx, iid in enumerate(self.item_ids):
            if idx in seen:
                continue
            try:
                pred = self.svd.predict(str(user_id), str(iid))
                predictions.append((iid, pred.est))
            except Exception:
                pass

        predictions.sort(key=lambda x: x[1], reverse=True)
        return self._format(predictions[:top_n], max_score=5.0)

    # ── Item-based ────────────────────────────────────────────────────────────

    def recommend_similar_items(self, item_id: int, top_n: int = 10) -> list[dict]:
        """Return items most similar to the given item."""
        valid_ids = [
            iid for iid in self.item_ids
            if str(iid) in self.svd.trainset._raw2inner_id_items
        ]
        if item_id not in valid_ids:
            logger.warning(f"Unknown item_id={item_id}")
            return []

        sim_matrix = self._get_item_similarity()
        idx = valid_ids.index(item_id)
        scores = sim_matrix[idx]
        top_indices = np.argsort(scores)[::-1][1:top_n + 1]

        results = [(valid_ids[i], float(scores[i])) for i in top_indices]
        return self._format(results, max_score=1.0)

    # ── Hybrid ────────────────────────────────────────────────────────────────

    def recommend_hybrid(self, user_id: int, top_n: int = 10,
                         alpha: float = 0.6) -> list[dict]:
        """
        Blend user-based (alpha) and item-based popularity signals (1-alpha).
        """
        user_recs = {r["id"]: r["score"] for r in self.recommend_for_user(user_id, top_n * 2)}
        popular = {r["id"]: r["score"] for r in self._popular_items(top_n * 2)}

        all_ids = set(user_recs) | set(popular)
        blended = {
            iid: alpha * user_recs.get(iid, 0.0) + (1 - alpha) * popular.get(iid, 0.0)
            for iid in all_ids
        }

        top = sorted(blended.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return self._format(top, max_score=1.0)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _popular_items(self, top_n: int) -> list[dict]:
        """Fallback: items ranked by interaction count."""
        counts = self.interaction_matrix.sum(axis=0)
        top_indices = np.argsort(counts)[::-1][:top_n]
        max_count = counts[top_indices[0]] or 1
        results = [(self.item_ids[i], counts[i] / max_count) for i in top_indices]
        return self._format(results, max_score=1.0)

    def _format(self, pairs: list[tuple], max_score: float) -> list[dict]:
        out = []
        for iid, raw_score in pairs:
            meta = self.item_metadata.get(iid, {})
            out.append({
                "id": iid,
                "title": meta.get("title", str(iid)),
                "score": round(float(raw_score) / max_score, 4),
                "genres": meta.get("genres") or meta.get("tags"),
            })
        return out
