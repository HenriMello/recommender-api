#!/usr/bin/env python3
"""
Train recommendation models for movies (MovieLens) and music (Last.fm).

Usage:
    python scripts/train.py --domain movies
    python scripts/train.py --domain music
    python scripts/train.py --domain all
"""

import argparse
import pickle
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ── Movies (MovieLens 100K) ───────────────────────────────────────────────────

def train_movies():
    logger.info("📥 Loading MovieLens 100K dataset...")
    try:
        from surprise import Dataset, SVD
        from surprise.model_selection import cross_validate
        import pandas as pd
    except ImportError:
        raise ImportError("Install: pip install scikit-surprise pandas")

    # Load dataset
    data = Dataset.load_builtin("ml-100k", prompt=False)
    trainset = data.build_full_trainset()

    logger.info("🧠 Training SVD model for movies...")
    svd = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    svd.fit(trainset)

    # Cross-validate
    logger.info("📊 Running cross-validation...")
    results = cross_validate(svd, data, measures=["RMSE", "MAE"], cv=5, verbose=False)
    logger.info(f"   RMSE: {np.mean(results['test_rmse']):.4f} ± {np.std(results['test_rmse']):.4f}")
    logger.info(f"   MAE:  {np.mean(results['test_mae']):.4f} ± {np.std(results['test_mae']):.4f}")

    # Build interaction matrix
    raw_ratings = data.raw_ratings
    df = pd.DataFrame(raw_ratings, columns=["user_id", "item_id", "rating", "timestamp"])
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)

    user_ids = sorted(df["user_id"].unique().tolist())
    item_ids = sorted(df["item_id"].unique().tolist())
    user_idx = {u: i for i, u in enumerate(user_ids)}
    item_idx = {it: i for i, it in enumerate(item_ids)}

    matrix = np.zeros((len(user_ids), len(item_ids)), dtype=np.float32)
    for _, row in df.iterrows():
        matrix[user_idx[row["user_id"]], item_idx[row["item_id"]]] = row["rating"]

    # Item metadata (titles and genres from MovieLens)
    item_metadata = _load_movielens_metadata()

    payload = {
        "svd": svd,
        "interaction_matrix": matrix,
        "user_ids": user_ids,
        "item_ids": item_ids,
        "item_metadata": item_metadata,
        "domain": "movies",
        "algorithm": "SVD (Collaborative Filtering)",
        "dataset": "MovieLens 100K",
        "num_users": len(user_ids),
        "num_items": len(item_ids),
        "trained_at": datetime.utcnow().isoformat(),
        "metrics": {
            "rmse": float(np.mean(results["test_rmse"])),
            "mae": float(np.mean(results["test_mae"])),
        },
    }

    path = MODELS_DIR / "movies_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    logger.info(f"✅ Movies model saved → {path}")


def _load_movielens_metadata() -> dict:
    """Load MovieLens u.item file for titles and genres."""
    from surprise import get_dataset_dir

    genre_labels = [
        "unknown", "Action", "Adventure", "Animation", "Children",
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
        "Sci-Fi", "Thriller", "War", "Western",
    ]

    item_file = Path(get_dataset_dir()) / "ml-100k" / "ml-100k" / "u.item"
    metadata = {}
    if item_file.exists():
        with open(item_file, encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) < 2:
                    continue
                movie_id = int(parts[0])
                title = parts[1]
                genre_flags = parts[5:] if len(parts) > 5 else []
                genres = [g for g, flag in zip(genre_labels, genre_flags) if flag == "1"]
                metadata[movie_id] = {"title": title, "genres": genres}
    return metadata


# ── Music (Last.fm synthetic fallback) ───────────────────────────────────────

def train_music():
    logger.info("📥 Preparing music dataset...")
    try:
        from surprise import Dataset, Reader, SVD
        from surprise.model_selection import cross_validate
        import pandas as pd
    except ImportError:
        raise ImportError("Install: pip install scikit-surprise pandas")

    # Use Last.fm 360K if available, otherwise generate synthetic data
    data_path = Path("data/lastfm_ratings.csv")
    if data_path.exists():
        logger.info("   Using Last.fm dataset from data/lastfm_ratings.csv")
        df = pd.read_csv(data_path)
        df.columns = ["user_id", "artist_id", "plays"]
        # Normalize plays to 1-5 rating scale
        df["rating"] = 1 + 4 * (df["plays"] - df["plays"].min()) / (df["plays"].max() - df["plays"].min())
    else:
        logger.info("   Last.fm CSV not found → generating synthetic dataset")
        df = _generate_synthetic_music()

    reader = Reader(rating_scale=(1, 5))
    surprise_data = Dataset.load_from_df(df[["user_id", "artist_id", "rating"]], reader)
    trainset = surprise_data.build_full_trainset()

    logger.info("🧠 Training SVD model for music...")
    svd = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    svd.fit(trainset)

    logger.info("📊 Running cross-validation...")
    results = cross_validate(svd, surprise_data, measures=["RMSE", "MAE"], cv=5, verbose=False)
    logger.info(f"   RMSE: {np.mean(results['test_rmse']):.4f}")
    logger.info(f"   MAE:  {np.mean(results['test_mae']):.4f}")

    user_ids = sorted(df["user_id"].unique().tolist())
    artist_ids = sorted(df["artist_id"].unique().tolist())
    user_idx = {u: i for i, u in enumerate(user_ids)}
    item_idx = {it: i for i, it in enumerate(artist_ids)}

    matrix = np.zeros((len(user_ids), len(artist_ids)), dtype=np.float32)
    for _, row in df.iterrows():
        matrix[user_idx[row["user_id"]], item_idx[row["artist_id"]]] = row["rating"]

    item_metadata = _generate_artist_metadata(artist_ids)

    payload = {
        "svd": svd,
        "interaction_matrix": matrix,
        "user_ids": user_ids,
        "item_ids": artist_ids,
        "item_metadata": item_metadata,
        "domain": "music",
        "algorithm": "SVD (Collaborative Filtering)",
        "dataset": "Last.fm (synthetic fallback)",
        "num_users": len(user_ids),
        "num_items": len(artist_ids),
        "trained_at": datetime.utcnow().isoformat(),
        "metrics": {
            "rmse": float(np.mean(results["test_rmse"])),
            "mae": float(np.mean(results["test_mae"])),
        },
    }

    path = MODELS_DIR / "music_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    logger.info(f"✅ Music model saved → {path}")


def _generate_synthetic_music():
    import pandas as pd

    np.random.seed(42)
    n_users, n_artists = 500, 200
    n_interactions = 8000

    user_ids = np.random.randint(1, n_users + 1, n_interactions)
    artist_ids = np.random.randint(1, n_artists + 1, n_interactions)
    ratings = np.clip(np.random.normal(3.5, 1.0, n_interactions), 1, 5).round(1)

    return pd.DataFrame({"user_id": user_ids, "artist_id": artist_ids, "rating": ratings})


def _generate_artist_metadata(artist_ids: list) -> dict:
    artists = [
        ("The Beatles", ["rock", "pop", "classic rock"]),
        ("Radiohead", ["alternative", "art rock", "electronic"]),
        ("Daft Punk", ["electronic", "french house", "dance"]),
        ("Miles Davis", ["jazz", "bebop", "modal jazz"]),
        ("Kendrick Lamar", ["hip-hop", "rap", "conscious rap"]),
        ("Björk", ["art pop", "electronic", "experimental"]),
        ("Led Zeppelin", ["rock", "hard rock", "blues rock"]),
        ("Sade", ["soul", "R&B", "smooth jazz"]),
        ("Aphex Twin", ["electronic", "ambient", "IDM"]),
        ("Nina Simone", ["jazz", "soul", "blues"]),
    ]
    metadata = {}
    for i, aid in enumerate(artist_ids):
        name, tags = artists[i % len(artists)]
        metadata[aid] = {"title": f"{name} #{aid}", "tags": tags}
    return metadata


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train recommendation models")
    parser.add_argument("--domain", choices=["movies", "music", "all"], default="all")
    args = parser.parse_args()

    if args.domain in ("movies", "all"):
        train_movies()
    if args.domain in ("music", "all"):
        train_music()

    logger.info("🎉 Training complete!")