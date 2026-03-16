from fastapi import APIRouter, HTTPException, Query
from app.schemas.schemas import (
    MovieRecommendRequest, RecommendationResponse, RecommendationItem, MovieItem
)
from app.ml.model_manager import ModelManager
from app.ml.recommender import RecommenderEngine

router = APIRouter()


def _get_engine() -> RecommenderEngine:
    model = ModelManager.get("movies")
    if not model:
        raise HTTPException(
            status_code=503,
            detail="Movies model not loaded. Run `python scripts/train.py --domain movies` first."
        )
    return RecommenderEngine(
        svd_model=model["svd"],
        interaction_matrix=model["interaction_matrix"],
        item_ids=model["item_ids"],
        user_ids=model["user_ids"],
        item_metadata=model["item_metadata"],
    )


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    summary="Get movie recommendations",
    description="""
Returns personalized movie recommendations using collaborative filtering.

**Strategies:**
- `user_based`: SVD predictions for a specific user
- `item_based`: movies similar to a given movie
- `hybrid`: blends user-based and popularity signals (recommended)

At least one of `user_id` or `movie_id` must be provided.
    """,
)
async def recommend_movies(body: MovieRecommendRequest):
    if not body.user_id and not body.movie_id:
        raise HTTPException(status_code=422, detail="Provide at least user_id or movie_id.")

    engine = _get_engine()
    strategy = body.strategy

    if strategy == "item_based":
        if not body.movie_id:
            raise HTTPException(status_code=422, detail="item_based strategy requires movie_id.")
        recs = engine.recommend_similar_items(body.movie_id, body.top_n)
    elif strategy == "user_based":
        if not body.user_id:
            raise HTTPException(status_code=422, detail="user_based strategy requires user_id.")
        recs = engine.recommend_for_user(body.user_id, body.top_n)
    else:  # hybrid
        user_id = body.user_id or 1
        recs = engine.recommend_hybrid(user_id, body.top_n)

    return RecommendationResponse(
        user_id=body.user_id,
        item_id=body.movie_id,
        strategy=strategy,
        total=len(recs),
        recommendations=[RecommendationItem(**r) for r in recs],
    )


@router.get(
    "/",
    response_model=list[MovieItem],
    summary="List movies",
    description="Returns a paginated list of movies available in the dataset.",
)
async def list_movies(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(20, ge=1, le=100, description="Number of items to return"),
):
    model = ModelManager.get("movies")
    if not model:
        raise HTTPException(status_code=503, detail="Movies model not loaded.")

    metadata = model["item_metadata"]
    items = []
    for mid, info in list(metadata.items())[skip: skip + limit]:
        items.append(MovieItem(
            movie_id=mid,
            title=info.get("title", str(mid)),
            genres=info.get("genres", []),
        ))
    return items


@router.get(
    "/{movie_id}",
    response_model=MovieItem,
    summary="Get movie by ID",
)
async def get_movie(movie_id: int):
    model = ModelManager.get("movies")
    if not model:
        raise HTTPException(status_code=503, detail="Movies model not loaded.")

    info = model["item_metadata"].get(movie_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found.")

    return MovieItem(movie_id=movie_id, title=info["title"], genres=info.get("genres", []))
