from fastapi import APIRouter, HTTPException, Query
from app.schemas.schemas import (
    MusicRecommendRequest, RecommendationResponse, RecommendationItem, ArtistItem
)
from app.ml.model_manager import ModelManager
from app.ml.recommender import RecommenderEngine

router = APIRouter()


def _get_engine() -> RecommenderEngine:
    model = ModelManager.get("music")
    if not model:
        raise HTTPException(
            status_code=503,
            detail="Music model not loaded. Run `python scripts/train.py --domain music` first."
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
    summary="Get music recommendations",
    description="""
Returns personalized artist recommendations using collaborative filtering.

**Strategies:**
- `user_based`: SVD predictions based on a user's listening history
- `item_based`: artists similar to a given artist
- `hybrid`: blends user-based and popularity signals (recommended)

At least one of `user_id` or `artist_id` must be provided.
    """,
)
async def recommend_music(body: MusicRecommendRequest):
    if not body.user_id and not body.artist_id:
        raise HTTPException(status_code=422, detail="Provide at least user_id or artist_id.")

    engine = _get_engine()
    strategy = body.strategy

    if strategy == "item_based":
        if not body.artist_id:
            raise HTTPException(status_code=422, detail="item_based strategy requires artist_id.")
        recs = engine.recommend_similar_items(body.artist_id, body.top_n)
    elif strategy == "user_based":
        if not body.user_id:
            raise HTTPException(status_code=422, detail="user_based strategy requires user_id.")
        recs = engine.recommend_for_user(body.user_id, body.top_n)
    else:
        user_id = body.user_id or 1
        recs = engine.recommend_hybrid(user_id, body.top_n)

    return RecommendationResponse(
        user_id=body.user_id,
        item_id=body.artist_id,
        strategy=strategy,
        total=len(recs),
        recommendations=[RecommendationItem(**r) for r in recs],
    )


@router.get(
    "/",
    response_model=list[ArtistItem],
    summary="List artists",
    description="Returns a paginated list of artists available in the dataset.",
)
async def list_artists(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    model = ModelManager.get("music")
    if not model:
        raise HTTPException(status_code=503, detail="Music model not loaded.")

    metadata = model["item_metadata"]
    items = []
    for aid, info in list(metadata.items())[skip: skip + limit]:
        items.append(ArtistItem(
            artist_id=aid,
            name=info.get("title", str(aid)),
            tags=info.get("tags"),
        ))
    return items


@router.get(
    "/{artist_id}",
    response_model=ArtistItem,
    summary="Get artist by ID",
)
async def get_artist(artist_id: int):
    model = ModelManager.get("music")
    if not model:
        raise HTTPException(status_code=503, detail="Music model not loaded.")

    info = model["item_metadata"].get(artist_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Artist {artist_id} not found.")

    return ArtistItem(artist_id=artist_id, name=info["title"], tags=info.get("tags"))
