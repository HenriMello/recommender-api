from pydantic import BaseModel, Field

# ─── Shared ────────────────────────────────────────────────────────────────────


class RecommendationItem(BaseModel):
    id: int = Field(..., description="Unique identifier of the item")
    title: str = Field(..., description="Title of the item")
    score: float = Field(
        ..., description="Recommendation score (higher = more relevant)", ge=0.0, le=1.0
    )
    genres: list[str] | None = Field(default=None, description="Associated genres")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": 1,
                "title": "Toy Story (1995)",
                "score": 0.94,
                "genres": ["Animation", "Children", "Comedy"],
            }
        }
    }


class RecommendationResponse(BaseModel):
    user_id: int | None = Field(
        default=None, description="User ID used for personalized recommendations"
    )
    item_id: int | None = Field(
        default=None, description="Item ID used for item-based recommendations"
    )
    strategy: str = Field(
        ..., description="Algorithm used: 'user_based', 'item_based', or 'hybrid'"
    )
    total: int = Field(..., description="Total number of recommendations returned")
    recommendations: list[RecommendationItem]


# ─── Movies ────────────────────────────────────────────────────────────────────


class MovieRecommendRequest(BaseModel):
    user_id: int | None = Field(
        default=None, description="User ID for personalized recommendations", ge=1
    )
    movie_id: int | None = Field(
        default=None, description="Movie ID for similar movie recommendations", ge=1
    )
    top_n: int = Field(default=10, description="Number of recommendations to return", ge=1, le=50)
    strategy: str = Field(
        default="hybrid", description="Strategy: 'user_based', 'item_based', or 'hybrid'"
    )

    model_config = {
        "json_schema_extra": {"example": {"user_id": 42, "top_n": 10, "strategy": "hybrid"}}
    }


class MovieItem(BaseModel):
    movie_id: int
    title: str
    genres: list[str]
    avg_rating: float | None = None
    num_ratings: int | None = None


# ─── Music ─────────────────────────────────────────────────────────────────────


class MusicRecommendRequest(BaseModel):
    user_id: int | None = Field(
        default=None, description="User ID for personalized recommendations", ge=1
    )
    artist_id: int | None = Field(
        default=None, description="Artist ID for similar artist recommendations", ge=1
    )
    top_n: int = Field(default=10, description="Number of recommendations to return", ge=1, le=50)
    strategy: str = Field(
        default="hybrid", description="Strategy: 'user_based', 'item_based', or 'hybrid'"
    )

    model_config = {
        "json_schema_extra": {"example": {"user_id": 7, "top_n": 10, "strategy": "user_based"}}
    }


class ArtistItem(BaseModel):
    artist_id: int
    name: str
    tags: list[str] | None = None
    listen_count: int | None = None


# ─── Health ────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: dict[str, bool]


class ModelInfoResponse(BaseModel):
    domain: str
    algorithm: str
    num_users: int
    num_items: int
    dataset: str
    trained_at: str
