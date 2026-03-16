from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager

from app.routers import movies, music, health
from app.ml.model_manager import ModelManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models on startup."""
    ModelManager.load_all()
    yield
    ModelManager.clear_all()


app = FastAPI(
    title="RecommenderAPI",
    description="""
## 🎬🎵 RecommenderAPI

A production-grade recommendation engine for **movies** and **music** using collaborative filtering and matrix factorization.

### Features
- **Movie recommendations** based on MovieLens dataset
- **Music recommendations** based on Last.fm dataset
- **User-based** and **item-based** collaborative filtering
- **Hybrid** recommendations combining multiple strategies
- Full **Swagger UI** documentation
- Health check & model metadata endpoints

### How it works
The API uses **Singular Value Decomposition (SVD)** via the `Surprise` library for collaborative filtering,
combined with **cosine similarity** for item-based recommendations.
    """,
    version="1.0.0",
    contact={
        "name": "RecommenderAPI",
        "url": "https://github.com/yourusername/recommender-api",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(movies.router, prefix="/movies", tags=["Movies"])
app.include_router(music.router, prefix="/music", tags=["Music"])


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
