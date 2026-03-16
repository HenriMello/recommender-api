# 🎬🎵 RecommenderAPI

> A production-grade recommendation engine for movies and music, powered by collaborative filtering and exposed via a fully documented REST API.

[![CI/CD](https://github.com/yourusername/recommender-api/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/recommender-api/actions)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📌 About

**RecommenderAPI** is a production-ready recommendation engine that combines **collaborative filtering** (SVD matrix factorization) with **cosine similarity** to generate personalized recommendations for movies and music artists.

Built to demonstrate:

- Applied Machine Learning (Surprise / SVD / scikit-learn)
- Software engineering with FastAPI, Pydantic, and REST best practices
- Automated testing with pytest (>90% coverage)
- Multi-stage Docker containerization
- CI/CD pipeline with GitHub Actions
- Cloud deployment on Render

---

## 🏗️ Architecture

```
recommender-api/
├── app/
│   ├── main.py                 # FastAPI entrypoint + lifespan
│   ├── routers/
│   │   ├── movies.py           # Movie endpoints
│   │   ├── music.py            # Music endpoints
│   │   └── health.py           # Health check + model metadata
│   ├── ml/
│   │   ├── recommender.py      # Engine: user_based, item_based, hybrid
│   │   └── model_manager.py    # Model loading & in-memory registry
│   └── schemas/
│       └── schemas.py          # Pydantic models (request/response)
├── scripts/
│   └── train.py                # ML model training script
├── tests/
│   └── test_api.py             # Full test suite (pytest)
├── .github/workflows/
│   └── ci.yml                  # CI/CD pipeline
├── Dockerfile                  # Multi-stage build
├── docker-compose.yml          # Local orchestration
└── requirements.txt
```

### How the algorithm works

```
Raw interaction data (ratings)
          │
          ▼
┌──────────────────────┐
│   Surprise SVD        │  ← Matrix Factorization (n_factors=100)
│   (training phase)    │     minimizes RMSE via Stochastic Gradient Descent
└──────────────────────┘
          │
          ├──► user_based   → predicts ratings for unseen items per user
          │
          ├──► item_based   → cosine similarity between item latent vectors (qi)
          │
          └──► hybrid       → weighted blend (α=0.6 user + 0.4 popularity)
```

---

## 🚀 Quickstart

### Prerequisites

- Python 3.11+
- Docker (optional but recommended)

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/recommender-api.git
cd recommender-api
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Train the models

```bash
# Train both movies (MovieLens 100K — auto-downloaded) and music
python scripts/train.py --domain all

# Or individually:
python scripts/train.py --domain movies
python scripts/train.py --domain music
```

> The MovieLens 100K dataset (~5MB) is downloaded automatically by the `Surprise` library.  
> The music model uses synthetic data by default. For real Last.fm data, see the [Datasets](#-datasets) section.

### 4. Start the API

```bash
uvicorn app.main:app --reload
```

Open in your browser:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health check:** http://localhost:8000/health

---

## 🐳 Docker

### Run with Docker Compose

```bash
# Train models first (one-time)
docker compose --profile train up trainer

# Start the API
docker compose up api
```

### Manual Docker build

```bash
docker build -t recommender-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models recommender-api
```

---

## 📡 API Endpoints

### Health

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | API status and loaded models |
| GET | `/models/{domain}` | Model metadata (movies or music) |

### Movies

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/movies/` | List movies (paginated) |
| GET | `/movies/{id}` | Get movie by ID |
| POST | `/movies/recommend` | Get personalized recommendations |

### Music

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/music/` | List artists (paginated) |
| GET | `/music/{id}` | Get artist by ID |
| POST | `/music/recommend` | Get personalized recommendations |

### Example request

```bash
curl -X POST http://localhost:8000/movies/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 42,
    "top_n": 5,
    "strategy": "hybrid"
  }'
```

```json
{
  "user_id": 42,
  "strategy": "hybrid",
  "total": 5,
  "recommendations": [
    { "id": 318, "title": "Shawshank Redemption, The (1994)", "score": 0.97, "genres": ["Drama"] },
    { "id": 858, "title": "Godfather, The (1972)", "score": 0.95, "genres": ["Crime", "Drama"] },
    { "id": 50,  "title": "Usual Suspects, The (1995)", "score": 0.93, "genres": ["Crime", "Thriller"] }
  ]
}
```

### Recommendation strategies

| Strategy | Description | Required fields |
|----------|-------------|-----------------|
| `user_based` | SVD predictions for unseen items | `user_id` |
| `item_based` | Cosine similarity between item vectors | `movie_id` or `artist_id` |
| `hybrid` | Weighted blend of user-based + popularity | `user_id` |

---

## 🧪 Tests

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=app --cov-report=term-missing

# Run specific test classes
pytest tests/test_api.py::TestMovies -v
pytest tests/test_api.py::TestRecommenderEngine -v
```

The test suite covers:
- All endpoints (200, 404, 422, 503 status codes)
- Pydantic input validation
- Fallback behavior when models are not loaded (503)
- Recommendation engine unit tests
- Score normalization (0.0 to 1.0 range)

---

## ⚙️ CI/CD (GitHub Actions)

The pipeline in `.github/workflows/ci.yml` runs automatically on every push:

```
push → main/develop
    │
    ├── 1. Lint (Ruff — style + format check)
    ├── 2. Tests + coverage (pytest-cov → Codecov)
    ├── 3. Docker build (validation)
    └── 4. Deploy → Render (main branch only)
```

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `RENDER_API_KEY` | Your Render API key |
| `RENDER_SERVICE_ID` | Service ID from Render dashboard |
| `RENDER_APP_URL` | Public URL of your deployed app |
| `CODECOV_TOKEN` | Codecov token (optional) |

---

## ☁️ Deploying to Render

1. Create a free account at [render.com](https://render.com)
2. New Service → **Web Service** → connect your GitHub repository
3. Configure the service:
   - **Runtime:** Docker
   - **Dockerfile Path:** `./Dockerfile`
   - **Health Check Path:** `/health`
   - **Environment Variable:** `MODELS_DIR=/app/models`
4. Under **Disks**, add a persistent disk mounted at `/app/models` to persist trained models across deploys
5. Open the Render **Shell** and run: `python scripts/train.py --domain all`
6. Add the Render secrets to your GitHub repo to enable automatic deploys on push to `main`

---

## 📦 Datasets

| Domain | Dataset | Size | Download |
|--------|---------|------|----------|
| Movies | MovieLens 100K | ~5MB | Automatic (via Surprise) |
| Music | Synthetic (default) | — | Included |
| Music | Last.fm 360K | ~500MB | [link](http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html) |

To use real Last.fm data, save the file as `data/lastfm_ratings.csv` with columns `user_id, artist_id, plays` before running training.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| API framework | FastAPI + Uvicorn |
| ML | Surprise (SVD), scikit-learn (cosine similarity) |
| Validation | Pydantic v2 |
| Testing | pytest + httpx |
| Containerization | Docker multi-stage build |
| CI/CD | GitHub Actions |
| Cloud deployment | Render |
| Linter | Ruff |

---

## 📄 License

MIT © 2024 — see [LICENSE](LICENSE)
