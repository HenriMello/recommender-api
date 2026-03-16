#  RecommenderAPI

> Motor de recomendação de filmes e músicas com ML aplicado, exposto via API REST documentada com Swagger.

[![CI/CD](https://github.com/yourusername/recommender-api/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/recommender-api/actions)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Sobre o Projeto

O **RecommenderAPI** é um motor de recomendação production-ready que combina **filtragem colaborativa** (SVD) com **similaridade por cosseno** para gerar recomendações personalizadas de filmes e artistas musicais.

Desenvolvido como projeto de portfólio para demonstrar:

- Machine Learning aplicado (Surprise / SVD / scikit-learn)
- Engenharia de software com FastAPI, Pydantic e boas práticas REST
- Testes automatizados com pytest (cobertura > 90%)
- Containerização com Docker multi-stage
- Pipeline CI/CD com GitHub Actions
- Deploy na nuvem (Render)

---

## Arquitetura

```
recommender-api/
├── app/
│   ├── main.py                 # Entrypoint FastAPI + lifespan
│   ├── routers/
│   │   ├── movies.py           # Endpoints de filmes
│   │   ├── music.py            # Endpoints de músicas
│   │   └── health.py           # Health check + model info
│   ├── ml/
│   │   ├── recommender.py      # Engine: user_based, item_based, hybrid
│   │   └── model_manager.py    # Carregamento e cache de modelos
│   └── schemas/
│       └── schemas.py          # Modelos Pydantic (request/response)
├── scripts/
│   └── train.py                # Treinamento dos modelos ML
├── tests/
│   └── test_api.py             # Suite de testes (pytest)
├── .github/workflows/
│   └── ci.yml                  # Pipeline CI/CD
├── Dockerfile                  # Build multi-stage
├── docker-compose.yml          # Orquestração local
└── requirements.txt
```

### Como funciona o algoritmo

```
Dados brutos (ratings)
       │
       ▼
┌─────────────────────┐
│  Surprise SVD        │  ← Fatoração de Matriz (n_factors=100)
│  (treinamento)       │     minimiza RMSE via SGD
└─────────────────────┘
       │
       ├──► user_based   → prediz ratings para itens não vistos
       │
       ├──► item_based   → similaridade cosseno entre vetores de itens (qi)
       │
       └──► hybrid       → blend ponderado (α=0.6 user + 0.4 popularidade)
```

---

## Quickstart

### Pré-requisitos

- Python 3.11+
- Docker (opcional, mas recomendado)

### 1. Clonar o repositório

```bash
git clone https://github.com/yourusername/recommender-api.git
cd recommender-api
```

### 2. Instalar dependências

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Treinar os modelos

```bash
# Treina filmes (MovieLens 100K — baixado automaticamente) e músicas
python scripts/train.py --domain all

# Ou individualmente:
python scripts/train.py --domain movies
python scripts/train.py --domain music
```

> O dataset MovieLens 100K (~5MB) é baixado automaticamente pela biblioteca `Surprise`.  
> O modelo de música usa dados sintéticos por padrão. Para usar Last.fm real, veja a seção [Datasets](#-datasets).

### 4. Iniciar a API

```bash
uvicorn app.main:app --reload
```

Acesse:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health check:** http://localhost:8000/health

---

## 🐳 Docker

### Build e execução com Docker Compose

```bash
# Treinar modelos (necessário uma vez)
docker compose --profile train up trainer

# Subir a API
docker compose up api
```

### Build manual

```bash
docker build -t recommender-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models recommender-api
```

---

## 📡 Endpoints da API

### Health

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/health` | Status da API e modelos carregados |
| GET | `/models/{domain}` | Metadados do modelo (movies ou music) |

### Filmes

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/movies/` | Lista filmes (paginado) |
| GET | `/movies/{id}` | Busca filme por ID |
| POST | `/movies/recommend` | Recomendações personalizadas |

### Músicas

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/music/` | Lista artistas (paginado) |
| GET | `/music/{id}` | Busca artista por ID |
| POST | `/music/recommend` | Recomendações personalizadas |

### Exemplo de requisição

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

---

## 🧪 Testes

```bash
# Rodar todos os testes
pytest

# Com cobertura de código
pytest --cov=app --cov-report=term-missing

# Testes específicos
pytest tests/test_api.py::TestMovies -v
pytest tests/test_api.py::TestRecommenderEngine -v
```

A suite cobre:
- Todos os endpoints (200, 404, 422, 503)
- Validação de inputs com Pydantic
- Comportamento de fallback (modelo não carregado)
- Engine de recomendação (unitário)
- Normalização dos scores (0.0 a 1.0)

---

## ⚙️ CI/CD (GitHub Actions)

O pipeline `.github/workflows/ci.yml` executa automaticamente a cada push:

```
push → main/develop
    │
    ├── 1. Lint (Ruff)
    ├── 2. Testes + cobertura (pytest-cov → Codecov)
    ├── 3. Build Docker (validação)
    └── 4. Deploy → Render (apenas branch main)
```

### Secrets necessários no GitHub

| Secret | Descrição |
|--------|-----------|
| `RENDER_API_KEY` | API Key do Render |
| `RENDER_SERVICE_ID` | ID do serviço no Render |
| `RENDER_APP_URL` | URL pública do app |
| `CODECOV_TOKEN` | Token do Codecov (opcional) |

---

## ☁️ Deploy no Render

1. Crie uma conta em [render.com](https://render.com)
2. Novo serviço → **Web Service** → conecte seu repositório GitHub
3. Configure:
   - **Runtime:** Docker
   - **Dockerfile Path:** `./Dockerfile`
   - **Health Check Path:** `/health`
   - **Environment Variable:** `MODELS_DIR=/app/models`
4. Em **Disks**, adicione um disco persistente em `/app/models` para os modelos treinados
5. Rode o treinamento via **Shell** do Render: `python scripts/train.py --domain all`
6. Adicione os secrets do Render no repositório GitHub para ativar o deploy automático

---

## 📦 Datasets

| Domínio | Dataset | Tamanho | Download |
|---------|---------|---------|----------|
| Filmes | MovieLens 100K | ~5MB | Automático (Surprise) |
| Músicas | Sintético (padrão) | — | Incluso |
| Músicas | Last.fm 360K | ~500MB | [link](http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html) |

Para usar o Last.fm real, salve o arquivo como `data/lastfm_ratings.csv` com colunas `user_id, artist_id, plays` antes de treinar.

---

## 🛠️ Stack Tecnológica

| Camada | Tecnologia |
|--------|------------|
| API | FastAPI + Uvicorn |
| ML | Surprise (SVD), scikit-learn (cosine similarity) |
| Validação | Pydantic v2 |
| Testes | pytest + httpx |
| Container | Docker multi-stage |
| CI/CD | GitHub Actions |
| Deploy | Render |
| Lint | Ruff |

---

## 📄 Licença

MIT © 2024 — veja [LICENSE](LICENSE)
