from fastapi import APIRouter

from app.ml.model_manager import ModelManager
from app.schemas.schemas import HealthResponse, ModelInfoResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns API status and which ML models are currently loaded.",
)
async def health():
    return HealthResponse(
        status="ok",
        version="1.0.0",
        models_loaded={
            "movies": ModelManager.is_loaded("movies"),
            "music": ModelManager.is_loaded("music"),
        },
    )


@router.get(
    "/models/{domain}",
    response_model=ModelInfoResponse,
    summary="Model metadata",
    description="Returns metadata for a loaded model (movies or music).",
)
async def model_info(domain: str):
    from fastapi import HTTPException

    model = ModelManager.get(domain)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{domain}' not loaded.")

    return ModelInfoResponse(
        domain=model["domain"],
        algorithm=model["algorithm"],
        num_users=model["num_users"],
        num_items=model["num_items"],
        dataset=model["dataset"],
        trained_at=model["trained_at"],
    )
