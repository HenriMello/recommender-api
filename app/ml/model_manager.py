import pickle
import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))


class ModelManager:
    """Central registry for loaded ML models."""

    _registry: dict = {}

    @classmethod
    def load_all(cls):
        """Load all available models from disk."""
        logger.info("Loading ML models...")
        for domain in ["movies", "music"]:
            path = MODELS_DIR / f"{domain}_model.pkl"
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        cls._registry[domain] = pickle.load(f)
                    logger.info(f"✅ Loaded model: {domain}")
                except Exception as e:
                    logger.error(f"❌ Failed to load {domain} model: {e}")
            else:
                logger.warning(f"⚠️  Model not found: {path} — run scripts/train.py first")

    @classmethod
    def get(cls, domain: str) -> Optional[object]:
        return cls._registry.get(domain)

    @classmethod
    def clear_all(cls):
        cls._registry.clear()

    @classmethod
    def is_loaded(cls, domain: str) -> bool:
        return domain in cls._registry
