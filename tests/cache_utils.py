"""Filesystem cache for API responses in tests."""

import hashlib
import json
from pathlib import Path
import numpy as np


CACHE_DIR = Path(__file__).parent / ".cache"


def _get_cache_key(model_name: str, texts: list[str]) -> str:
    """Generate cache key from request parameters."""
    # Create a deterministic hash from model and sorted texts
    content = json.dumps({"model": model_name, "texts": sorted(texts)}, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def get_cached_response(model_name: str, texts: list[str]) -> np.ndarray | None:
    """Get cached API response if it exists."""
    cache_key = _get_cache_key(model_name, texts)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if not cache_file.exists():
        return None

    try:
        with open(cache_file) as f:
            data = json.load(f)
            # Convert list back to numpy array
            return np.array(data["embeddings"], dtype=np.float32)
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def save_cached_response(model_name: str, texts: list[str], embeddings: np.ndarray) -> None:
    """Save API response to cache."""
    CACHE_DIR.mkdir(exist_ok=True)

    cache_key = _get_cache_key(model_name, texts)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    data = {
        "model": model_name,
        "texts": texts,
        "embeddings": embeddings.tolist(),
    }

    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)


def cached_encode(original_encode):
    """Wrapper that adds caching to any backend's encode method."""
    def wrapper(self, texts: list[str]) -> np.ndarray:
        # Try to get from cache
        cached = get_cached_response(self.model_name, texts)
        if cached is not None:
            return cached

        # Call original method
        embeddings = original_encode(self, texts)

        # Save to cache
        save_cached_response(self.model_name, texts, embeddings)

        return embeddings

    return wrapper
