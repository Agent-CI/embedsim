from typing import TypedDict
import os
import numpy as np

from .backends import SentenceTransformerBackend, OpenAIBackend, EmbeddingBackend

DEFAULT_MODEL = os.getenv("EMBEDSIM_MODEL", "openai-3-small")

_backend_cache: dict[tuple, EmbeddingBackend] = {}


class ModelConfig(TypedDict):
    """Configuration for an embedding model."""

    backend_class: type[EmbeddingBackend]
    model_name: str
    max_seq_length: int


# Model configurations mapping model names to backend classes and settings
MODEL_CONFIGS: dict[str, ModelConfig] = {
    "all-MiniLM-L6-v2": {
        "backend_class": SentenceTransformerBackend,
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "max_seq_length": 256,
    },
    "jina-v2-base": {
        "backend_class": SentenceTransformerBackend,
        "model_name": "jinaai/jina-embeddings-v2-base-en",
        "max_seq_length": 8192,
    },
    "bge-large": {
        "backend_class": SentenceTransformerBackend,
        "model_name": "BAAI/bge-large-en-v1.5",
        "max_seq_length": 1024,
    },
    "e5-large": {
        "backend_class": SentenceTransformerBackend,
        "model_name": "intfloat/e5-large-v2",
        "max_seq_length": 1024,
    },
    "openai-3-small": {
        "backend_class": OpenAIBackend,
        "model_name": "text-embedding-3-small",
        "max_seq_length": 8191,
    },
    "openai-3-large": {
        "backend_class": OpenAIBackend,
        "model_name": "text-embedding-3-large",
        "max_seq_length": 8191,
    },
}


def _get_backend(model_id: str, **config) -> EmbeddingBackend:
    """Get or create a backend for the given model ID."""
    global _backend_cache, MODEL_CONFIGS

    if model_id not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model ID: {model_id}. Available: {list(MODEL_CONFIGS.keys())}"
        )

    # Create cache key from model_id and sorted config
    cache_key = (model_id, tuple(sorted(config.items())))

    if cache_key not in _backend_cache:
        _config = MODEL_CONFIGS[model_id].copy()
        _config.update(config)

        # Explicitly instantiate the backend
        _backend_cache[cache_key] = _config["backend_class"](
            model_name=_config["model_name"],
            max_seq_length=_config["max_seq_length"],
        )

    return _backend_cache[cache_key]


def pairsim(
    text_a: str,
    text_b: str,
    model_id: str = DEFAULT_MODEL,
    **config,
) -> float:
    """Compute similarity between two texts.

    Args:
        text_a: First text
        text_b: Second text
        model_id: Embedding model to use (defaults to EMBEDSIM_MODEL env var or "openai-3-small")
        **config: Optional configuration overrides

    Returns:
        Similarity score (0-1, higher = more similar)

    Examples:
        >>> score = pairsim("hello world", "hello there")
        >>> print(score)  # 0.98
    """
    # Get backend and generate embeddings
    backend = _get_backend(model_id, **config)
    embeddings = backend.encode([text_a, text_b])

    # Normalize and compute cosine similarity
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    return float(normalized[0] @ normalized[1])


def groupsim(
    texts: list[str],
    model_id: str = DEFAULT_MODEL,
    **config,
) -> list[float]:
    """Compute similarity of texts using centroid-based scoring.

    Computes the centroid of all text embeddings and measures how closely
    each text aligns with that centroid. Works with 2+ texts.

    Args:
        texts: List of texts to analyze (minimum 2)
        model_id: Embedding model to use (defaults to EMBEDSIM_MODEL env var or "openai-3-small")
        **config: Optional configuration overrides

    Returns:
        List of similarity scores (floats), one per input text

    Examples:
        >>> # Analyze group coherence
        >>> scores = groupsim(["hello", "hi", "goodbye"])
        >>> print(scores)  # [0.92, 0.88, 0.45]
    """
    # Get backend and generate embeddings
    backend = _get_backend(model_id, **config)
    embeddings = backend.encode(texts)

    # Normalize embeddings
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Compute and normalize centroid
    centroid = normalized.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    # Compute similarities
    similarities = normalized @ centroid

    return similarities.tolist()
