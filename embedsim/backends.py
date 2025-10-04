from abc import ABC, abstractmethod
import os
import numpy as np


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""

    model_name: str
    max_seq_length: int

    @abstractmethod
    def __init__(self, model_name: str, max_seq_length: int):
        """Initialize the backend.

        Args:
            model_name: Name/identifier of the model
            max_seq_length: Maximum sequence length for the model
        """
        pass

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode text(s) into embeddings."""
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        pass


class SentenceTransformerBackend(EmbeddingBackend):
    """Backend using sentence-transformers models."""

    def __init__(self, model_name: str, max_seq_length: int):
        """Initialize with a sentence transformer model.

        Args:
            model_name: Name of the sentence transformer model.
            max_seq_length: Maximum sequence length to set.
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self._model = None

    @property
    def model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._model.max_seq_length = self.max_seq_length
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode text(s) into embeddings."""
        embeddings = self.model.encode(texts)
        return embeddings.astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()


class OpenAIBackend(EmbeddingBackend):
    """Backend using OpenAI's embedding API."""

    def __init__(self, model_name: str, max_seq_length: int):
        """Initialize with OpenAI API settings.

        Args:
            model_name: OpenAI model name (e.g., "text-embedding-3-small").
            max_seq_length: Maximum sequence length for the model.
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI backend requires 'openai' package. "
                "Install with: pip install openai"
            )

        if not os.getenv("OPENAI_API_KEY") and not os.getenv("EMBEDSIM_OPENAI_API_KEY"):
            raise ValueError(
                "OpenAI API key not found. Set either OPENAI_API_KEY or "
                "EMBEDSIM_OPENAI_API_KEY environment variable."
            )

        self.model_name = model_name
        self.max_seq_length = max_seq_length

        # Use EMBEDSIM_OPENAI_API_KEY if available, otherwise fall back to OPENAI_API_KEY
        api_key = os.getenv("EMBEDSIM_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=api_key)

        # Model-specific configurations
        self._model_configs = {
            "text-embedding-3-small": {"dim": 1536, "max_tokens": 8191},
            "text-embedding-3-large": {"dim": 3072, "max_tokens": 8191},
            "text-embedding-ada-002": {"dim": 1536, "max_tokens": 8191},
        }

        if model_name not in self._model_configs:
            raise ValueError(
                f"Unknown OpenAI model: {model_name}. Available models: {list(self._model_configs.keys())}"
            )

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode text(s) into embeddings using OpenAI API."""
        try:
            response = self.client.embeddings.create(input=texts, model=self.model_name)

            embeddings = np.array([item.embedding for item in response.data])
            return embeddings.astype(np.float32)

        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        if self.model_name in self._model_configs:
            return self._model_configs[self.model_name]["dim"]
        return 1536  # Default for most OpenAI models
