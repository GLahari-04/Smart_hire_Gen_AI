"""Module — Local Embeddings Wrapper (SentenceTransformers).

Provides helper functions to generate text embeddings using local SentenceTransformer models (all-MiniLM-L6-v2).
Used by semantic job search and the AI Career Mentor RAG pipeline without calling external APIs.
"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from src import config

_MODEL_CACHE = {}


def _get_local_model(model_name: Optional[str] = None) -> SentenceTransformer:
    """Retrieve or lazily initialize the local SentenceTransformer model."""
    target_model = model_name or config.EMBED_MODEL
    if target_model not in _MODEL_CACHE:
        _MODEL_CACHE[target_model] = SentenceTransformer(target_model)
    return _MODEL_CACHE[target_model]


def get_embedding(text: str, model_name: Optional[str] = None) -> List[float]:
    """Generate an embedding vector for a single string of text using local SentenceTransformer.

    Args:
        text (str): Input text string to embed.
        model_name (Optional[str]): Embedding model name (defaults to config.EMBED_MODEL).

    Returns:
        List[float]: Embedding vector containing float values (384 dimensions for all-MiniLM-L6-v2).
    """
    if not text or not text.strip():
        raise ValueError("Cannot compute embedding for empty or whitespace-only text.")

    model = _get_local_model(model_name)
    vector = model.encode(text.strip(), convert_to_numpy=True)
    return vector.astype(float).tolist()


def get_embeddings_batch(
    texts: List[str],
    model_name: Optional[str] = None,
    batch_size: int = 32
) -> List[List[float]]:
    """Generate embedding vectors for a list of text strings using local SentenceTransformer.

    Args:
        texts (List[str]): List of text strings to convert into embeddings.
        model_name (Optional[str]): Embedding model name (defaults to config.EMBED_MODEL).
        batch_size (int): Batch size for batch encoding (default: 32).

    Returns:
        List[List[float]]: List of float embedding vectors corresponding to each input string.
    """
    if not texts:
        return []

    clean_texts = [t.strip() for t in texts if t and t.strip()]
    if not clean_texts:
        return []

    model = _get_local_model(model_name)
    vectors = model.encode(clean_texts, batch_size=batch_size, convert_to_numpy=True)
    return [v.astype(float).tolist() for v in vectors]
