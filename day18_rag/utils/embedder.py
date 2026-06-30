"""
embedder.py
-----------
Wraps sentence-transformers to produce embeddings.
Used across all notebooks so the embedding logic is in one place.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# Load once at import time. Model is downloaded on first run (~90MB).
# all-MiniLM-L6-v2 is fast, small, and good enough for learning.
MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(text: str) -> np.ndarray:
    """
    Embed a single string. Returns a 1D numpy array of shape (384,).
    """
    model = get_model()
    return model.encode(text, normalize_embeddings=True)


def embed_batch(texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings. Returns a 2D numpy array of shape (N, 384).
    Batching is faster than calling embed() in a loop.
    """
    model = get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two 1D vectors.
    Assumes vectors are already normalized (embed() does this by default).
    For normalized vectors: cosine_similarity = dot product.
    """
    return float(np.dot(a, b))
