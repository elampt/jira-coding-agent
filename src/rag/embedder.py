"""
The embedding model, loaded once per process and shared.

Loading all-MiniLM-L6-v2 from disk takes ~5s, while embedding a query takes ~0.01s. Both the
indexer (once per ticket) and the retriever (once per search) used to reload it every time;
sharing one instance removes that cost from every ticket after the first.
"""

import logging
import threading

from sentence_transformers import SentenceTransformer

from src.config import config

logger = logging.getLogger(__name__)

_lock = threading.Lock()  # tickets run in background threads; avoid two simultaneous loads
_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Return the shared embedding model, loading it on first use."""
    global _model
    with _lock:
        if _model is None:
            logger.info(f"Loading embedding model: {config.embeddings.model}")
            _model = SentenceTransformer(config.embeddings.model)
        return _model
