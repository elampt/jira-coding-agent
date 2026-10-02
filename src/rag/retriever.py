"""
Retriever — queries FAISS to find files semantically similar to the ticket.

How it works:
  1. Embed the query text (ticket summary) into a vector
  2. Ask FAISS: "which stored vectors are closest to this?"
  3. FAISS returns indices → we look up metadata to get file path + content

Returns the same format as grep results: list of {path, content}
so the SEARCH node can combine both seamlessly.
"""

import json
import logging

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import config

logger = logging.getLogger(__name__)

INDEX_FILE = config.vector_store.index_file
METADATA_FILE = config.vector_store.metadata_file
MODEL_NAME = config.embeddings.model


def retrieve_similar(query: str, top_k: int = 5) -> list[dict]:
    """Find the top_k most similar files to the query text.

    Args:
        query: text to search for (e.g. "navigation bar color")
        top_k: how many results to return (default 5)

    Returns list of:
        {"path": "src/App.js", "content": "...full file..."}
    """
    # Check if index exists
    if not INDEX_FILE.exists() or not METADATA_FILE.exists():
        logger.warning("FAISS index not found. Run indexer first: python -m src.rag.indexer")
        return []

    # Load FAISS index and metadata
    index = faiss.read_index(str(INDEX_FILE))
    with open(METADATA_FILE) as f:
        metadata = json.load(f)

    # Embed the query — same model used during indexing
    model = SentenceTransformer(MODEL_NAME)
    query_vector = model.encode([query])
    # Shape: (1, 384) — one query, 384 dimensions

    # Search FAISS — returns distances and indices of closest vectors
    distances, indices = index.search(
        np.array(query_vector).astype("float32"),
        min(top_k, index.ntotal),  # can't return more results than we have
    )
    # distances[0] = [0.45, 0.67, 0.89, ...]  — how far each match is
    # indices[0] = [2, 0, 4, ...]  — which chunks matched (by index)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:  # FAISS returns -1 for empty slots
            continue
        chunk = metadata[idx]
        logger.info(f"  RAG match: {chunk['path']} (distance: {distances[0][i]:.3f})")
        results.append({"path": chunk["path"], "content": chunk["content"]})

    return results
