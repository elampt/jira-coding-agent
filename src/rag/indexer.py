"""
Indexer — embeds code chunks and stores them in FAISS.

Two things are saved to disk:
  1. FAISS index (data/codebase.index) — the vectors for similarity search
  2. Metadata (data/codebase_metadata.json) — path + content for each chunk

FAISS only stores vectors (numbers), not text. So when FAISS says
"chunk #3 is closest", we look up metadata[3] to get the actual file.

Usage:
  python -m src.rag.indexer              ← run from project root
  or: from src.rag.indexer import index_repo
"""

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.rag.chunker import chunk_codebase
from src.config import config

logger = logging.getLogger(__name__)

# The embedding model — runs locally, free, no API key needed
# all-MiniLM-L6-v2 produces 384-dimensional vectors
MODEL_NAME = config.embeddings.model
INDEX_DIR = Path("data")


def index_repo(repo_path: Path) -> None:
    """Index the entire codebase: chunk → embed → store in FAISS.

    Saves two files:
      data/codebase.index — FAISS vector index
      data/codebase_metadata.json — chunk metadata (path, content)
    """
    # Step 1: Chunk the codebase
    chunks = chunk_codebase(repo_path)
    if not chunks:
        logger.warning("No chunks to index!")
        return

    # Step 2: Load the embedding model (downloads on first run, ~80MB)
    logger.info(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Step 3: Embed all chunks
    embed_texts = [chunk["embed_text"] for chunk in chunks]
    logger.info(f"Embedding {len(embed_texts)} chunks...")
    embeddings = model.encode(embed_texts)
    # embeddings shape: (num_chunks, 384) — each chunk is a 384-dim vector

    # Step 4: Create FAISS index
    dimension = embeddings.shape[1]  # 384
    index = faiss.IndexFlatL2(dimension)
    # IndexFlatL2 = brute-force L2 (euclidean) distance search
    # Simple and exact — fine for small codebases (<1000 files)
    # For millions of files, you'd use IndexIVFFlat (approximate, faster)

    index.add(np.array(embeddings).astype("float32"))
    logger.info(f"FAISS index created with {index.ntotal} vectors")

    # Step 5: Save to disk
    INDEX_DIR.mkdir(exist_ok=True)

    faiss.write_index(index, str(INDEX_DIR / "codebase.index"))
    logger.info(f"Saved FAISS index to {INDEX_DIR / 'codebase.index'}")

    # Save metadata — FAISS doesn't store text, only vectors
    # metadata[i] corresponds to the vector at index i in FAISS
    metadata = [{"path": c["path"], "content": c["content"]} for c in chunks]
    with open(INDEX_DIR / "codebase_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata for {len(metadata)} chunks")


# Allow running as: python -m src.rag.indexer
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    # Default to workspace repo, or pass a custom path
    if len(sys.argv) > 1:
        repo = Path(sys.argv[1])
    else:
        # Find the first repo in workspace/
        workspace = Path("workspace")
        repos = list(workspace.glob("*/codingAgentUI"))
        if repos:
            repo = repos[0]
        else:
            print("No repo found. Pass a path: python -m src.rag.indexer /path/to/repo")
            sys.exit(1)

    print(f"Indexing: {repo}")
    index_repo(repo)
    print("Done!")
