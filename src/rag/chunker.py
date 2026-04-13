"""
Chunker — splits the React codebase into chunks for embedding.

Strategy: one chunk per file (with path as context).
For most React projects, components are one-per-file, so file-level
chunking maps naturally to component-level chunks.

Each chunk contains:
  - path: relative file path → used by WRITE node to know which file to edit
  - content: full file content → used by PLAN node to see all the code
  - embed_text: path + content preview → fed to embedding model for FAISS vector
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# File extensions to index
INDEX_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx", ".css", ".scss"}

# Directories to skip
SKIP_DIRS = {"node_modules", ".git", "build", "dist", "coverage", "public"}

# Maximum characters for embed_text — embedding models have token limits (~512 tokens).
# First 1500 chars is enough for FAISS to understand what the file is about.
MAX_EMBED_CHARS = 1500


def chunk_codebase(repo_path: Path) -> list[dict]:
    """Walk the codebase and create one chunk per file.

    Returns list of:
    {
        "path": "src/App.js",
        "content": "...full file...",
        "embed_text": "src/App.js\n\n...first 1500 chars..."
    }
    """
    chunks = []
    src_dir = repo_path / "src"

    if not src_dir.exists():
        logger.warning(f"No src/ directory found in {repo_path}")
        return chunks

    for file_path in src_dir.rglob("*"):
        # Skip directories and non-code files
        if file_path.is_dir():
            continue
        if file_path.suffix not in INDEX_EXTENSIONS:
            continue

        # Skip files inside excluded directories
        if any(skip in file_path.parts for skip in SKIP_DIRS):
            continue

        try:
            content = file_path.read_text()
            relative_path = str(file_path.relative_to(repo_path))

            # embed_text = path + content preview
            # Path gives FAISS the file name context ("App.js", "Header.jsx")
            # Content preview gives it the actual code patterns
            embed_text = f"{relative_path}\n\n{content[:MAX_EMBED_CHARS]}"

            chunks.append({
                "path": relative_path,
                "content": content,
                "embed_text": embed_text,
            })

            logger.info(f"Chunked: {relative_path} ({len(content)} chars)")

        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")

    logger.info(f"Total chunks: {len(chunks)}")
    return chunks
