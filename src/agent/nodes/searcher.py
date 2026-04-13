"""
SEARCH node — finds relevant files using RAG (semantic) + grep (exact).

Two search strategies combined:
  1. RAG: embed ticket text → FAISS finds semantically similar files
         "navigation bar" → finds "header" (different words, same meaning)
  2. Grep: search for exact strings from component_hints
         "Learn React" → finds the exact text in App.js

Together they catch both semantic matches AND exact string matches.

Input:  ticket_plan.component_hints + repo_path + summary from state
Output: relevant_files — list of {path, content} for files that match
"""

import logging
import subprocess
from pathlib import Path
from src.agent.state import AgentState
from src.rag.retriever import retrieve_similar

logger = logging.getLogger(__name__)

# File extensions we care about in a React project
SEARCH_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx", ".css", ".scss", ".json"}

# Directories to skip — never relevant, waste of time
SKIP_DIRS = {"node_modules", ".git", "build", "dist", "coverage"}


def grep_codebase(repo_path: Path, keyword: str) -> set[str]:
    """Grep the codebase for a keyword. Returns set of matching file paths.

    Uses subprocess to run grep — faster than reading every file in Python.
    -r = recursive, -l = only print file names (not matching lines),
    --include = only search these file types.
    """
    matching_files = set()

    for ext in SEARCH_EXTENSIONS:
        try:
            result = subprocess.run(
                ["grep", "-rl", "--include", f"*{ext}", keyword, str(repo_path / "src")],
                capture_output=True,
                text=True,
            )
            # Each line of stdout is a file path
            for line in result.stdout.strip().split("\n"):
                if line:
                    matching_files.add(line)
        except Exception as e:
            logger.warning(f"Grep failed for '{keyword}' with ext {ext}: {e}")

    return matching_files


def read_file_content(file_path: str, repo_path: Path) -> dict:
    """Read a file and return {path (relative), content}.

    We store the relative path (e.g. "src/App.js") not the absolute path,
    because the PLAN and WRITE nodes need relative paths for edits.
    """
    full_path = Path(file_path)
    content = full_path.read_text()
    # Convert absolute path to relative: /workspace/KAN-5/codingAgentUI/src/App.js → src/App.js
    relative_path = str(full_path.relative_to(repo_path))

    return {"path": relative_path, "content": content}


def search_codebase(state: AgentState) -> dict:
    """SEARCH node — called by LangGraph.

    Reads: ticket_plan.component_hints, repo_path, summary from state
    Writes: relevant_files to state

    Combines two strategies:
      1. RAG — semantic search using ticket summary
      2. Grep — exact string match using component hints
    Deduplicates by file path so each file appears once.
    """
    hints = state["ticket_plan"]["component_hints"]
    repo_path = Path(state["repo_path"])
    summary = state["summary"]

    logger.info(f"Searching codebase for hints: {hints}")

    # --- Strategy 1: RAG (semantic search) ---
    logger.info("  RAG search...")
    rag_results = retrieve_similar(query=summary, top_k=5)

    # --- Strategy 2: Grep (exact match) ---
    logger.info("  Grep search...")
    all_matching_files = set()
    for hint in hints:
        matches = grep_codebase(repo_path, hint)
        all_matching_files.update(matches)
        logger.info(f"    '{hint}' → {len(matches)} files")

    grep_results = []
    for file_path in all_matching_files:
        grep_results.append(read_file_content(file_path, repo_path))

    # --- Combine and deduplicate ---
    relevant_files = []
    seen_paths = set()

    # Add RAG results first (semantic matches)
    for f in rag_results:
        if f["path"] not in seen_paths:
            seen_paths.add(f["path"])
            relevant_files.append(f)
            logger.info(f"  [RAG] {f['path']}")

    # Add grep results (exact matches, skip duplicates)
    for f in grep_results:
        if f["path"] not in seen_paths:
            seen_paths.add(f["path"])
            relevant_files.append(f)
            logger.info(f"  [GREP] {f['path']}")

    if not relevant_files:
        logger.warning("No relevant files found!")
    else:
        logger.info(f"  Total: {len(relevant_files)} unique files")

    return {"relevant_files": relevant_files}
