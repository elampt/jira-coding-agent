"""
SEARCH node — finds relevant files in the codebase using grep.

Input:  ticket_plan.component_hints + repo_path from state
Output: relevant_files — list of {path, content} for files that match

Phase 2: Simple grep search.
Phase 3: Will add RAG (FAISS + embeddings) for semantic search.
"""

import logging
import subprocess
from pathlib import Path
from src.agent.state import AgentState

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

    Reads: ticket_plan.component_hints, repo_path from state
    Writes: relevant_files to state

    Greps for each component hint, collects all matching files,
    reads their full content, and deduplicates.
    """
    hints = state["ticket_plan"]["component_hints"]
    repo_path = Path(state["repo_path"])

    logger.info(f"Searching codebase for hints: {hints}")

    # Grep for each hint, collect all matching file paths
    all_matching_files = set()
    for hint in hints:
        matches = grep_codebase(repo_path, hint)
        all_matching_files.update(matches)
        logger.info(f"  '{hint}' → {len(matches)} files")

    # Read content of each matching file
    relevant_files = []
    seen_paths = set()  # deduplicate — a file might match multiple hints
    for file_path in all_matching_files:
        if file_path not in seen_paths:
            seen_paths.add(file_path)
            file_context = read_file_content(file_path, repo_path)
            relevant_files.append(file_context)
            logger.info(f"  Found: {file_context['path']}")

    if not relevant_files:
        logger.warning("No relevant files found!")

    return {"relevant_files": relevant_files}
