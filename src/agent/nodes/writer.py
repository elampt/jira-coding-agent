"""
WRITE node — applies the edit plan to actual files on disk.

Input:  edit_plan + repo_path from state
Output: changes_made — list of descriptions of what was changed

Each edit is a string replacement: find old_string in the file, replace with new_string.
If old_string is not found (LLM hallucinated), that edit is skipped with a warning.
"""

import logging
from pathlib import Path
from src.agent.state import AgentState

logger = logging.getLogger(__name__)


def apply_changes(state: AgentState) -> dict:
    """WRITE node — called by LangGraph.

    Reads: edit_plan, repo_path from state
    Writes: changes_made to state
    """
    edit_plan = state["edit_plan"]
    repo_path = Path(state["repo_path"])

    logger.info(f"Applying {len(edit_plan)} edits")

    changes_made = []

    for edit in edit_plan:
        file_path = repo_path / edit["file"]

        # Check the file exists
        if not file_path.exists():
            logger.warning(f"File not found: {edit['file']} — skipping edit")
            changes_made.append(f"SKIPPED: {edit['file']} not found")
            continue

        # Read current content
        content = file_path.read_text()

        # Check old_string exists in the file
        if edit["old_string"] not in content:
            logger.warning(
                f"old_string not found in {edit['file']}: '{edit['old_string'][:50]}...' — skipping edit"
            )
            changes_made.append(f"SKIPPED: '{edit['old_string'][:30]}...' not found in {edit['file']}")
            continue

        # Apply the replacement
        new_content = content.replace(edit["old_string"], edit["new_string"], 1)
        # replace(..., 1) = replace only the FIRST occurrence
        # This prevents accidentally changing other identical strings in the file

        file_path.write_text(new_content)

        change_desc = f"{edit['file']}: replaced '{edit['old_string'][:30]}' with '{edit['new_string'][:30]}'"
        changes_made.append(change_desc)
        logger.info(f"  Applied: {change_desc}")

    logger.info(f"Done: {len(changes_made)} edits processed")

    return {"changes_made": changes_made}
