"""
Git operations — handles all local Git work (clone, branch, commit, push).

This is LOCAL git operations on disk, not GitHub API.
- git_ops.py  = local (clone, branch, commit, push) → uses GitPython
- github_client.py = remote API (create PR) → uses PyGithub

The cloned repo lives in workspace/ (gitignored).
"""

import logging
import shutil
from pathlib import Path

from git import Repo

from src.config import config

logger = logging.getLogger(__name__)

# All cloned repos go here — gitignored so we don't commit someone else's code
WORKSPACE_DIR = Path("workspace")


def clone_repo(issue_key: str) -> Path:
    """Clone the target repo into workspace/<issue_key>/<repo_name>.

    Each ticket gets its own folder so concurrent tickets don't collide:
      workspace/KAN-4/codingAgentUI/
      workspace/KAN-5/codingAgentUI/

    If this ticket's workspace exists, delete and re-clone (clean slate).
    Returns the path to the cloned repo.
    """
    repo_url = config.target_repo.url
    # Extract repo name from URL: "https://github.com/Nirachan8/codingAgentUI" → "codingAgentUI"
    repo_name = repo_url.rstrip("/").split("/")[-1]
    ticket_dir = WORKSPACE_DIR / issue_key
    repo_path = ticket_dir / repo_name

    # Clean slate — remove if exists from a previous run
    if ticket_dir.exists():
        shutil.rmtree(ticket_dir)
        logger.info(f"Removed existing workspace at {ticket_dir}")

    # Create the ticket-specific directory
    ticket_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Cloning {repo_url} into {repo_path}")
    Repo.clone_from(repo_url, repo_path)
    logger.info("Clone complete")

    return repo_path


def create_branch(repo_path: Path, issue_key: str) -> str:
    """Create and checkout a new branch named agent/<issue_key>.

    The 'agent/' prefix makes it clear this branch was created by
    the AI agent, not a human developer.

    Returns the branch name.
    """
    repo = Repo(repo_path)
    branch_prefix = config.target_repo.branch_prefix  # "agent/" from config
    branch_name = f"{branch_prefix}{issue_key}"

    # Create and checkout the new branch
    repo.git.checkout("-b", branch_name)
    logger.info(f"Created and checked out branch: {branch_name}")

    return branch_name


def commit_changes(repo_path: Path, issue_key: str, message: str) -> None:
    """Stage all changes and create a commit.

    git add . → stages all modified/new/deleted files
    git commit -m "KAN-4: <message>"
    """
    repo = Repo(repo_path)

    # Stage all changes
    repo.git.add(".")

    # Commit with ticket key prefix for traceability
    commit_message = f"{issue_key}: {message}"
    repo.index.commit(commit_message)
    logger.info(f"Committed: {commit_message}")


def push_branch(repo_path: Path, branch_name: str) -> None:
    """Push the branch to the remote (GitHub).

    'origin' is the default remote name that git clone creates,
    pointing back to the GitHub repo URL.

    If the branch already exists on the remote (from a previous run),
    we force-push to overwrite it. This ensures re-processing a
    ticket always works cleanly.
    """
    repo = Repo(repo_path)

    # Delete remote branch if it exists (from a previous run)
    try:
        repo.git.push("origin", "--delete", branch_name)
        logger.info(f"Deleted existing remote branch: {branch_name}")
    except Exception:
        pass  # Branch doesn't exist on remote — that's fine

    repo.git.push("origin", branch_name)
    logger.info(f"Pushed branch: {branch_name}")
