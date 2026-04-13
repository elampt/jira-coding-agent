"""
GitHub client — creates Pull Requests via the GitHub API.

This is separate from git_ops.py:
- git_ops.py    = local git (clone, branch, commit, push) → GitPython
- github_client = remote API (create PR, get PR URL) → PyGithub

We extract owner/repo from the config URL:
  "https://github.com/Nirachan8/codingAgentUI" → "Nirachan8/codingAgentUI"
"""

import logging
from github import Github
from src.config import config, secrets

logger = logging.getLogger(__name__)


def _get_repo_full_name() -> str:
    """Extract 'owner/repo' from the config URL.

    "https://github.com/Nirachan8/codingAgentUI" → "Nirachan8/codingAgentUI"

    We split the URL by '/' and take the last two parts.
    """
    parts = config.target_repo.url.rstrip("/").split("/")
    return f"{parts[-2]}/{parts[-1]}"


def create_pull_request(branch_name: str, title: str, body: str) -> str:
    """Create a Pull Request on GitHub.

    Args:
        branch_name: Source branch with changes (e.g. "agent/KAN-4")
        title: PR title (e.g. "KAN-4: Change login button text")
        body: PR description (what changed, why, test results)

    Returns:
        The PR URL (e.g. "https://github.com/Nirachan8/codingAgentUI/pull/1")
    """
    g = Github(secrets.github_token)
    repo_full_name = _get_repo_full_name()
    repo = g.get_repo(repo_full_name)

    pr = repo.create_pull(
        title=title,
        body=body,
        head=branch_name,   # branch WITH changes (source)
        base="main",        # branch to merge INTO (target)
    )

    logger.info(f"PR created: {pr.html_url}")
    return pr.html_url
