"""
SCREENSHOT node — visual verification using Playwright MCP.

Two screenshots are taken:
  - "before" → captured at the start (right after clone, before agent changes anything)
  - "after"  → captured at the end (after WRITE + TEST passes)

Both are saved to disk and later embedded in the PR description so reviewers
can visually compare what changed.

Flow for each screenshot:
  1. Start dev server (npm run dev)
  2. Wait for it to respond at localhost:3000
  3. Use Playwright MCP to take screenshot
  4. Stop dev server (saves RAM)
"""

import logging
from pathlib import Path

from src.agent.state import AgentState
from src.config import config
from src.mcp.playwright_client import take_screenshot
from src.tools.dev_server import start_dev_server, wait_for_server, stop_dev_server

logger = logging.getLogger(__name__)


def _capture_screenshot(repo_path: Path, output_path: Path, label: str) -> bool:
    """Start dev server, take a screenshot, kill dev server.

    Args:
        repo_path: Path to the React project
        output_path: Where to save the PNG
        label: "before" or "after" — for logging

    Returns: True on success, False on failure
    """
    logger.info(f"Capturing {label} screenshot...")

    process = start_dev_server(repo_path)
    try:
        # Wait for the dev server to become responsive
        if not wait_for_server(config.target_repo.dev_server_url):
            logger.error(f"{label}: dev server didn't start, skipping screenshot")
            return False

        # Take the screenshot via Playwright MCP
        success = take_screenshot(config.target_repo.dev_server_url, output_path)

        if success:
            logger.info(f"{label} screenshot saved → {output_path}")
        else:
            logger.warning(f"{label}: screenshot failed")

        return success

    finally:
        # Always kill the dev server, even if something failed
        stop_dev_server(process)


def screenshot_before(state: AgentState) -> dict:
    """SCREENSHOT (BEFORE) node — captures the UI before any changes.

    Reads: repo_path, issue_key from state
    Writes: screenshot_before path to state
    """
    repo_path = Path(state["repo_path"])
    issue_key = state["issue_key"]

    output_path = Path(config.playwright.screenshot_dir) / issue_key / "before.png"
    success = _capture_screenshot(repo_path, output_path, "BEFORE")

    return {"screenshot_before": str(output_path) if success else ""}


def screenshot_after(state: AgentState) -> dict:
    """SCREENSHOT (AFTER) node — captures the UI after agent changes + tests pass.

    Reads: repo_path, issue_key from state
    Writes: screenshot_after path to state
    """
    repo_path = Path(state["repo_path"])
    issue_key = state["issue_key"]

    output_path = Path(config.playwright.screenshot_dir) / issue_key / "after.png"
    success = _capture_screenshot(repo_path, output_path, "AFTER")

    return {"screenshot_after": str(output_path) if success else ""}
