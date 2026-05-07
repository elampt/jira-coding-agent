"""
TEST node — runs npm test and npm run lint after the agent makes changes.

If tests pass → continue to commit
If tests fail → the self-heal loop kicks in:
  1. Read the error message
  2. Send it to LLM: "here's the error, fix the code"
  3. LLM generates new edits
  4. WRITE applies them
  5. TEST runs again
  Max 3 retries, then give up.

The self-heal logic lives in graph.py (conditional edges).
This node just runs tests and reports pass/fail.
"""

import logging
import subprocess
from pathlib import Path

from src.agent.state import AgentState
from src.config import config

logger = logging.getLogger(__name__)

# Maximum characters of test output to keep — LLM doesn't need 500 lines of output,
# just the relevant error. Last 3000 chars usually contain the actual failure.
MAX_OUTPUT_CHARS = 3000


def _run_command(command: str, cwd: Path) -> tuple[int, str]:
    """Run a shell command and return (exit_code, output).

    Combines stdout + stderr into one string — some tools write
    errors to stdout, others to stderr, so we capture both.
    """
    try:
        result = subprocess.run(
            command.split(),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout — tests shouldn't take longer
            env={**__import__("os").environ, "CI": "true"},
            # CI=true tells Jest to run once and exit (no watch mode)
            # This is an alternative to --watchAll=false
        )
        output = result.stdout + "\n" + result.stderr
        return result.returncode, output.strip()

    except subprocess.TimeoutExpired:
        return 1, "ERROR: Tests timed out after 120 seconds"
    except FileNotFoundError:
        return 1, f"ERROR: Command not found: {command.split()[0]}"


def run_tests(state: AgentState) -> dict:
    """TEST node — called by LangGraph.

    Reads: repo_path from state
    Writes: test_passed, test_output to state

    Runs npm test. If it fails, the output is saved so the self-heal
    loop can feed it to the LLM for a fix.
    """
    repo_path = Path(state["repo_path"])
    test_command = config.target_repo.test_command  # "npm test" from config.yaml

    logger.info(f"Running tests: {test_command}")

    # Run tests
    exit_code, output = _run_command(test_command, repo_path)

    # Keep only the last N chars — the actual error is usually at the end
    trimmed_output = output[-MAX_OUTPUT_CHARS:] if len(output) > MAX_OUTPUT_CHARS else output

    if exit_code == 0:
        logger.info("Tests PASSED")
        return {
            "test_passed": True,
            "test_output": trimmed_output,
        }
    else:
        logger.warning(f"Tests FAILED (exit code {exit_code})")
        logger.warning(f"Output (last 500 chars): ...{trimmed_output[-500:]}")
        return {
            "test_passed": False,
            "test_output": trimmed_output,
        }
