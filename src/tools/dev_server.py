"""
Dev server helper — start, wait for ready, kill.

We can't screenshot a React app without it running.
The flow is: start → wait → screenshot → kill (then repeat for "after" screenshot).

Why kill between screenshots?
1. Saves RAM (especially on EC2 t2.micro with 1GB)
2. Forces a fresh load with the agent's new code
"""

import logging
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from src.config import config

logger = logging.getLogger(__name__)

# Maximum seconds to wait for dev server to become responsive
STARTUP_TIMEOUT = 60


def start_dev_server(repo_path: Path) -> subprocess.Popen:
    """Start the React dev server in the background.

    Returns the subprocess.Popen object so we can kill it later.
    """
    command = config.target_repo.dev_server_command  # "npm run dev" from config
    logger.info(f"Starting dev server: {command} (in {repo_path})")

    # Start as background process — returns immediately, server runs in background
    # We pipe output to DEVNULL so the dev server's logs don't clutter our logs
    # start_new_session=True puts it in its own process group so we can kill the whole tree
    import os

    process = subprocess.Popen(
        command.split(),
        cwd=str(repo_path),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "BROWSER": "none", "CI": "false"},
        # BROWSER=none stops Create React App from auto-opening a browser tab
        # CI=false ensures interactive mode (CI=true would exit immediately)
        start_new_session=True,  # creates its own process group → easier to kill
    )

    return process


def wait_for_server(url: str, timeout: int = STARTUP_TIMEOUT) -> bool:
    """Poll the URL every second until it responds, or timeout.

    Returns True if server is responsive, False if timed out.
    """
    logger.info(f"Waiting for dev server at {url}...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = urllib.request.urlopen(url, timeout=2)
            if response.status == 200:
                elapsed = int(time.time() - start_time)
                logger.info(f"Dev server ready in {elapsed}s")
                return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            # Server not ready yet — keep waiting
            pass

        time.sleep(1)

    logger.error(f"Dev server did not start within {timeout}s")
    return False


def stop_dev_server(process: subprocess.Popen) -> None:
    """Kill the dev server subprocess and any children.

    npm start spawns child processes (Webpack, etc.) — we need to kill the whole tree.
    """
    if process.poll() is None:  # still running
        # Kill the process group so all children die too
        try:
            import os
            import signal

            # Send SIGTERM to the whole process group
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=5)
        except (ProcessLookupError, OSError):
            pass
        except subprocess.TimeoutExpired:
            # If SIGTERM didn't work, force kill
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass

        logger.info("Dev server stopped")
