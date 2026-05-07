"""
Playwright MCP client — wrapper around the @playwright/mcp server.

Why MCP instead of using Playwright directly?
- MCP is Anthropic's standard for tool integration
- The agent calls tools via a standardized protocol (not custom code)
- Same pattern works for any MCP server (filesystem, GitHub, Slack, etc.)
- Demonstrates familiarity with cutting-edge AI tooling

How it works:
  1. Spawn @playwright/mcp as a subprocess via npx
  2. Open MCP session (initialize handshake)
  3. Call tools: browser_navigate, browser_take_screenshot
  4. Close session

The MCP server runs Playwright internally — we never touch Playwright directly.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.client.stdio import stdio_client

from mcp import ClientSession, StdioServerParameters

logger = logging.getLogger(__name__)


@asynccontextmanager
async def playwright_session():
    """Open an MCP session with the Playwright server.

    Used as: async with playwright_session() as session: ...
    Automatically spawns the server, initializes, and cleans up.
    """
    # Configure how to start the MCP server
    # npx will download @playwright/mcp the first time, cache for next time
    server_params = StdioServerParameters(
        command="npx",
        args=["--yes", "@playwright/mcp@latest", "--headless"],
        env=None,
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the MCP protocol handshake
            await session.initialize()
            logger.info("Playwright MCP session opened")
            yield session
            logger.info("Playwright MCP session closed")


async def take_screenshot_async(url: str, output_path: Path) -> bool:
    """Navigate to URL and save a screenshot.

    Args:
        url: The URL to screenshot (e.g., http://localhost:3000)
        output_path: Where to save the screenshot

    Returns: True on success, False on failure
    """
    try:
        async with playwright_session() as session:
            # 1. Navigate to the URL
            logger.info(f"  Playwright: navigating to {url}")
            await session.call_tool("browser_navigate", arguments={"url": url})

            # 2. Take a screenshot
            logger.info(f"  Playwright: taking screenshot → {output_path}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            await session.call_tool(
                "browser_take_screenshot",
                arguments={
                    "filename": str(output_path),
                    "fullPage": True,  # capture entire page, not just viewport
                },
            )

            # The MCP server saves the screenshot to disk for us
            return output_path.exists()

    except Exception as e:
        logger.error(f"Screenshot failed: {e}")
        return False


def take_screenshot(url: str, output_path: Path) -> bool:
    """Sync wrapper around the async screenshot function.

    LangGraph nodes are sync functions — this lets them call the async MCP code.
    """
    return asyncio.run(take_screenshot_async(url, output_path))
