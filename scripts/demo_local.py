"""
Local demo runner — exercises the LangGraph agent without Jira, ngrok or GitHub.

The production entry point (src/server/app.py) needs a Jira webhook, a public
HTTPS tunnel, a fresh clone and an npm install. None of that is interesting to
watch, and all of it can fail in front of an audience. This script skips
straight to the part that matters: the seven-node graph running against a repo
that is already on disk.

Usage:
    python -m scripts.demo_local                      # run the default ticket
    python -m scripts.demo_local --reset              # restore the repo first
    python -m scripts.demo_local --summary "..." --description "..."
    python -m scripts.demo_local --resume approved    # continue a paused run

What it does NOT do: clone, npm install, screenshot, commit, push, open a PR.
Those live in process_new_ticket() and are exercised by the deployed server.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env before importing src.config, which reads secrets at import time.
env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))

from langgraph.types import Command  # noqa: E402

from src.agent.graph import agent  # noqa: E402
from src.rag.indexer import index_repo  # noqa: E402

DEFAULT_REPO = ROOT / "workspace" / "KAN-99" / "codingAgentUI"
DEFAULT_ISSUE = "DEMO-1"
DEFAULT_SUMMARY = "Change the homepage link text from 'CPU Torch Test' to 'Learn React Docs'"
DEFAULT_DESCRIPTION = (
    "The anchor tag in the header currently reads 'CPU Torch Test', which was "
    "leftover test copy. It should read 'Learn React Docs' instead."
)

BAR = "=" * 78


def banner(text: str) -> None:
    print(f"\n{BAR}\n  {text}\n{BAR}")


def show(node: str, update: dict) -> None:
    """Pretty-print one node's contribution to state."""
    print(f"\n  ▸ NODE: {node}")

    if "ticket_plan" in update:
        plan = update["ticket_plan"]
        print(f"      intent          : {plan.get('intent')}")
        print(f"      component_hints : {plan.get('component_hints')}")
        print(f"      risk_level      : {plan.get('risk_level')}")

    if "relevant_files" in update:
        files = update["relevant_files"]
        print(f"      found {len(files)} file(s):")
        for f in files:
            print(f"        - {f['path']} ({len(f['content'])} chars)")

    if "edit_plan" in update:
        for i, edit in enumerate(update["edit_plan"], 1):
            print(f"      edit {i} → {edit['file']}")
            print(f"        OLD: {edit['old_string'][:70]!r}")
            print(f"        NEW: {edit['new_string'][:70]!r}")

    if "changes_made" in update:
        for change in update["changes_made"]:
            print(f"      applied: {change}")

    if "test_passed" in update:
        status = "PASSED" if update["test_passed"] else "FAILED"
        print(f"      tests: {status}")
        if not update["test_passed"]:
            tail = update.get("test_output", "")[-400:]
            print(f"      output tail:\n{tail}")

    if "retry_count" in update:
        print(f"      retry_count: {update['retry_count']}")

    if "approval_status" in update:
        print(f"      approval_status: {update['approval_status']}")


def _consume(stream, issue_key: str) -> bool:
    """Print each node update. Returns True if the graph paused at an interrupt."""
    for event in stream:
        for node, update in event.items():
            if node == "__interrupt__":
                banner("GRAPH PAUSED — checkpointed, waiting for a human")
                print(f"  thread_id = {issue_key}. Nothing is running; state is on the")
                print("  checkpointer until someone replies approve or reject.")
                return True
            show(node, update or {})
    return False


def reset_repo(repo: Path) -> None:
    """Discard working-tree changes so the demo is repeatable."""
    subprocess.run(["git", "checkout", "--", "."], cwd=str(repo), check=False)
    print(f"  Reset working tree in {repo}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Jira agent locally.")
    parser.add_argument("--repo", default=str(DEFAULT_REPO))
    parser.add_argument("--issue", default=DEFAULT_ISSUE)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION)
    parser.add_argument("--reset", action="store_true", help="git checkout -- . first")
    parser.add_argument("--resume", help="Resume a paused run: approved | rejected")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument(
        "--no-jira",
        action="store_true",
        help="Print the approval comment to stdout instead of posting it to Jira",
    )
    parser.add_argument(
        "--auto-resume",
        choices=["approved", "rejected"],
        help="If the graph pauses, immediately resume with this reply (same process)",
    )
    args = parser.parse_args()

    if args.no_jira:
        # The approval node posts to Jira before it interrupts. For a local demo
        # there is no real ticket to post to, so redirect the side effect to the
        # console — the graph mechanics are identical either way.
        from src.agent.nodes import approver

        def _print_comment(issue_key: str, body: str) -> None:
            print(f"\n  --- would post to Jira on {issue_key} ---")
            for line in body.splitlines():
                print(f"  | {line}")
            print("  --- end comment ---")

        approver.add_comment = _print_comment

    repo = Path(args.repo)
    if not repo.exists():
        print(f"ERROR: repo not found at {repo}")
        return 1

    config = {"configurable": {"thread_id": args.issue}}

    if args.resume:
        banner(f"RESUMING {args.issue} with '{args.resume}'")
        stream = agent.stream(Command(resume=args.resume), config=config, stream_mode="updates")
    else:
        if args.reset:
            banner("RESET")
            reset_repo(repo)

        if not args.skip_index:
            banner("INDEXING CODEBASE")
            index_repo(repo)

        banner("TICKET")
        print(f"  {args.issue}: {args.summary}")
        print(f"  repo: {repo}")

        banner("AGENT RUN")
        stream = agent.stream(
            {
                "issue_key": args.issue,
                "summary": args.summary,
                "description": args.description,
                "repo_path": str(repo),
                "branch_name": "demo-local",
            },
            config=config,
            stream_mode="updates",
        )

    paused = _consume(stream, args.issue)

    if paused and args.auto_resume:
        banner(f"HUMAN REPLIES '{args.auto_resume}' — resuming the same thread")
        # MemorySaver keeps checkpoints in this process's RAM, so the resume has
        # to happen here. In production the checkpointer would be PostgresSaver
        # and this could be a completely different request, hours later.
        _consume(
            agent.stream(Command(resume=args.auto_resume), config=config, stream_mode="updates"),
            args.issue,
        )
    elif paused:
        return 0

    banner("DONE")
    state = agent.get_state(config)
    print(f"  tests passed : {state.values.get('test_passed')}")
    print(f"  retries used : {state.values.get('retry_count', 0)}")
    print("\n  git diff:")
    sys.stdout.flush()
    subprocess.run(["git", "--no-pager", "diff", "--stat"], cwd=str(repo), check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
