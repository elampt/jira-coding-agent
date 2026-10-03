"""
Eval harness for the Jira coding agent.

    python -m evals.run                         # every case, once
    python -m evals.run --repeat 3              # every case, three times (LLM output varies)
    python -m evals.run --cases low_risk_text_change,llm_failure_ends_cleanly
    python -m evals.run --list

Each run: fresh copy of evals/fixtures/react-app -> real agent graph -> check `expect` from
evals/cases.yaml -> save evals/results/<time>_<git sha>.json and compare with the previous run.
Jira is stubbed, and the search index lives in a temp folder so a running dev server's
data/ is never touched. See evals/README.md.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "evals"
FIXTURE = EVALS / "fixtures" / "react-app"
CACHE = EVALS / ".cache"
RESULTS = EVALS / "results"

# src.config reads config.yaml (relative path) and .env at import time.
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
for _line in (ROOT / ".env").read_text().splitlines():
    if _line.strip() and not _line.startswith("#") and "=" in _line:
        _key, _value = _line.split("=", 1)
        os.environ.setdefault(_key, _value.strip().strip('"').strip("'"))

import yaml  # noqa: E402
from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402
from langgraph.types import Command  # noqa: E402

import src.llm as llm_module  # noqa: E402
import src.rag.indexer as indexer_module  # noqa: E402
import src.rag.retriever as retriever_module  # noqa: E402
from src.agent.graph import agent  # noqa: E402
from src.agent.nodes import approver  # noqa: E402
from src.config import config  # noqa: E402
from src.integrations.git_ops import worktree_changed_files  # noqa: E402

# Groq's free tier allows 8K tokens/minute; pace runs to ~6K to stay clear of it.
TOKENS_PER_SECOND_BUDGET = 100


# ---------------------------------------------------------------- environment helpers


def ensure_node_on_path() -> None:
    """The fixture needs npm. Find an nvm-installed Node if the shell didn't load one."""
    if shutil.which("npm"):
        return
    for bin_dir in sorted((Path.home() / ".nvm" / "versions" / "node").glob("*/bin"), reverse=True):
        if (bin_dir / "npm").exists():
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"
            return
    sys.exit("npm not found. Install Node, or add it to PATH, then re-run.")


def ensure_node_modules() -> Path:
    """Install the fixture's dependencies once; every run clones this cache (copy-on-write)."""
    cache = CACHE / "node_modules"
    if cache.exists():
        return cache
    print("First run: installing the fixture's dependencies once (npm ci)...", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "app"
        shutil.copytree(FIXTURE, work)
        subprocess.run(["npm", "ci", "--no-audit", "--no-fund"], cwd=work, check=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        shutil.move(str(work / "node_modules"), str(cache))
    return cache


def _git(repo: Path, *args: str) -> None:
    identity = ["-c", "user.email=eval@local", "-c", "user.name=eval"]
    subprocess.run(["git", "-C", str(repo), *identity, *args], check=True, capture_output=True)


def make_workdir(case: dict, node_modules: Path) -> Path:
    """Fresh, git-tracked copy of the fixture (plus the case's setup files) with node_modules."""
    root = Path(tempfile.mkdtemp(prefix=f"eval-{case['id']}-"))
    repo = root / "app"
    shutil.copytree(FIXTURE, repo)
    for rel, text in ((case.get("setup") or {}).get("write_files") or {}).items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "baseline")
    # cp -c = APFS clone (instant, no extra disk). Fall back to a normal copy elsewhere.
    cloned = subprocess.run(["cp", "-cR", str(node_modules), str(repo / "node_modules")])
    if cloned.returncode != 0:
        shutil.copytree(node_modules, repo / "node_modules", symlinks=True)
    return repo


# ---------------------------------------------------------------- measuring and faults


class UsageCounter(BaseCallbackHandler):
    """Counts model calls and tokens for one run (what LangFuse shows, without the network)."""

    def __init__(self) -> None:
        self.calls = 0
        self.tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:  # noqa: ANN001
        self.calls += 1
        for generations in response.generations:
            for generation in generations:
                metadata = getattr(getattr(generation, "message", None), "usage_metadata", None)
                self.tokens += (metadata or {}).get("total_tokens", 0)


class LogCapture(logging.Handler):
    """Collects the agent's own log lines (plans, edits, fix explanations) for one run."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(f"{record.name.removeprefix('src.agent.')}: {record.getMessage()}"[:300])


def _final_diff(repo: Path) -> str:
    """What the run left behind: tracked-file diff plus any new files (capped for the JSON)."""
    subprocess.run(["git", "-C", str(repo), "add", "-N", "."], capture_output=True)
    diff = subprocess.run(
        ["git", "-C", str(repo), "diff", "HEAD"], capture_output=True, text=True
    ).stdout
    return diff[:4000] if diff else "(no changes: working tree identical to baseline)"


def _always_failing_llm():
    """Stand-in for get_llm(): every structured call raises Groq's tool_use_failed 400."""
    import groq
    import httpx

    body = {
        "error": {
            "message": "Tool call validation failed: attempted to call tool 'exec' "
            "which was not in request.tools",
            "code": "tool_use_failed",
        }
    }

    class _Structured:
        def invoke(self, messages):  # noqa: ANN001
            response = httpx.Response(400, request=httpx.Request("POST", "http://eval"))
            raise groq.BadRequestError(f"Error code: 400 - {body}", response=response, body=body)

    class _LLM:
        def with_structured_output(self, schema):  # noqa: ANN001
            return _Structured()

    return _LLM()


def _case_patches(case: dict, index_dir: Path) -> list:
    patches = [
        mock.patch.object(indexer_module, "INDEX_FILE", index_dir / "codebase.index"),
        mock.patch.object(indexer_module, "METADATA_FILE", index_dir / "codebase_metadata.json"),
        mock.patch.object(retriever_module, "INDEX_FILE", index_dir / "codebase.index"),
        mock.patch.object(retriever_module, "METADATA_FILE", index_dir / "codebase_metadata.json"),
        mock.patch.object(approver, "add_comment", lambda issue_key, body: None),
    ]
    if case.get("inject") == "llm_always_fails":
        patches.append(mock.patch.object(llm_module, "get_llm", _always_failing_llm))
    if (case.get("env") or {}).get("no_node"):
        patches.append(mock.patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}))
    return patches


# ---------------------------------------------------------------- one run of one case


def run_once(case: dict, repeat_index: int, node_modules: Path, keep: bool) -> dict:
    issue = f"EVAL-{case['id']}-{repeat_index}-{uuid.uuid4().hex[:6]}"
    repo = make_workdir(case, node_modules)
    index_dir = repo.parent / "index"
    index_dir.mkdir()
    usage = UsageCounter()
    run_config = {"configurable": {"thread_id": issue}, "callbacks": [usage]}
    inputs = {
        "issue_key": issue,
        "summary": case["ticket"]["summary"],
        "description": case["ticket"]["description"],
        "repo_path": str(repo),
        "branch_name": "eval",
    }

    observed: dict = {"paused_first": False, "error": "none"}
    capture = LogCapture()
    agent_logger = logging.getLogger("src.agent")
    previous_level = agent_logger.level
    agent_logger.setLevel(logging.INFO)
    agent_logger.addHandler(capture)
    started = time.time()
    try:
        with ExitStack() as stack:
            for patch in _case_patches(case, index_dir):
                stack.enter_context(patch)
            indexer_module.index_repo(repo)
            agent.invoke(inputs, config=run_config)
            if agent.get_state(run_config).next:
                observed["paused_first"] = True
                if case.get("resume"):
                    agent.invoke(Command(resume=case["resume"]), config=run_config)
    except Exception as exc:  # noqa: BLE001 — the run's outcome IS the exception for some cases
        observed["error"] = type(exc).__name__
        observed["error_message"] = str(exc)[:200]
    finally:
        agent_logger.removeHandler(capture)
        agent_logger.setLevel(previous_level)

    state = agent.get_state(run_config).values or {}
    observed.update(
        tests_passed=bool(state.get("test_passed", False)),
        environment_failure=bool(state.get("environment_failure", False)),
        retries=int(state.get("retry_count", 0)),
        approval_status=state.get("approval_status"),
        files_changed=sorted(worktree_changed_files(repo)),
        llm_calls=usage.calls,
        tokens=usage.tokens,
        seconds=round(time.time() - started, 1),
        diff=_final_diff(repo),
        log=capture.lines[-60:],
    )

    failures = evaluate(case["expect"], observed, repo)
    result = {"repeat": repeat_index, "passed": not failures, "failures": failures, **observed}
    if keep:
        result["kept_at"] = str(repo)
    else:
        shutil.rmtree(repo.parent, ignore_errors=True)
    return result


def evaluate(expect: dict, observed: dict, repo: Path) -> list[str]:
    """Compare what happened with what the case promised. Returns a list of failed checks."""
    failures: list[str] = []

    def check(ok: bool, message: str) -> None:
        if not ok:
            failures.append(message)

    def same(key: str, label: str | None = None) -> None:
        if key in expect:
            got = observed[label or key]
            check(got == expect[key], f"{key}: expected {expect[key]!r}, got {got!r}")

    same("paused_first")
    same("approval_status")
    same("tests_passed")
    same("environment_failure")
    same("error")
    if "retries_min" in expect:
        check(
            observed["retries"] >= expect["retries_min"],
            f"retries {observed['retries']} < min {expect['retries_min']}",
        )
    if "retries_max" in expect:
        check(
            observed["retries"] <= expect["retries_max"],
            f"retries {observed['retries']} > max {expect['retries_max']}",
        )
    if "llm_calls_max" in expect:
        check(
            observed["llm_calls"] <= expect["llm_calls_max"],
            f"llm_calls {observed['llm_calls']} > max {expect['llm_calls_max']}",
        )
    for path in expect.get("files_changed_include", []):
        check(
            path in observed["files_changed"],
            f"expected {path} to be changed; changed: {observed['files_changed']}",
        )
    if expect.get("no_files_changed"):
        check(
            not observed["files_changed"], f"expected no changes, got {observed['files_changed']}"
        )
    for path, text in (expect.get("file_contains") or {}).items():
        content = (repo / path).read_text() if (repo / path).exists() else ""
        check(text in content, f"{path} should contain {text!r}")
    for path, text in (expect.get("file_not_contains") or {}).items():
        content = (repo / path).read_text() if (repo / path).exists() else ""
        check(text not in content, f"{path} should not contain {text!r}")
    return failures


# ---------------------------------------------------------------- results and comparison


def git_version() -> str:
    def out(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()

    dirty = "+uncommitted" if out("status", "--porcelain", "--", "src", "evals") else ""
    return out("rev-parse", "--short", "HEAD") + dirty


def latest_previous() -> dict | None:
    files = sorted(RESULTS.glob("*.json"))
    return json.loads(files[-1].read_text()) if files else None


def summarise(case_results: list[dict], previous: dict | None) -> bool:
    """Print the table. Returns True if everything met its bar with no regression."""
    prev_rates = {c["id"]: c["pass_rate"] for c in (previous or {}).get("cases", [])}
    print(f"\n{'case':38} {'pass':>6} {'prev':>6} {'tokens':>8} {'secs':>6}")
    all_ok = True
    for case in case_results:
        runs = case["runs"]
        rate = case["pass_rate"]
        prev = prev_rates.get(case["id"])
        regressed = prev is not None and rate < prev
        short = rate < case["min_pass_rate"] or regressed
        all_ok &= not short
        tokens = sum(r["tokens"] for r in runs)
        secs = sum(r["seconds"] for r in runs)
        passed = sum(r["passed"] for r in runs)
        prev_text = "-" if prev is None else f"{prev:.0%}"
        flag = "  <-- REGRESSION" if regressed else ("  <-- BELOW BAR" if short else "")
        print(
            f"{case['id']:38} {passed}/{len(runs):<4} {prev_text:>6} {tokens:>8} {secs:>6.0f}{flag}"
        )
        for run in runs:
            for failure in run["failures"]:
                print(f"    run {run['repeat']}: {failure}")
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the agent eval cases.")
    parser.add_argument("--cases", help="comma-separated case ids (default: all)")
    parser.add_argument("--repeat", type=int, default=1, help="runs per case (LLM output varies)")
    parser.add_argument("--list", action="store_true", help="list cases and exit")
    parser.add_argument(
        "--keep", action="store_true", help="keep each run's temp folder for debugging"
    )
    args = parser.parse_args()

    cases = yaml.safe_load((EVALS / "cases.yaml").read_text())
    if args.list:
        for case in cases:
            print(f"{case['id']:38} [{case['kind']}] {' '.join(case['why'].split())[:90]}")
        return 0
    if args.cases:
        wanted = set(args.cases.split(","))
        unknown = wanted - {c["id"] for c in cases}
        if unknown:
            sys.exit(f"Unknown case id(s): {sorted(unknown)}")
        cases = [c for c in cases if c["id"] in wanted]

    ensure_node_on_path()
    node_modules = ensure_node_modules()
    previous = latest_previous()
    version = git_version()
    print(f"model={config.llm.model}  code={version}  cases={len(cases)}  repeat={args.repeat}")

    case_results = []
    for case in cases:
        runs = []
        for i in range(1, args.repeat + 1):
            result = run_once(case, i, node_modules, args.keep)
            runs.append(result)
            print(
                f"  {case['id']} #{i}: {'PASS' if result['passed'] else 'FAIL'} "
                f"({result['seconds']}s, {result['tokens']} tokens, {result['llm_calls']} calls)",
                flush=True,
            )
            # Pace by tokens spent so the free tier's tokens-per-minute cap is never hit.
            time.sleep(max(0.0, result["tokens"] / TOKENS_PER_SECOND_BUDGET - result["seconds"]))
        pass_rate = sum(r["passed"] for r in runs) / len(runs)
        case_results.append(
            {
                "id": case["id"],
                "kind": case["kind"],
                "min_pass_rate": case.get("min_pass_rate", 1.0),
                "pass_rate": pass_rate,
                "runs": runs,
            }
        )

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS / f"{stamp}_{version.replace('+', '-')}.json"
    out_path.write_text(
        json.dumps(
            {
                "timestamp": stamp,
                "model": config.llm.model,
                "code_version": version,
                "repeat": args.repeat,
                "cases": case_results,
            },
            indent=2,
        )
    )
    ok = summarise(case_results, previous)
    print(
        f"\nsaved {out_path.relative_to(ROOT)}  (compared with: "
        f"{previous['timestamp'] + ' / ' + previous['code_version'] if previous else 'no previous run'})"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
