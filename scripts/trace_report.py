"""
Summarise recent agent runs from LangFuse: per-step latency, tokens, estimated cost, outcomes.

    python -m scripts.trace_report                    # last 30 traces, real tickets and eval runs
    python -m scripts.trace_report --env prod         # only real tickets (tag "prod")
    python -m scripts.trace_report --env eval --limit 60

A paused-then-resumed ticket produces two traces under one session (the ticket key), so the
report counts tickets by session and steps by trace. Cost is an estimate at the list price in
config.yaml (llm.price_per_million_tokens); the free tier is not actually billed.
"""

import argparse
import base64
import contextlib
import io
import json
import math
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
for _line in (ROOT / ".env").read_text().splitlines():
    if _line.strip() and not _line.startswith("#") and "=" in _line:
        _key, _value = _line.split("=", 1)
        os.environ.setdefault(_key, _value.strip().strip('"').strip("'"))

from src.config import config  # noqa: E402

STEP_ORDER = ["parse", "search", "plan", "wait_approval", "write", "test", "fix"]
RISKS = ["low", "medium", "high", "unknown"]
OUTCOMES = [
    "success",
    "awaiting_approval",
    "rejected",
    "tests_failing",
    "environment_failure",
    "error",
    "unlabelled",
]
PAUSE_BETWEEN_CALLS = 1.0  # LangFuse's API rate-limits aggressively


# ---------------------------------------------------------------- LangFuse API


class LangFuse:
    def __init__(self) -> None:
        self.host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        public, secret = (
            os.environ.get("LANGFUSE_PUBLIC_KEY"),
            os.environ.get("LANGFUSE_SECRET_KEY"),
        )
        if not public or not secret:
            sys.exit("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set in .env")
        self.auth = base64.b64encode(f"{public}:{secret}".encode()).decode()

    def get(self, path: str, **params) -> dict:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        request = urllib.request.Request(
            f"{self.host}{path}?{query}", headers={"Authorization": f"Basic {self.auth}"}
        )
        for attempt in range(6):
            time.sleep(PAUSE_BETWEEN_CALLS)
            try:
                return json.load(urllib.request.urlopen(request, timeout=30))
            except urllib.error.HTTPError as error:
                if error.code != 429:
                    raise
                wait = int(error.headers.get("Retry-After") or 5 * (attempt + 1))
                print(f"  (rate limited, waiting {wait}s)", file=sys.stderr)
                time.sleep(wait)
        sys.exit("LangFuse kept rate-limiting; try again in a minute or lower --limit.")

    def pages(self, path: str, max_pages: int = 40, **params):
        for page in range(1, max_pages + 1):
            rows = self.get(path, page=page, limit=100, **params).get("data", [])
            yield from rows
            if len(rows) < 100:
                return


# ---------------------------------------------------------------- helpers


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def seconds(observation: dict) -> float | None:
    if observation.get("startTime") and observation.get("endTime"):
        return (
            parse_time(observation["endTime"]) - parse_time(observation["startTime"])
        ).total_seconds()
    return None


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile; with few samples the high percentiles are just the max."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def cost_usd(tokens_in: int, tokens_out: int) -> float | None:
    price = config.llm.price_per_million_tokens
    if price is None:
        return None
    return tokens_in * price.input / 1e6 + tokens_out * price.output / 1e6


# ---------------------------------------------------------------- analysis


def collect(client: LangFuse, env: str, limit: int) -> dict:
    traces = client.get(
        "/api/public/traces",
        limit=min(limit, 100),
        orderBy="timestamp.desc",
        tags=None if env == "all" else env,
    ).get("data", [])
    if not traces:
        sys.exit(f"No traces found (env={env}). Run a ticket or `make evals ARGS=--trace` first.")

    trace_ids = {t["id"] for t in traces}
    oldest = min(parse_time(t["timestamp"]) for t in traces) - timedelta(minutes=1)
    observations = [
        o
        for o in client.pages("/api/public/observations", fromStartTime=oldest.isoformat())
        if o["traceId"] in trace_ids
    ]
    labels: dict[str, dict[str, str]] = defaultdict(dict)
    for name in ("outcome", "risk_level"):
        for score in client.pages("/api/public/v2/scores", max_pages=5, name=name):
            if score.get("sessionId"):
                labels[score["sessionId"]][name] = score.get("stringValue") or str(
                    score.get("value")
                )
    return {"traces": traces, "observations": observations, "labels": labels}


def step_of(observation: dict, by_id: dict[str, dict], roots: set[str]) -> str | None:
    """Walk up to the observation sitting directly under the root: that is the agent step."""
    current: dict | None = observation
    while current and current.get("parentObservationId") not in roots:
        current = by_id.get(current.get("parentObservationId") or "")
    return current["name"] if current else None


def analyse(data: dict) -> dict:
    by_trace: dict[str, list[dict]] = defaultdict(list)
    for observation in data["observations"]:
        by_trace[observation["traceId"]].append(observation)

    step_seconds: dict[str, list[float]] = defaultdict(list)
    step_tokens: dict[str, list[int]] = defaultdict(list)
    models: set[str] = set()
    session_tokens: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    session_seconds: dict[str, float] = defaultdict(float)

    for trace in data["traces"]:
        session = trace.get("sessionId") or trace["id"]
        session_seconds[session] += trace.get("latency") or 0.0
        observations = by_trace.get(trace["id"], [])
        by_id = {o["id"]: o for o in observations}
        roots = {o["id"] for o in observations if not o.get("parentObservationId")}

        for observation in observations:
            if observation["type"] == "CHAIN" and observation.get("parentObservationId") in roots:
                if (duration := seconds(observation)) is not None:
                    step_seconds[observation["name"]].append(duration)
            elif observation["type"] == "GENERATION":
                usage = observation.get("usage") or {}
                tokens_in, tokens_out = usage.get("input") or 0, usage.get("output") or 0
                step_tokens[step_of(observation, by_id, roots) or "?"].append(
                    tokens_in + tokens_out
                )
                session_tokens[session][0] += tokens_in
                session_tokens[session][1] += tokens_out
                if observation.get("model"):
                    models.add(observation["model"])

    return {
        "step_seconds": step_seconds,
        "step_tokens": step_tokens,
        "models": sorted(models),
        "session_tokens": session_tokens,
        "session_seconds": session_seconds,
    }


def git_version() -> str:
    def out(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()

    dirty = "+uncommitted" if out("status", "--porcelain", "--", "src", "evals", "scripts") else ""
    return out("rev-parse", "--short", "HEAD") + dirty


# ---------------------------------------------------------------- output


def print_report(data: dict, stats: dict, env: str) -> None:
    traces = data["traces"]
    sessions = {t.get("sessionId") or t["id"] for t in traces}
    first, last = min(t["timestamp"] for t in traces), max(t["timestamp"] for t in traces)
    print(f"\nLangFuse report  env={env}  traces={len(traces)}  tickets(sessions)={len(sessions)}")
    print(f"window {first[:16]} -> {last[:16]} UTC   models: {', '.join(stats['models']) or '-'}")

    print(f"\n{'step':16} {'runs':>5} {'median s':>9} {'p95 s':>7} {'max s':>7} {'avg tokens':>11}")
    steps = [s for s in STEP_ORDER if s in stats["step_seconds"]]
    steps += sorted(set(stats["step_seconds"]) - set(STEP_ORDER))
    for step in steps:
        values = stats["step_seconds"][step]
        tokens = stats["step_tokens"].get(step)
        avg_tokens = f"{statistics.mean(tokens):.0f}" if tokens else "-"
        print(
            f"{step:16} {len(values):>5} {statistics.median(values):>9.2f} "
            f"{percentile(values, 0.95):>7.2f} {max(values):>7.2f} {avg_tokens:>11}"
        )

    totals = [sum(v) for v in stats["session_tokens"].values()]
    walls = list(stats["session_seconds"].values())
    costs = [cost_usd(*v) for v in stats["session_tokens"].values()]
    print("\nper ticket (a paused ticket's two traces are summed)")
    if totals:
        print(
            f"  tokens        median {statistics.median(totals):.0f}   max {max(totals)}   total {sum(totals)}"
        )
    print(f"  wall seconds  median {statistics.median(walls):.1f}   max {max(walls):.1f}")
    if costs and all(c is not None for c in costs):
        print(
            f"  est. cost     median ${statistics.median(costs):.5f}   total ${sum(costs):.5f}  (list price)"
        )

    table: dict[str, dict[str, int]] = {o: defaultdict(int) for o in OUTCOMES}
    for session in sessions:
        label = data["labels"].get(session, {})
        outcome = label.get("outcome", "unlabelled")
        risk = label.get("risk_level", "unknown")
        table[outcome if outcome in table else "error"][risk if risk in RISKS else "unknown"] += 1
    print(f"\n{'outcome':22} " + " ".join(f"{r:>8}" for r in RISKS) + f" {'total':>7}")
    for outcome in OUTCOMES:
        row = table[outcome]
        if sum(row.values()):
            print(
                f"{outcome:22} "
                + " ".join(f"{row[r]:>8}" for r in RISKS)
                + f" {sum(row.values()):>7}"
            )

    if "wait_approval" in stats["step_seconds"]:
        print(
            "\nnote: a paused ticket runs wait_approval twice (the second, on resume, is instant), "
            "which pulls its median down."
        )
    if len(traces) < 20:
        print(
            "\nnote: fewer than 20 traces — percentiles are rough, and p95 is effectively the max."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise recent agent runs from LangFuse.")
    parser.add_argument("--env", choices=["all", "prod", "eval"], default="all")
    parser.add_argument(
        "--limit", type=int, default=30, help="most recent traces to read (max 100)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="also write the report to reports/trace_report_<time>_<env>.txt",
    )
    args = parser.parse_args()

    client = LangFuse()
    data = collect(client, args.env, args.limit)

    now = datetime.now(timezone.utc)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print(f"generated {now:%Y-%m-%d %H:%M} UTC   code {git_version()}")
        print_report(data, analyse(data), args.env)
    text = buffer.getvalue()
    print(text, end="")

    if args.save:
        REPORTS.mkdir(exist_ok=True)
        path = REPORTS / f"trace_report_{now:%Y-%m-%d_%H%M}_{args.env}.txt"
        path.write_text(text)
        print(f"\nsaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
