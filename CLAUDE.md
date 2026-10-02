# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An AI agent that watches Jira for new tickets, autonomously edits a **target React frontend repo** (a separate codebase, cloned at runtime — not this repo), runs its tests with a self-healing retry loop, takes before/after screenshots via Playwright MCP, and opens a GitHub PR. High-risk changes pause for human approval posted as a Jira comment. This repo (`jira-coding-agent`) is the agent/server itself; `config.yaml` points it at whichever React repo it operates on.

## Commands

```bash
make install      # uv sync — install all dependencies
make run           # uv run uvicorn src.server.app:app --reload --port 8000
make lint          # uv run ruff check src/
make format        # uv run ruff format src/ && ruff check src/ --fix
make type-check    # uv run pyright src/
make check         # lint + type-check — run before committing
make clean         # rm -rf workspace/ data/ screenshots/
```

There is no automated test suite in this repo (`tests/` is empty). To exercise the agent, use the local demo runner, which drives the LangGraph state machine directly against an already-cloned repo on disk — skipping Jira, ngrok, GitHub, `npm install`, and cloning:

```bash
python -m scripts.demo_local                              # run default ticket against workspace/KAN-99/codingAgentUI
python -m scripts.demo_local --reset                       # git checkout -- . in target repo first
python -m scripts.demo_local --summary "..." --description "..."
python -m scripts.demo_local --resume approved              # continue a paused (high-risk) run in a new process
python -m scripts.demo_local --auto-resume approved         # auto-continue a pause in the same process
python -m scripts.demo_local --no-jira                      # print the approval comment to stdout instead of posting to Jira
```

Both the server and the demo runner shell out to `npm`, so Node must be on the PATH of whatever process runs them (nvm users: the terminal must have loaded nvm, or `export PATH="$HOME/.nvm/versions/node/<version>/bin:$PATH"`). Without it every run fails with `Command not found: npm`. If you run uvicorn with `--reload`, wait for the reload to settle after multi-file edits — the reloader can load a half-edited file and never reload again.

Local dev needs `ngrok http 8000` to give Jira Cloud a public webhook URL (`.../webhook`); production skips this (see README's Production Deployment section for the AWS/Caddy setup).

## Architecture

### Two codebases, one process
The agent server (this repo) clones the **target repo** (configured in `config.yaml` under `target_repo`) into `workspace/<ISSUE_KEY>/`, edits it, tests it, screenshots it, and pushes/PRs it — all as a subprocess/git operation against that clone. Don't confuse edits to *this* repo with edits the agent makes to the target repo.

### Request flow (`src/server/app.py`)
`POST /webhook` returns 200 immediately and does everything else via `BackgroundTasks`:
- `jira:issue_created` → `process_new_ticket()`: clone target repo → `npm install` → `discard_tracked_changes()` (resets tracked files, i.e. npm's `package-lock.json` rewrite, so it never lands in the PR) → index codebase (FAISS) → screenshot BEFORE → create branch → invoke the LangGraph agent.
- `comment_created` → `process_comment()`: first skips the agent's own comments — **every** comment the agent posts starts with `🤖`, and this is the only way to tell them apart because the agent posts as the same Jira user as the human. Then only acts if the issue has a paused session in `_session_store`, and requires the comment body to be exactly `approve`/`approved`/`reject`/`rejected` (Jira mention prefixes stripped) — anything else is ignored. Every comment the agent posts triggers its own `comment_created` webhook, so the 🤖 check is what keeps it from reacting to itself. Keep the prefix on any new `add_comment` call.

`_session_store` (in-memory dict, issue_key → repo_path/branch_name/before_path/summary/description) exists because the *human approval* half of a run arrives as a separate later webhook request — it holds what `_finalize()` needs to know that the graph-internal `AgentState` doesn't carry across that gap. The LangGraph `MemorySaver` checkpointer (keyed by `thread_id` = issue_key) is what actually pauses/resumes the graph itself. Both are in-memory and lost on restart (noted as a known limitation, not yet fixed).

`_finalize()` runs after the graph completes or resumes: checks rejection, environment failure (tests couldn't run — comments and stops, no PR, status left alone) and test failure, bailing with a Jira comment in each case; otherwise screenshots AFTER, commits, pushes, builds the PR body, opens the PR, comments the link on Jira, and moves the ticket to "In Review". The PR body is the planner's plain-English `plan_summary` plus a "Files Changed" list read from the commit itself (`files_in_last_commit()`, not the plan — the fixer overwrites `edit_plan` on retries), with `raw.githubusercontent`-style screenshot links built from `SCREENSHOT_DIR_IN_REPO`.

### LangGraph agent (`src/agent/graph.py`, `src/agent/state.py`)
Nodes live in `src/agent/nodes/`, one file per node, wired into a `StateGraph(AgentState)`:

```
parse → search → plan → (risk check)
                          ├─ low/medium → write → test → END
                          │                        └─ fail (retry<3) → fix → write → test
                          └─ high → wait_approval → (human response)
                                        ├─ approved → write → test → END
                                        └─ rejected → END
```

- **parse** (`parser.py`): ticket text → `TicketPlan` (intent, component_hints, risk_level).
- **search** (`searcher.py`): combines RAG (FAISS semantic search) with grep so "navbar" in a ticket can still match a file literally called "header".
- **plan** (`planner.py`): LLM produces `EditInstruction` list — `{file, old_string, new_string}` pairs, not diffs or full-file rewrites; the design intent is "fewer, larger, atomic edits" over many small dependent ones.
- **wait_approval** (`approver.py`): posts the plan + risk concerns to Jira, then interrupts the graph (LangGraph `Command`/checkpoint mechanism) until a matching webhook comment resumes it.
- **write** (`writer.py`): applies `old_string`→`new_string` literally to files on disk.
- **test** (`tester.py`): runs `target_repo.test_command` from `config.yaml`. Sets `environment_failure=True` when the command couldn't run at all (binary not found, or "command not found" in the output, e.g. `react-scripts` missing). `graph.py` routes that straight to END — the LLM can't fix a missing `npm`, and given one it just invents edits. A test *timeout* is deliberately not an environment failure (an agent edit can cause an infinite loop, which the fixer can fix).
- **fix** (`fixer.py`): on test failure, shows the LLM the failure output plus source files chosen by `_files_to_show()` — files named in the failure output first (searched recursively, so nested tests work), then files the agent already edited (`worktree_changed_files()`), then the search results; capped at ~4K tokens (`MAX_CONTEXT_CHARS`) because Groq's free tier allows 8K tokens/minute. Falls back to every source file under `src/` if none of those apply. Loops back to `write`. Capped at `MAX_RETRIES = 3` (in `graph.py`); a session that runs out of retries ends via `_finalize()`'s "still failing" path rather than committing broken code.

All nodes read/write a shared `AgentState` TypedDict and return **partial** updates — LangGraph merges them. When extending the flow, add fields to `AgentState` (`src/agent/state.py`) rather than smuggling data through closures, and remember partial-update semantics when a node needs to *clear* a previous field.

### RAG pipeline (`src/rag/`)
`chunker.py` splits the target repo into embeddable chunks → `indexer.py` embeds them with `sentence-transformers` (model from `config.yaml`, CPU-only) and writes a FAISS index to `data/codebase.index` + a sidecar `data/codebase_metadata.json` (chunk metadata, since FAISS itself only stores vectors) → `retriever.py` embeds the query and does nearest-neighbor lookup, returning `{path, content}` dicts in the same shape as grep results so `searcher.py` can merge both. The index must be (re)built per target repo/ticket run — see `index_repo()` call in `process_new_ticket()` and `scripts/demo_local.py`.

### LLM access (`src/llm.py`)
Nodes never construct a chat client themselves. `get_llm()` builds it from `config.llm` (only the `groq` provider is implemented; anything else raises a clear error), and `invoke_structured(Schema, messages)` is what the parser, planner, fixer and approver call. It retries Groq's intermittent `tool_use_failed`/`json_validate_failed` 400s (the model emitting a malformed or unregistered tool call — seen with `gpt-oss-120b`) up to 3 times, then raises `LLMCallError` with a short message suitable for a Jira comment. Rate limits and 5xx errors are already retried by the Groq SDK. To change the model, edit `llm.model` in `config.yaml` — and check it still exists on your Groq account first (`llama-3.3-70b-versatile` disappeared from the free tier, which is why the project now uses `openai/gpt-oss-120b`).

### Integrations (`src/integrations/`)
Thin wrappers, one per external system: `jira_client.py` (read tickets, add comments, update status via `atlassian-python-api`), `github_client.py` (PR creation via PyGithub — note `_get_repo_full_name()` used by `app.py` to build raw-image URLs), `git_ops.py` (clone/branch/commit/push via GitPython).

### Playwright MCP (`src/mcp/playwright_client.py`, `src/agent/nodes/screenshotter.py`)
Visual verification uses the Playwright MCP server (not a direct Playwright dependency) to control a browser against the target repo's dev server (`src/tools/dev_server.py` starts/stops it). `_capture_screenshot()` is called directly from `app.py` (before/after), outside the LangGraph node graph itself.

### Config split (`src/config.py`)
`config.yaml` (committed, no secrets — LLM/embeddings/vector-store choice, target repo URL + test/lint/dev-server commands, Jira project key + auto-approve risk levels, screenshot dir) vs `.env` (gitignored secrets — API keys) are loaded into separate Pydantic models (`AppConfig`, `EnvSecrets`) and exposed as module-level singletons `config`/`secrets`, imported directly (`from src.config import config`) rather than passed around. To point the agent at a different target repo, change the LLM model, or move the FAISS index, edit `config.yaml` — no code changes needed (LLM: `src/llm.py`; index: `config.vector_store.index_file`/`metadata_file`). Two fields are loaded but **not yet used by any code**: `jira.auto_approve_risk_levels` (the graph pauses only when risk is `high`, so medium proceeds automatically regardless of this setting) and `jira.project_key` (the webhook does not filter by project).

### Observability (`src/observability.py`)
LangFuse callback handler, attached to the LangGraph `config["callbacks"]` per-invocation in `_agent_config()` (`app.py`) when LangFuse keys are set in `.env`; `None` (no-op) otherwise. Every node run and LLM call is traced this way, including retries and interrupted/resumed runs.

## Conventions worth knowing

- **Edits are string-replacement, not diffs.** The `write` node does literal `old_string`→`new_string` substitution per `EditInstruction`. Planner prompts constrain the LLM to precise, unique `old_string` snippets for this to work.
- **`ruff` line-length is 100, `E501` ignored** (formatter handles wrapping) — see `pyproject.toml`.
- **PyRight noise from third-party libs is deliberately suppressed** (`reportMissingTypeStubs`, `reportAttributeAccessIssue`, etc. set to `"none"` in `pyproject.toml`) — these come from `atlassian-python-api`/PyGithub lacking stubs and LangChain/LangGraph's intentionally broad union types (e.g. `with_structured_output()` returns `dict | BaseModel`), not from bugs in this codebase.
- **Torch is routed to the CPU-only PyPI index** (`[tool.uv.sources]` in `pyproject.toml`) to keep the Docker image small (~6 GB → ~800 MB) since deployment targets a CPU-only EC2 instance.
