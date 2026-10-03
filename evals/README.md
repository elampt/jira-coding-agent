# Eval harness

Runs the real agent graph against fixed, known tickets and checks what it did against what each case promised. Run it before and after any change to the agent, the prompts, or the model, so "did that make it worse?" has an answer.

No Jira, GitHub, or ngrok is needed. It does need Node (found automatically under `~/.nvm` if your shell didn't load it) and `GROQ_API_KEY` in `.env`.

## Quick start

```bash
.venv/bin/python -m evals.run --list                  # show the cases
.venv/bin/python -m evals.run                         # every case, once
.venv/bin/python -m evals.run --repeat 3              # every case 3x — model output varies run to run
.venv/bin/python -m evals.run --cases low_risk_text_change,llm_failure_ends_cleanly
.venv/bin/python -m evals.run --cases forced_test_failure_self_heals --keep   # keep the temp repo to inspect
```

The first run installs the fixture's dependencies once (`npm ci`, needs network) into `evals/.cache/`. Later runs reuse that copy.

## What happens in one run

1. Copy `evals/fixtures/react-app/` (a pinned, trimmed React app) to a temp folder, add any files the case needs, and commit it as the baseline.
2. Clone the cached `node_modules` into it (copy-on-write, so it's instant).
3. Run the real LangGraph agent on the case's ticket. Jira calls are stubbed, and the search index goes to a temp folder so a running dev server's `data/` is never touched.
4. For "break it on purpose" cases, change exactly one thing: remove npm from `PATH`, or make every model call fail.
5. Compare the outcome with the case's `expect` block, count model calls and tokens, and save everything.

## The cases (`cases.yaml`)

Each case is data. The file starts with a key to every field and check.

| Case | What it proves |
|---|---|
| `low_risk_text_change` | A one-line change runs straight through and passes first time |
| `high_risk_pauses_then_approves` | Risky ticket pauses before writing; "approve" finishes it |
| `high_risk_rejected_changes_nothing` | "reject" stops cleanly and leaves the code untouched |
| `forced_test_failure_self_heals` | A test the planner can't see (in `src/build/`, which search skips) fails; the fixer must repair the right file without undoing the ticket's change |
| `environment_failure_skips_fixer` | Missing npm is reported as an environment problem; the fixer is never called |
| `llm_failure_ends_cleanly` | Persistent model errors end with one short `LLMCallError` and no code touched |

All current cases are `kind: behavior`: contracts the system should always honour. A `kind: quality` set ("does it solve real tickets well?", measured as a pass rate over many tickets) needs a richer fixture and doesn't exist yet.

## Reading the results

```
case                                     pass   prev   tokens   secs
low_risk_text_change                   2/2    100%     5287     32
```

- **pass**: runs that met every check. **prev**: that case's pass rate in the previous saved run.
- `<-- REGRESSION`: pass rate fell compared with the previous run. `<-- BELOW BAR`: under the case's `min_pass_rate` (default 100%).
- Exit code is 1 if any case is flagged, 0 otherwise.
- With few runs, treat one-off differences as noise. A single run per side can't distinguish a real regression from variance; use `--repeat`.

Every run is saved to `evals/results/<UTC time>_<git sha>.json` (git-ignored) with the model name and code version. Each run records the failed checks, files changed, tokens, the final `git diff`, and the agent's own log lines (plans, edits, fixer explanations), so a failure can be diagnosed without re-running.

## Cost

A run that reaches the model costs about 2.4K–5K tokens (parse + plan, plus approval summary or fixer calls). `llm_failure_ends_cleanly` costs nothing. One pass of the whole suite is roughly 19K tokens. Groq's free tier allows about 200K tokens/day and 8K tokens/minute; the runner paces itself by tokens spent so the per-minute cap is never hit.

## Adding a case

Append to `cases.yaml` using the fields documented at the top. Give it a `why`, a `ticket`, and an `expect` block. Prefer checks on the final state (files changed, file contents, tests passing) over checks on the path the model took. If the case relies on a trap (a hidden test, a missing tool), run it against known-bad code once to confirm it can actually fail.

## Design notes

- **Pinned fixture.** `fixtures/FIXTURE_SOURCE` records the commit it was cut from, so results don't change when anyone edits the upstream repo.
- **Hidden test lives in `src/build/`.** Both halves of the search (RAG indexer and grep) skip `build/`, so the planner can't see it, but Jest still runs it.
- **Why repeats.** The same ticket can pass today and fail tomorrow. Pass rate over several runs is the honest measure.
- **Limits.** One small fixture app; no quality set yet; results are local only.
