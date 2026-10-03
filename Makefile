.PHONY: help install run lint format type-check check evals report clean

help:  ## Show this help message
	@echo "Jira Coding Agent — Available commands:"
	@echo ""
	@echo "  make install      Install all dependencies (UV sync)"
	@echo "  make run          Start the FastAPI server (with reload)"
	@echo "  make lint         Run Ruff linter"
	@echo "  make format       Auto-format code with Ruff"
	@echo "  make type-check   Run PyRight static type checker"
	@echo "  make check        Run lint + type-check (pre-commit)"
	@echo "  make evals        Run the eval harness (ARGS=\"--repeat 3\" to pass options)"
	@echo "  make report       Summarise recent runs from LangFuse (ARGS=\"--env prod\")"
	@echo "  make clean        Remove workspace, data, screenshots"
	@echo "  make help         Show this help message"

install:  ## Install dependencies
	uv sync

run:  ## Start the FastAPI server
	uv run uvicorn src.server.app:app --reload --port 8000

lint:  ## Run Ruff linter
	uv run ruff check src/ evals/ scripts/

format:  ## Auto-format code with Ruff
	uv run ruff format src/ evals/ scripts/
	uv run ruff check src/ evals/ scripts/ --fix

type-check:  ## Run PyRight static type checker
	uv run pyright src/

check: lint type-check  ## Run lint + type-check
	@echo "✅ All checks passed"

evals:  ## Run the eval harness (see evals/README.md)
	uv run python -m evals.run $(ARGS)

report:  ## Summarise recent agent runs from LangFuse
	uv run python -m scripts.trace_report $(ARGS)

clean:  ## Remove workspace, data, screenshots
	rm -rf workspace/ data/ screenshots/
	@echo "✅ Cleaned workspace, data, screenshots"
