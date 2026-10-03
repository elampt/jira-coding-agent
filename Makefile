.PHONY: help install run lint format type-check check evals clean

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
	@echo "  make clean        Remove workspace, data, screenshots"
	@echo "  make help         Show this help message"

install:  ## Install dependencies
	uv sync

run:  ## Start the FastAPI server
	uv run uvicorn src.server.app:app --reload --port 8000

lint:  ## Run Ruff linter
	uv run ruff check src/ evals/

format:  ## Auto-format code with Ruff
	uv run ruff format src/ evals/
	uv run ruff check src/ evals/ --fix

type-check:  ## Run PyRight static type checker
	uv run pyright src/

check: lint type-check  ## Run lint + type-check
	@echo "✅ All checks passed"

evals:  ## Run the eval harness (see evals/README.md)
	uv run python -m evals.run $(ARGS)

clean:  ## Remove workspace, data, screenshots
	rm -rf workspace/ data/ screenshots/
	@echo "✅ Cleaned workspace, data, screenshots"
