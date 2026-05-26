# =========================================================================
# Stage 1: BUILDER — install Python dependencies into a virtual environment
# =========================================================================
FROM python:3.10-slim AS builder

# Install UV (modern Python package manager, replaces pip + venv)
# Pinned to a specific version for reproducible builds
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy ONLY dependency files first — Docker layer caching trick.
# If these don't change, the slow `uv sync` step is cached on rebuild.
COPY pyproject.toml uv.lock ./

# Install dependencies into /app/.venv
# --no-dev: skip dev dependencies (ruff, pyright) — not needed at runtime
# --frozen: must match uv.lock exactly (no resolution at install time)
# UV_HTTP_TIMEOUT=600: 10 min timeout — some ML libs (torch, triton) are huge
ENV UV_HTTP_TIMEOUT=600
RUN uv sync --no-dev --frozen --no-cache


# =========================================================================
# Stage 2: RUNTIME — minimal image with only what's needed to RUN the agent
# =========================================================================
FROM python:3.10-slim

# Install system packages the agent needs at RUNTIME:
#  - git: for cloning target repos and pushing PR branches
#  - curl: for healthchecks
#  - nodejs + npm: for `npm install` and `npm test` inside the agent
# --no-install-recommends: skip optional extras → smaller image
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security (don't run apps as root)
RUN useradd --create-home --shell /bin/bash appuser

# Switch to the app directory
WORKDIR /app

# Copy the virtual environment from the builder stage
# This is the multi-stage magic: build tools stay in builder, runtime only gets the venv
COPY --from=builder /app/.venv /app/.venv

# Copy application code (after deps so code changes don't bust the deps cache)
COPY src/ ./src/
COPY config.yaml ./

# Set ownership so non-root user can read/write
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Make the venv's bin directory the default PATH
# So `uvicorn` is found without needing `uv run` inside the container
ENV PATH="/app/.venv/bin:$PATH"

# Visual verification (Playwright) needs Chromium browser binaries.
# We skip it by default in containers — agent runs everything else.
# To enable: pass -e ENABLE_VISUAL_VERIFICATION=true at runtime + ensure Chromium is installed.
ENV ENABLE_VISUAL_VERIFICATION=false

# Tell Docker which port the container listens on (documentation only;
# actual exposure happens with `-p` flag or compose port mapping)
EXPOSE 8000

# Health check — Docker uses this to know if the container is healthy
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# The command that runs when the container starts.
# We use exec form (JSON array) so the process gets PID 1 and receives signals properly.
CMD ["uvicorn", "src.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
