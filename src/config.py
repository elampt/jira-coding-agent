"""
Configuration loader.

Loads settings from two sources:
1. config.yaml  → project settings (LLM model, repo URL, commands)
2. .env         → secrets (API keys, tokens)

Why two files?
- config.yaml is committed to git (no secrets, safe to share)
- .env is gitignored (contains secrets, never committed)
"""

from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# --- Structured config from config.yaml ---


class LLMConfig(BaseModel):
    provider: str
    model: str


class EmbeddingsConfig(BaseModel):
    provider: str
    model: str


class VectorStoreConfig(BaseModel):
    provider: str
    index_path: str


class TargetRepoConfig(BaseModel):
    url: str
    branch_prefix: str
    test_command: str
    lint_command: str
    dev_server_command: str
    dev_server_url: str


class JiraProjectConfig(BaseModel):
    project_key: str
    auto_approve_risk_levels: list[str]


class PlaywrightConfig(BaseModel):
    screenshot_dir: str


class AppConfig(BaseModel):
    """All settings from config.yaml, validated by Pydantic."""

    llm: LLMConfig
    embeddings: EmbeddingsConfig
    vector_store: VectorStoreConfig
    target_repo: TargetRepoConfig
    jira: JiraProjectConfig
    playwright: PlaywrightConfig


# --- Secrets from .env ---


class EnvSecrets(BaseSettings):
    """API keys and tokens loaded from .env file.

    pydantic-settings automatically reads from environment variables
    and .env files. Field names map to env var names (case-insensitive).
    """

    groq_api_key: str = ""
    jira_base_url: str = ""
    jira_email: str = ""
    jira_api_token: str = ""
    github_token: str = ""

    # LangFuse — for LLM observability (free tier)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    model_config = {"env_file": ".env"}


# --- Load everything ---


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Load and validate config.yaml."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    return AppConfig(**raw)


def load_secrets() -> EnvSecrets:
    """Load secrets from .env file."""
    return EnvSecrets()


# Singleton instances — import these from other modules
config = load_config()
secrets = load_secrets()
