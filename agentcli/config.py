from functools import lru_cache
from pathlib import Path
from typing import Literal

from platformdirs import user_data_dir
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_NAME = "agentcli"


def _default_db_path() -> Path:
    """Return XDG-compliant database path: ~/.local/share/agentcli/agentcli.db"""
    data_dir = Path(user_data_dir(APP_NAME))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "agentcli.db"


def _find_dotenv() -> str | None:
    """Search for .env file: CWD first, then ~/.config/agentcli/"""
    local = Path(".env")
    if local.is_file():
        return str(local)

    config_dir = Path.home() / ".config" / APP_NAME
    config_env = config_dir / ".env"
    if config_env.is_file():
        return str(config_env)

    return None


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    base_url: str = ""
    api_key: str = ""
    timeout: float = 60.0
    enabled: bool = True
    priority: int = 0  # higher = tried first


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_find_dotenv(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- OpenRouter ---
    openrouter_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # --- Freebuff ---
    freebuff_enabled: bool = Field(default=False, validation_alias="FREEBUFF_ENABLED")
    freebuff_timeout: float = Field(default=120.0, validation_alias="FREEBUFF_TIMEOUT")
    freebuff_token: str = Field(default="", validation_alias="FREEBUFF_TOKEN")
    freebuff_auto_install: bool = Field(default=False, validation_alias="FREEBUFF_AUTO_INSTALL")

    # --- OpenCode ---
    opencode_api_key: str = Field(default="", validation_alias="OPENCODE_API_KEY")
    opencode_base_url: str = Field(
        default="https://opencode.ai/zen/v1", validation_alias="OPENCODE_BASE_URL"
    )
    opencode_tier: str = Field(
        default="zen",
        validation_alias="OPENCODE_TIER",
        description="'zen' for pay-per-use or 'go' for low-cost subscription",
    )

    # --- Generic custom providers (JSON in env) ---
    custom_providers: dict[str, ProviderConfig] = Field(
        default_factory=dict,
        description="Map of provider_name -> config for custom OpenAI-compatible endpoints",
    )

    planner_model: str = "openrouter/free"
    planner_max_retries: int = 2

    # --- Planner review / refinement ---
    # Master switch for heuristic plan review (quality scoring, budget,
    # contract coverage). Structural checks (cycles, unknown deps) always run.
    planner_review_enabled: bool = Field(
        default=True, validation_alias="PLANNER_REVIEW_ENABLED"
    )
    # A plan is accepted as-is when every task scores at or above this
    # threshold (0..1). Weaker plans trigger one critique re-plan round.
    planner_min_task_score: float = Field(
        default=0.55, validation_alias="PLANNER_MIN_TASK_SCORE"
    )
    # Fraction of tasks that must declare expected_outputs /
    # validation_criteria for a plan to pass review without re-planning.
    planner_min_contract_coverage: float = Field(
        default=0.6, validation_alias="PLANNER_MIN_CONTRACT_COVERAGE"
    )
    # Budget guard: sum of complexity weights (low=1, medium=2, high=4)
    # the plan may not exceed. Encourages splitting high-complexity tasks.
    planner_max_estimated_cost: float = Field(
        default=40.0, validation_alias="PLANNER_MAX_ESTIMATED_COST"
    )

    max_tasks: int = 20
    max_parallelism: int = 4
    task_max_retries: int = 2

    default_user_id: str = "local"

    # Model chains use provider prefix:
    #   "openrouter/model-name", "freebuff/default", "opencode/model-name"
    #
    # OpenCode models are selected by task complexity:
    #   planning / analysis  → stronger reasoning models (opus, gemini-2.5-pro)
    #   coding               → code-specialized models (sonnet, codex)
    #   writing / general    → fast, capable models (flash, haiku)
    task_type_models: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "planning": [
                "opencode/anthropic/claude-sonnet-4-5",
                "opencode/google/gemini-2.5-pro",
                "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
                "openrouter/google/gemma-4-31b-it:free",
                "freebuff/default",
                "openrouter/openrouter/free",
            ],
            "coding": [
                "opencode/anthropic/claude-sonnet-4-5",
                "opencode/openai/codex-mini-latest",
                "opencode/google/gemini-2.5-flash",
                "openrouter/google/gemma-4-31b-it:free",
                "openrouter/cohere/north-mini-code:free",
                "freebuff/default",
                "openrouter/openrouter/free",
            ],
            "analysis": [
                "opencode/anthropic/claude-sonnet-4-5",
                "opencode/google/gemini-2.5-pro",
                "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
                "openrouter/google/gemma-4-31b-it:free",
                "freebuff/default",
                "openrouter/openrouter/free",
            ],
            "writing": [
                "opencode/google/gemini-2.5-flash",
                "opencode/anthropic/claude-haiku-3-5",
                "openrouter/google/gemma-4-31b-it:free",
                "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
                "openrouter/minimax/minimax-m3:free",
                "freebuff/default",
                "openrouter/openrouter/free",
            ],
            "general": [
                "opencode/google/gemini-2.5-flash",
                "opencode/anthropic/claude-haiku-3-5",
                "openrouter/nvidia/nemotron-3.5-lightning:free",
                "openrouter/google/gemma-4-31b-it:free",
                "freebuff/default",
                "openrouter/openrouter/free",
            ],
        }
    )

    db_path: Path = Field(default_factory=_default_db_path)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
