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

    # --- Generic custom providers (JSON in env) ---
    custom_providers: dict[str, ProviderConfig] = Field(
        default_factory=dict,
        description="Map of provider_name -> config for custom OpenAI-compatible endpoints",
    )

    planner_model: str = "openrouter/free"
    planner_max_retries: int = 2

    max_tasks: int = 20
    max_parallelism: int = 4
    task_max_retries: int = 2

    default_user_id: str = "local"

    # Model chains use provider prefix: "openrouter/model-name" or "freebuff/model-name"
    task_type_models: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "planning": [
                "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
                "openrouter/google/gemma-4-31b-it:free",
                "freebuff/default",
                "openrouter/openrouter/free",
            ],
            "coding": [
                "openrouter/google/gemma-4-31b-it:free",
                "openrouter/cohere/north-mini-code:free",
                "openrouter/nvidia/nemotron-3.5-lightning:free",
                "freebuff/default",
                "openrouter/openrouter/free",
            ],
            "analysis": [
                "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
                "openrouter/google/gemma-4-31b-it:free",
                "freebuff/default",
                "openrouter/openrouter/free",
            ],
            "writing": [
                "openrouter/google/gemma-4-31b-it:free",
                "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
                "openrouter/minimax/minimax-m3:free",
                "freebuff/default",
                "openrouter/openrouter/free",
            ],
            "general": [
                "openrouter/nvidia/nemotron-3.5-lightning:free",
                "openrouter/google/gemma-4-31b-it:free",
                "openrouter/stealth/ox-alpha",
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