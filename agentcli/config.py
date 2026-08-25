from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    planner_model: str = "openrouter/free"
    planner_max_retries: int = 2

    max_tasks: int = 20
    max_parallelism: int = 4
    task_max_retries: int = 2

    default_user_id: str = "local"

    task_type_models: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "planning": [
                "nvidia/nemotron-3-ultra-550b-a55b:free",
                "google/gemma-4-31b-it:free",
                "openrouter/free",
            ],
            "coding": [
                "google/gemma-4-31b-it:free",
                "cohere/north-mini-code:free",
                "nvidia/nemotron-3.5-lightning:free",
                "openrouter/free",
            ],
            "analysis": [
                "nvidia/nemotron-3-ultra-550b-a55b:free",
                "google/gemma-4-31b-it:free",
                "openrouter/free",
            ],
            "writing": [
                "google/gemma-4-31b-it:free",
                "nvidia/nemotron-3-ultra-550b-a55b:free",
                "minimax/minimax-m3:free",
                "openrouter/free",
            ],
            "general": [
                "nvidia/nemotron-3.5-lightning:free",
                "google/gemma-4-31b-it:free",
                "stealth/ox-alpha",
                "openrouter/free",
            ],
        }
    )

    db_path: Path = Path("agentcli.db")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()