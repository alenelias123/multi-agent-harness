import asyncio
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .config import settings
from .providers.base import BaseProvider, ProviderError
from .providers.freebuff import FreebuffClient
from .providers.openrouter import OpenRouterClient

logger = logging.getLogger(__name__)


def _parse_model_ref(model_ref: str) -> tuple[str, str]:
    """Parse 'provider/model-name' into (provider_name, model_name).

    If there is no '/' the provider defaults to 'openrouter'.
    """
    if "/" not in model_ref:
        return "openrouter", model_ref
    # 'openrouter/google/gemma-4-31b-it:free' -> ('openrouter', 'google/gemma-4-31b-it:free')
    first, rest = model_ref.split("/", 1)
    known = {"openrouter", "freebuff"}
    if first in known:
        return first, rest
    # Unknown prefix — treat the whole thing as an openrouter model
    return "openrouter", model_ref


@dataclass
class ModelStats:
    failures: int = 0
    last_failure: float = 0
    consecutive_failures: int = 0


class ModelRouter:
    """Routes LLM calls across multiple providers with fallback."""

    def __init__(self, providers: dict[str, BaseProvider]) -> None:
        self._providers = providers
        self._model_stats: dict[str, ModelStats] = defaultdict(ModelStats)
        self._lock = asyncio.Lock()

    def get_model_chain(self, task_type: str) -> list[str]:
        return settings.task_type_models.get(task_type, settings.task_type_models["general"])

    def _is_model_healthy(self, model: str) -> bool:
        stats = self._model_stats[model]
        if stats.consecutive_failures >= 3:
            return time.time() - stats.last_failure > 60
        return True

    def _record_failure(self, model: str) -> None:
        stats = self._model_stats[model]
        stats.failures += 1
        stats.last_failure = time.time()
        stats.consecutive_failures += 1

    def _record_success(self, model: str) -> None:
        stats = self._model_stats[model]
        stats.consecutive_failures = 0

    def _get_available_models(self, task_type: str) -> list[str]:
        chain = self.get_model_chain(task_type)
        return [m for m in chain if self._is_model_healthy(m)]

    def _get_provider(self, model_ref: str) -> tuple[BaseProvider, str]:
        """Resolve a provider and strip the prefix from the model name."""
        provider_name, model_name = _parse_model_ref(model_ref)
        provider = self._providers.get(provider_name)
        if provider is None:
            raise ModelRoutingError(
                f"Unknown provider '{provider_name}' in model ref '{model_ref}'. "
                f"Available: {list(self._providers.keys())}"
            )
        return provider, model_name

    async def call(
        self,
        task_type: str,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4000,
        response_format: dict[str, Any] | None = None,
        task_id: str | None = None,  # noqa: ARG002
    ) -> tuple[str, str]:
        models = self._get_available_models(task_type)
        if not models:
            models = self.get_model_chain(task_type)

        last_error: Exception | None = None

        for model_ref in models:
            try:
                provider, model_name = self._get_provider(model_ref)
                async for attempt in AsyncRetrying(
                    retry=retry_if_exception_type((ProviderError, httpx.HTTPStatusError)),
                    wait=wait_exponential_jitter(initial=1, max=10),
                    stop=stop_after_attempt(2),
                    reraise=True,
                ):
                    with attempt:
                        response = await provider.chat_completion(
                            model=model_name,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            response_format=response_format,
                        )
                        self._record_success(model_ref)
                        return response, model_ref
            except Exception as e:
                self._record_failure(model_ref)
                last_error = e
                logger.warning(f"Model {model_ref} failed for task_type={task_type}: {e}")
                continue

        raise ModelRoutingError(f"All models failed for task_type={task_type}: {last_error}")


class ModelRoutingError(Exception):
    pass


@asynccontextmanager
async def create_model_router() -> Any:
    providers: dict[str, BaseProvider] = {}

    # OpenRouter (always available if key is set)
    if settings.openrouter_api_key:
        client = OpenRouterClient(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )
        await client.__aenter__()
        providers["openrouter"] = client

    # Freebuff (opt-in via FREEBUFF_ENABLED=true)
    if settings.freebuff_enabled:
        try:
            freebuff = FreebuffClient(timeout=settings.freebuff_timeout)
            await freebuff.__aenter__()
            providers["freebuff"] = freebuff
        except ProviderError as e:
            logger.warning(f"Freebuff provider unavailable: {e}")

    # Custom providers from config
    for name, cfg in settings.custom_providers.items():
        if cfg.enabled and cfg.base_url:
            try:
                custom = OpenRouterClient(
                    api_key=cfg.api_key,
                    base_url=cfg.base_url,
                    timeout=cfg.timeout,
                )
                await custom.__aenter__()
                providers[name] = custom
            except Exception as e:
                logger.warning(f"Custom provider '{name}' unavailable: {e}")

    if not providers:
        raise ModelRoutingError(
            "No providers available. Set OPENROUTER_API_KEY or enable FREEBUFF_ENABLED."
        )

    try:
        yield ModelRouter(providers)
    finally:
        for provider in providers.values():
            await provider.__aexit__(None, None, None)