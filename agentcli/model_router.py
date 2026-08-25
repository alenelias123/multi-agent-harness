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
from .providers.openrouter import OpenRouterClient, OpenRouterError

logger = logging.getLogger(__name__)


@dataclass
class ModelStats:
    failures: int = 0
    last_failure: float = 0
    consecutive_failures: int = 0


class ModelRouter:
    def __init__(self, client: OpenRouterClient):
        self.client = client
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

    async def call(
        self,
        task_type: str,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4000,
        response_format: dict | None = None,
        task_id: str | None = None,  # noqa: ARG002
    ) -> tuple[str, str]:
        models = self._get_available_models(task_type)
        if not models:
            models = self.get_model_chain(task_type)

        last_error: Exception | None = None

        for model in models:
            try:
                async for attempt in AsyncRetrying(
                    retry=retry_if_exception_type((OpenRouterError, httpx.HTTPStatusError)),
                    wait=wait_exponential_jitter(initial=1, max=10),
                    stop=stop_after_attempt(2),
                    reraise=True,
                ):
                    with attempt:
                        response = await self.client.chat_completion(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            response_format=response_format,
                        )
                        self._record_success(model)
                        return response, model
            except Exception as e:
                self._record_failure(model)
                last_error = e
                logger.warning(f"Model {model} failed for task_type={task_type}: {e}")
                continue

        raise ModelRoutingError(f"All models failed for task_type={task_type}: {last_error}")


class ModelRoutingError(Exception):
    pass


@asynccontextmanager
async def create_model_router() -> Any:
    client = OpenRouterClient(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )
    async with client:
        yield ModelRouter(client)