from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ProviderError(Exception):
    """Base error for all provider failures."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class BaseProvider(ABC):
    """Abstract interface every LLM provider must implement."""

    provider_name: str = "base"

    @abstractmethod
    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4000,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Send a chat completion request and return the response content."""

    async def __aenter__(self) -> BaseProvider:
        return self

    async def __aexit__(  # noqa: B027
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        pass
