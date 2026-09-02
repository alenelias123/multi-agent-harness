from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import BaseProvider, ProviderError

logger = logging.getLogger(__name__)

# Re-export for backwards compat
OpenRouterError = ProviderError


class OpenRouterClient(BaseProvider):
    provider_name = "openrouter"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 60.0,
    ):
        if not api_key:
            raise ValueError("OpenRouter API key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> OpenRouterClient:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/agentcli",
                "X-Title": "AgentCLI",
            },
            timeout=self.timeout,
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4000,
        response_format: dict | None = None,
    ) -> str:
        if not self._client:
            raise OpenRouterError("Client not initialized. Use async context manager.")

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            payload["response_format"] = response_format

        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            body = e.response.text
            if e.response.status_code == 429:
                raise OpenRouterError(
                    f"Rate limited: {body}",
                    status_code=429,
                    response_body=body,
                ) from e
            if 500 <= e.response.status_code < 600:
                raise OpenRouterError(
                    f"Server error: {body}",
                    status_code=e.response.status_code,
                    response_body=body,
                ) from e
            raise OpenRouterError(
                f"HTTP error: {body}",
                status_code=e.response.status_code,
                response_body=body,
            ) from e
        except httpx.RequestError as e:
            raise OpenRouterError(f"Request failed: {e}") from e

        choices = data.get("choices", [])
        if not choices:
            raise OpenRouterError(f"No choices in response: {data}")

        message: dict[str, Any] = choices[0].get("message", {})
        content: str = message.get("content", "")
        return content