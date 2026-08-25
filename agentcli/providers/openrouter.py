from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class OpenRouterError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class OpenRouterClient:
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

    async def chat_completion_json(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4000,
    ) -> dict[str, Any]:
        response_format = {"type": "json_object"}
        content = await self.chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        try:
            result: dict[str, Any] = json.loads(content)
            return result
        except json.JSONDecodeError as e:
            raise OpenRouterError(
                f"Invalid JSON response: {e}, content: {content[:500]}"
            ) from e