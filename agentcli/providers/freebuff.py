from __future__ import annotations

import asyncio
import logging
import shutil
from typing import Any

from .base import BaseProvider, ProviderError

logger = logging.getLogger(__name__)

FreebuffError = ProviderError


class FreebuffClient(BaseProvider):
    """Thin async wrapper around the ``freebuff`` CLI.

    The CLI is interactive, so we construct a one-shot prompt that asks
    Freebuff to answer a question and return only the answer text.
    """

    provider_name = "freebuff"

    def __init__(self, timeout: float = 120.0) -> None:
        self.timeout = timeout
        freebuff_bin = shutil.which("freebuff")
        if not freebuff_bin:
            raise ProviderError(
                "freebuff CLI not found. Install with: npm install -g freebuff"
            )
        self._freebuff_bin: str = freebuff_bin

    async def __aenter__(self) -> FreebuffClient:
        return self

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4000,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Invoke the freebuff CLI and capture its output.

        We send the conversation as a structured prompt and ask Freebuff
        to reply with only the raw answer (no markdown fences, no prose).
        """
        prompt = self._build_prompt(messages)

        try:
            proc = await asyncio.create_subprocess_exec(
                self._freebuff_bin,
                "--prompt",
                prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
        except asyncio.TimeoutError as e:
            raise FreebuffError(
                f"Freebuff CLI timed out after {self.timeout}s"
            ) from e
        except FileNotFoundError as e:
            raise FreebuffError(
                f"freebuff binary not found: {e}"
            ) from e

        if proc.returncode != 0:
            error_msg = stderr.decode(errors="replace").strip()
            raise FreebuffError(
                f"freebuff CLI exited with code {proc.returncode}: {error_msg}"
            )

        output = stdout.decode(errors="replace").strip()
        if not output:
            raise FreebuffError("freebuff CLI returned empty output")

        return output

    @staticmethod
    def _build_prompt(messages: list[dict[str, str]]) -> str:
        """Convert OpenAI-style messages into a single prompt string."""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[System Instructions]\n{content}")
            elif role == "assistant":
                parts.append(f"[Assistant]\n{content}")
            else:
                parts.append(content)
        return "\n\n".join(parts)
