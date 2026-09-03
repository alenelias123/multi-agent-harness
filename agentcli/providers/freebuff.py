from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .base import BaseProvider, ProviderError

logger = logging.getLogger(__name__)

FreebuffError = ProviderError


def _find_freebuff_config_token() -> str | None:
    """Read the auth token from freebuff's config.json.

    Checks multiple possible key names and platform paths.
    """
    # Determine config directory per platform
    home = Path.home()
    if os.environ.get("XDG_CONFIG_HOME"):
        config_dir = Path(os.environ["XDG_CONFIG_HOME"]) / "freebuff"
    elif home / ".config" / "freebuff" .is_dir():
        config_dir = home / ".config" / "freebuff"
    elif home / "Library" / "Application Support" / "freebuff" .is_dir():
        config_dir = home / "Library" / "Application Support" / "freebuff"
    elif home / ".freebuff" .is_dir():
        config_dir = home / ".freebuff"
    else:
        config_dir = home / ".config" / "freebuff"

    config_file = config_dir / "config.json"
    if not config_file.is_file():
        return None

    try:
        with open(config_file, encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.debug(f"Failed to read freebuff config: {e}")
        return None

    for key in ("token", "api_key", "auth_token", "apiKey", "accessToken"):
        val = cfg.get(key)
        if val and isinstance(val, str) and len(val) > 0:
            return val
    return None


def _get_freebuff_token() -> str | None:
    """Get the freebuff token from env var or config file."""
    # 1. Environment variable
    token = os.environ.get("FREEBUFF_TOKEN")
    if token:
        return token

    # 2. .env files
    env_locations = [
        Path(".env"),
        home / ".config" / "agentcli" / ".env" if (home := Path.home()) else None,
    ]
    for env_path in env_locations:
        if env_path and env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("FREEBUFF_TOKEN=") and not stripped.startswith("#"):
                    val = stripped.split("=", 1)[1].strip().strip("'\"")
                    if val:
                        return val

    # 3. freebuff config.json
    return _find_freebuff_config_token()


class FreebuffClient(BaseProvider):
    """Thin async wrapper around the ``freebuff`` CLI.

    The CLI is interactive, so we construct a one-shot prompt that asks
    Freebuff to answer a question and return only the answer text.

    Token detection priority:
      1. FREEBUFF_TOKEN env var
      2. .env file FREEBUFF_TOKEN=...
      3. freebuff config.json (auto-detected from platform paths)
    """

    provider_name = "freebuff"

    def __init__(self, timeout: float = 120.0, auto_install: bool = False) -> None:
        self.timeout = timeout
        self._token: str | None = None
        freebuff_bin = shutil.which("freebuff")

        if not freebuff_bin:
            if auto_install:
                self._try_auto_install()
                freebuff_bin = shutil.which("freebuff")

            if not freebuff_bin:
                raise ProviderError(
                    "freebuff CLI not found. Install with:\n"
                    "  npm install -g freebuff\n"
                    "Or run: python -m agentcli setup"
                )

        self._freebuff_bin: str = freebuff_bin

        # Detect and store the auth token
        self._token = _get_freebuff_token()
        if self._token:
            logger.info("Freebuff auth token detected from config/env")
        else:
            logger.warning(
                "No freebuff auth token found. "
                "Set FREEBUFF_TOKEN or run: freebuff auth"
            )

    @staticmethod
    def _try_auto_install() -> None:
        """Attempt to install freebuff via npm."""
        if not shutil.which("npm"):
            logger.warning("npm not available for auto-install")
            return

        logger.info("Attempting auto-install of freebuff CLI...")
        try:
            result = subprocess.run(
                ["npm", "install", "-g", "freebuff"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                logger.info("freebuff installed successfully via auto-install")
            else:
                logger.warning(f"Auto-install failed: {result.stderr[:200]}")
        except Exception as e:
            logger.warning(f"Auto-install error: {e}")

    @property
    def has_token(self) -> bool:
        """Whether an auth token was detected."""
        return self._token is not None

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

        # Build environment with token injection
        env = os.environ.copy()
        if self._token:
            env["FREEBUFF_TOKEN"] = self._token

        try:
            proc = await asyncio.create_subprocess_exec(
                self._freebuff_bin,
                "--prompt",
                prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
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
