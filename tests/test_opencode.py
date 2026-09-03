"""Tests for the OpenCode provider."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentcli.providers.opencode import (
    OPENCODE_GO_BASE_URL,
    OPENCODE_ZEN_BASE_URL,
    OpenCodeClient,
    OpenCodeError,
)


class TestOpenCodeClient:
    def test_init_requires_api_key(self) -> None:
        with pytest.raises(ValueError, match="API key is required"):
            OpenCodeClient(api_key="")

    def test_init_default_base_url(self) -> None:
        client = OpenCodeClient(api_key="test-key")
        assert client.base_url == OPENCODE_ZEN_BASE_URL

    def test_init_custom_base_url(self) -> None:
        client = OpenCodeClient(api_key="test-key", base_url="https://custom.api/v1")
        assert client.base_url == "https://custom.api/v1"

    def test_init_strips_trailing_slash(self) -> None:
        client = OpenCodeClient(api_key="test-key", base_url="https://api.example.com/v1/")
        assert client.base_url == "https://api.example.com/v1"

    def test_provider_name(self) -> None:
        client = OpenCodeClient(api_key="test-key")
        assert client.provider_name == "opencode"

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        client = OpenCodeClient(api_key="test-key")
        async with client:
            assert client._client is not None
        assert client._client is None

    @pytest.mark.asyncio
    async def test_chat_completion_success(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from OpenCode!"}}]
        }

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        client = OpenCodeClient(api_key="test-key")
        client._client = mock_client

        result = await client.chat_completion(
            model="anthropic/claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert result == "Hello from OpenCode!"
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/chat/completions"
        payload = call_args[1]["json"]
        assert payload["model"] == "anthropic/claude-sonnet-4-5"

    @pytest.mark.asyncio
    async def test_chat_completion_no_choices(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"choices": []}

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        client = OpenCodeClient(api_key="test-key")
        client._client = mock_client

        with pytest.raises(OpenCodeError, match="No choices"):
            await client.chat_completion(
                model="test-model",
                messages=[{"role": "user", "content": "Hi"}],
            )

    @pytest.mark.asyncio
    async def test_chat_completion_not_initialized(self) -> None:
        client = OpenCodeClient(api_key="test-key")
        with pytest.raises(OpenCodeError, match="not initialized"):
            await client.chat_completion(
                model="test-model",
                messages=[{"role": "user", "content": "Hi"}],
            )


class TestOpenCodeBaseUrls:
    def test_zen_url(self) -> None:
        assert OPENCODE_ZEN_BASE_URL == "https://opencode.ai/zen/v1"

    def test_go_url(self) -> None:
        assert OPENCODE_GO_BASE_URL == "https://opencode.ai/zen/go/v1"

    def test_urls_are_different(self) -> None:
        assert OPENCODE_ZEN_BASE_URL != OPENCODE_GO_BASE_URL


class TestOpenCodeConfig:
    def test_config_has_opencode_fields(self) -> None:
        from agentcli.config import Settings

        # Check that the Settings model has opencode fields
        fields = Settings.model_fields
        assert "opencode_api_key" in fields
        assert "opencode_base_url" in fields
        assert "opencode_tier" in fields

    def test_default_tier_is_zen(self) -> None:
        from agentcli.config import Settings

        s = Settings()
        assert s.opencode_tier == "zen"

    def test_task_type_models_include_opencode(self) -> None:
        from agentcli.config import Settings

        s = Settings()
        for task_type, models in s.task_type_models.items():
            opencode_models = [m for m in models if m.startswith("opencode/")]
            assert len(opencode_models) > 0, (
                f"Task type '{task_type}' has no opencode models"
            )


class TestModelRouterOpenCode:
    def test_parse_model_ref_opencode(self) -> None:
        from agentcli.model_router import _parse_model_ref

        provider, model = _parse_model_ref("opencode/anthropic/claude-sonnet-4-5")
        assert provider == "opencode"
        assert model == "anthropic/claude-sonnet-4-5"

    def test_parse_model_ref_opencode_simple(self) -> None:
        from agentcli.model_router import _parse_model_ref

        provider, model = _parse_model_ref("opencode/default")
        assert provider == "opencode"
        assert model == "default"
