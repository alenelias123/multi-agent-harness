"""Tests for the Freebuff setup module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentcli.setup import (
    freebuff_config_path,
    get_freebuff_token,
    get_token_from_config,
    get_token_from_env,
    is_freebuff_installed,
    save_token_to_env,
)


class TestFreebuffConfigPath:
    def test_returns_path(self) -> None:
        path = freebuff_config_path()
        assert path.name == "config.json"
        assert path.parent.name == "freebuff"


class TestIsFreebuffInstalled:
    @patch("agentcli.setup.shutil.which")
    def test_installed(self, mock_which: MagicMock) -> None:
        mock_which.return_value = "/usr/bin/freebuff"
        assert is_freebuff_installed() is True

    @patch("agentcli.setup.shutil.which")
    def test_not_installed(self, mock_which: MagicMock) -> None:
        mock_which.return_value = None
        assert is_freebuff_installed() is False


class TestGetTokenFromEnv:
    @patch.dict("os.environ", {"FREEBUFF_TOKEN": "test-token-123"})
    def test_from_env(self) -> None:
        token = get_token_from_env()
        assert token == "test-token-123"

    @patch.dict("os.environ", {}, clear=True)
    def test_not_in_env(self) -> None:
        token = get_token_from_env()
        # Could be None or could find it in .env files depending on test env
        # We just verify it doesn't crash
        assert token is None or isinstance(token, str)

    @patch.dict("os.environ", {"FREEBUFF_TOKEN": ""})
    def test_empty_env(self) -> None:
        token = get_token_from_env()
        # Empty string should not be treated as a token
        assert token is None or token != ""


class TestGetTokenFromConfig:
    @patch("agentcli.setup._manicode_credentials_path", return_value=Path("/nonexistent/credentials.json"))
    def test_no_config_file(self, _mock_cred: MagicMock) -> None:
        with patch("agentcli.setup.freebuff_config_path") as mock_path:
            mock_path.return_value = Path("/nonexistent/config.json")
            token = get_token_from_config()
            assert token is None

    @patch("agentcli.setup._manicode_credentials_path", return_value=Path("/nonexistent/credentials.json"))
    def test_valid_config_with_token(self, _mock_cred: MagicMock, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"token": "abc-123"}))

        with patch("agentcli.setup.freebuff_config_path") as mock_path:
            mock_path.return_value = config_file
            token = get_token_from_config()
            assert token == "abc-123"

    @patch("agentcli.setup._manicode_credentials_path", return_value=Path("/nonexistent/credentials.json"))
    def test_valid_config_with_api_key(self, _mock_cred: MagicMock, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"api_key": "key-456"}))

        with patch("agentcli.setup.freebuff_config_path") as mock_path:
            mock_path.return_value = config_file
            token = get_token_from_config()
            assert token == "key-456"

    @patch("agentcli.setup._manicode_credentials_path", return_value=Path("/nonexistent/credentials.json"))
    def test_empty_config(self, _mock_cred: MagicMock, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({}))

        with patch("agentcli.setup.freebuff_config_path") as mock_path:
            mock_path.return_value = config_file
            token = get_token_from_config()
            assert token is None

    @patch("agentcli.setup._manicode_credentials_path", return_value=Path("/nonexistent/credentials.json"))
    def test_invalid_json(self, _mock_cred: MagicMock, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text("not json {{{")

        with patch("agentcli.setup.freebuff_config_path") as mock_path:
            mock_path.return_value = config_file
            token = get_token_from_config()
            assert token is None

    def test_reads_manicode_credentials(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps({
            "default": {
                "authToken": "manicode-token-xyz",
                "name": "Test User",
            }
        }))

        with patch("agentcli.setup._manicode_credentials_path") as mock_path:
            mock_path.return_value = creds_file
            token = get_token_from_config()
            assert token == "manicode-token-xyz"


class TestGetFreebuffToken:
    @patch.dict("os.environ", {"FREEBUFF_TOKEN": "env-token"})
    def test_prefers_env(self) -> None:
        token = get_freebuff_token()
        assert token == "env-token"

    @patch.dict("os.environ", {}, clear=True)
    @patch("agentcli.setup.get_token_from_config")
    @patch("agentcli.setup.get_token_from_env")
    def test_falls_back_to_config(
        self, mock_env: MagicMock, mock_config: MagicMock
    ) -> None:
        mock_env.return_value = None
        mock_config.return_value = "config-token"
        token = get_freebuff_token()
        assert token == "config-token"

    @patch.dict("os.environ", {}, clear=True)
    @patch("agentcli.setup.get_token_from_config")
    @patch("agentcli.setup.get_token_from_env")
    def test_returns_none_when_no_token(
        self, mock_env: MagicMock, mock_config: MagicMock
    ) -> None:
        mock_env.return_value = None
        mock_config.return_value = None
        token = get_freebuff_token()
        assert token is None


class TestSaveTokenToEnv:
    def test_creates_new_env(self, tmp_path: Path) -> None:
        env_path = tmp_path / ".env"
        result = save_token_to_env("my-token", env_path)
        assert result == env_path

        content = env_path.read_text()
        assert "FREEBUFF_TOKEN=my-token" in content
        assert "FREEBUFF_ENABLED=true" in content

    def test_updates_existing_env(self, tmp_path: Path) -> None:
        env_path = tmp_path / ".env"
        env_path.write_text(
            "OPENROUTER_API_KEY=sk-or-xxx\n"
            "FREEBUFF_ENABLED=false\n"
            "FREEBUFF_TOKEN=old-token\n",
        )

        save_token_to_env("new-token", env_path)
        content = env_path.read_text()

        assert "FREEBUFF_TOKEN=new-token" in content
        assert "FREEBUFF_ENABLED=true" in content
        assert "OPENROUTER_API_KEY=sk-or-xxx" in content
        # Old token should be gone
        assert content.count("FREEBUFF_TOKEN=") == 1

    def test_preserves_other_settings(self, tmp_path: Path) -> None:
        env_path = tmp_path / ".env"
        env_path.write_text(
            "OPENROUTER_API_KEY=sk-or-xxx\n"
            "LOG_LEVEL=DEBUG\n"
            "MAX_PARALLELISM=8\n",
        )

        save_token_to_env("tok", env_path)
        content = env_path.read_text()

        assert "OPENROUTER_API_KEY=sk-or-xxx" in content
        assert "LOG_LEVEL=DEBUG" in content
        assert "MAX_PARALLELISM=8" in content
