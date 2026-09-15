"""Tests for the Freebuff and OpenCode setup module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from agentcli.setup import (
    freebuff_config_path,
    get_freebuff_token,
    get_token_from_config,
    get_token_from_env,
    is_freebuff_installed,
    save_opencode_config,
    save_token_to_env,
    setup_all,
    setup_provider,
    verify_opencode_key,
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
    @patch(
    "agentcli.setup._manicode_credentials_path",
    return_value=Path("/nonexistent/credentials.json"),
)
    def test_no_config_file(self, _mock_cred: MagicMock) -> None:
        with patch("agentcli.setup.freebuff_config_path") as mock_path:
            mock_path.return_value = Path("/nonexistent/config.json")
            token = get_token_from_config()
            assert token is None

    @patch(
    "agentcli.setup._manicode_credentials_path",
    return_value=Path("/nonexistent/credentials.json"),
)
    def test_valid_config_with_token(self, _mock_cred: MagicMock, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"token": "abc-123"}))

        with patch("agentcli.setup.freebuff_config_path") as mock_path:
            mock_path.return_value = config_file
            token = get_token_from_config()
            assert token == "abc-123"

    @patch(
    "agentcli.setup._manicode_credentials_path",
    return_value=Path("/nonexistent/credentials.json"),
)
    def test_valid_config_with_api_key(self, _mock_cred: MagicMock, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"api_key": "key-456"}))

        with patch("agentcli.setup.freebuff_config_path") as mock_path:
            mock_path.return_value = config_file
            token = get_token_from_config()
            assert token == "key-456"

    @patch(
    "agentcli.setup._manicode_credentials_path",
    return_value=Path("/nonexistent/credentials.json"),
)
    def test_empty_config(self, _mock_cred: MagicMock, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({}))

        with patch("agentcli.setup.freebuff_config_path") as mock_path:
            mock_path.return_value = config_file
            token = get_token_from_config()
            assert token is None

    @patch(
    "agentcli.setup._manicode_credentials_path",
    return_value=Path("/nonexistent/credentials.json"),
)
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


class TestOpenCodeConfig:
    @patch("agentcli.setup._find_opencode_env_path")
    def test_save_opencode_config(self, mock_find_env: MagicMock, tmp_path: Path) -> None:
        env_path = tmp_path / ".env"
        mock_find_env.return_value = env_path

        result = save_opencode_config("test-key-123", "zen")

        assert result == env_path
        content = env_path.read_text()
        assert "OPENCODE_API_KEY=test-key-123" in content
        assert "OPENCODE_TIER=zen" in content
        assert "OPENCODE_BASE_URL=https://opencode.ai/zen/v1" in content

    @patch("agentcli.setup._find_opencode_env_path")
    def test_save_opencode_config_go_tier(
        self, mock_find_env: MagicMock, tmp_path: Path
    ) -> None:
        env_path = tmp_path / ".env"
        mock_find_env.return_value = env_path

        save_opencode_config("test-key-456", "go")

        content = env_path.read_text()
        assert "OPENCODE_TIER=go" in content
        assert "OPENCODE_BASE_URL=https://opencode.ai/zen/go/v1" in content

    @patch("agentcli.setup._find_opencode_env_path")
    def test_save_opencode_config_updates_existing(
        self, mock_find_env: MagicMock, tmp_path: Path
    ) -> None:
        env_path = tmp_path / ".env"
        env_path.write_text(
            "OPENCODE_API_KEY=old-key\n"
            "OPENCODE_TIER=go\n"
            "OPENCODE_BASE_URL=https://opencode.ai/zen/go/v1\n"
            "OTHER_SETTING=value\n",
        )
        mock_find_env.return_value = env_path

        save_opencode_config("new-key-789", "zen")

        content = env_path.read_text()
        assert "OPENCODE_API_KEY=new-key-789" in content
        assert "OPENCODE_TIER=zen" in content
        assert "OPENCODE_BASE_URL=https://opencode.ai/zen/v1" in content
        assert "OTHER_SETTING=value" in content


class TestVerifyOpenCodeKey:
    @patch("httpx.Client")
    def test_verify_valid_key(self, mock_client_class: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "model1"}, {"id": "model2"}]}
        mock_client.get.return_value = mock_resp

        valid, msg = verify_opencode_key("valid-key", "zen")

        assert valid is True
        assert "2 models available" in msg

    @patch("httpx.Client")
    def test_verify_invalid_key_401(self, mock_client_class: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_client.get.return_value = mock_resp

        valid, msg = verify_opencode_key("invalid-key", "zen")

        assert valid is False
        assert "401" in msg

    @patch("httpx.Client")
    def test_verify_network_error(self, mock_client_class: MagicMock) -> None:
        mock_client_class.return_value.__enter__.side_effect = Exception("Connection failed")

        valid, msg = verify_opencode_key("test-key", "zen")

        assert valid is False
        assert "Verification failed" in msg


class TestSetupProvider:
    @patch("agentcli.setup.is_freebuff_installed", return_value=True)
    @patch("agentcli.setup.get_freebuff_version", return_value="1.0.0")
    @patch("agentcli.setup.get_freebuff_token", return_value="freebuff-token")
    @patch("agentcli.setup.get_token_from_env", return_value="freebuff-token")
    def test_setup_freebuff_already_configured(
        self, _mock_env: MagicMock, _mock_token: MagicMock,
        _mock_version: MagicMock, _mock_installed: MagicMock
    ) -> None:
        result = setup_provider("freebuff")

        assert result["provider"] == "freebuff"
        assert result["installed"] is True
        assert result["authenticated"] is True
        assert result["version"] == "1.0.0"
        assert result["token_source"] == "env"

    @patch("agentcli.setup.is_freebuff_installed", return_value=True)
    @patch("agentcli.setup.get_freebuff_version", return_value="1.0.0")
    @patch("agentcli.setup.get_freebuff_token", side_effect=[None, "new-token"])
    @patch("agentcli.setup.run_auth_flow", return_value=True)
    @patch("agentcli.setup.get_token_from_env", return_value="new-token")
    def test_setup_freebuff_with_auth(
        self, _mock_env2: MagicMock, _mock_auth: MagicMock,
        _mock_token: MagicMock, _mock_version: MagicMock, _mock_installed: MagicMock
    ) -> None:
        result = setup_provider("freebuff", auth=True)

        assert result["provider"] == "freebuff"
        assert result["installed"] is True
        assert result["authenticated"] is True
        assert result["token_source"] == "env"

    def test_setup_opencode_missing_key(self) -> None:
        result = setup_provider("opencode")

        assert result["provider"] == "opencode"
        assert result["error"] == "OpenCode API key is required"

    def test_setup_opencode_invalid_tier(self) -> None:
        result = setup_provider("opencode", api_key="test", tier="invalid")

        assert result["provider"] == "opencode"
        assert result["error"] == "Invalid tier. Use 'zen' or 'go'"

    @patch("agentcli.setup.verify_opencode_key", return_value=(True, "Valid"))
    @patch("agentcli.setup.save_opencode_config")
    def test_setup_opencode_success(
        self,
        mock_save: MagicMock,
        _mock_verify: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_save.return_value = tmp_path / ".env"

        result = setup_provider("opencode", api_key="test-key", tier="zen")

        assert result["provider"] == "opencode"
        assert result["installed"] is True
        assert result["authenticated"] is True
        assert result["version"] == "zen"
        assert result["token_source"] == "env"

    @patch("agentcli.setup.verify_opencode_key", return_value=(False, "Invalid key"))
    def test_setup_opencode_verification_fails(
        self, mock_verify: MagicMock  # noqa: ARG002 — result uses return_value
    ) -> None:
        result = setup_provider("opencode", api_key="bad-key", tier="zen")

        assert result["provider"] == "opencode"
        assert "verification failed" in result["error"].lower()

    def test_setup_unknown_provider(self) -> None:
        result = setup_provider("unknown")

        assert result["error"] == "Unknown provider: unknown"


class TestSetupAll:
    @patch("agentcli.setup.setup_provider")
    def test_setup_all_both_providers(self, mock_setup: MagicMock) -> None:
        mock_setup.side_effect = [
            {"provider": "opencode", "installed": True, "authenticated": True},
            {"provider": "freebuff", "installed": True, "authenticated": False},
        ]

        results = setup_all(opencode_api_key="test-key")

        assert "opencode" in results
        assert "freebuff" in results
        assert mock_setup.call_count == 2

    @patch("agentcli.setup.setup_provider")
    @patch("agentcli.config.get_settings")
    def test_setup_all_existing_opencode(
        self, mock_settings: MagicMock, mock_setup: MagicMock
    ) -> None:
        mock_settings.return_value.opencode_api_key = "existing-key"
        mock_settings.return_value.opencode_tier = "zen"
        mock_setup.return_value = {"provider": "freebuff", "installed": True}

        results = setup_all()

        assert "opencode" in results
        assert results["opencode"]["installed"] is True
        assert results["opencode"]["authenticated"] is True
        assert mock_setup.call_count == 1  # Only freebuff called
