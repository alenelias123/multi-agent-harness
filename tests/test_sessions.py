"""Tests for the multi-session manager."""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentcli.sessions import (
    SESSION_DATA_DIR,
    SessionInfo,
    SessionManager,
    TmuxError,
    _load_session_meta,
    _remove_session_meta,
    _save_session_meta,
    is_tmux_available,
)


class TestTmuxAvailability:
    @patch("agentcli.sessions.subprocess.run")
    def test_is_tmux_available_true(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        assert is_tmux_available() is True

    @patch("agentcli.sessions.subprocess.run")
    def test_is_tmux_available_false(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="no server")
        assert is_tmux_available() is False

    @patch("agentcli.sessions.subprocess.run", side_effect=FileNotFoundError)
    def test_is_tmux_available_not_installed(self, mock_run: MagicMock) -> None:
        assert is_tmux_available() is False


class TestSessionMetaPersistence:
    def setup_method(self) -> None:
        """Use a temp directory for session data in tests."""
        self._patcher = patch("agentcli.sessions.SESSION_DATA_DIR")
        self.mock_dir = self._patcher.start()
        self.tmpdir = Path(tempfile.mkdtemp())
        self.mock_dir.__truediv__ = lambda self_, x: self.tmpdir / x
        # Patch glob to work with the mock
        self.mock_dir.glob = lambda pattern: self.tmpdir.glob(pattern)

    def teardown_method(self) -> None:
        self._patcher.stop()
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load_meta(self) -> None:
        _save_session_meta("abc123", "my-session", "gpt-4", 1234567890.0, "agentcli-abc123")
        meta = _load_session_meta("abc123")
        assert meta is not None
        assert meta["session_id"] == "abc123"
        assert meta["name"] == "my-session"
        assert meta["model"] == "gpt-4"
        assert meta["created_at"] == 1234567890.0
        assert meta["tmux_window"] == "agentcli-abc123"

    def test_load_nonexistent_returns_none(self) -> None:
        assert _load_session_meta("nonexistent") is None

    def test_remove_meta(self) -> None:
        _save_session_meta("abc123", "s", None, 0.0, "w")
        _remove_session_meta("abc123")
        assert _load_session_meta("abc123") is None

    def test_remove_nonexistent_is_noop(self) -> None:
        # Should not raise
        _remove_session_meta("nonexistent")


class TestSessionManager:
    def setup_method(self) -> None:
        self._patcher = patch("agentcli.sessions.SESSION_DATA_DIR")
        self.mock_dir = self._patcher.start()
        self.tmpdir = Path(tempfile.mkdtemp())
        self.mock_dir.__truediv__ = lambda self_, x: self.tmpdir / x
        self.mock_dir.glob = lambda pattern: self.tmpdir.glob(pattern)
        self.mock_dir.mkdir = lambda *a, **kw: self.tmpdir.mkdir(*a, **kw)

    def teardown_method(self) -> None:
        self._patcher.stop()
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("agentcli.sessions._run_tmux")
    @patch("agentcli.sessions.is_tmux_available", return_value=True)
    def test_create_session(self, _mock_avail: MagicMock, mock_tmux: MagicMock) -> None:
        mock_tmux.return_value = MagicMock(returncode=0, stdout="", stderr="")
        manager = SessionManager()

        session = manager.create_session(name="test-session", model="gpt-4")
        assert session.name == "test-session"
        assert session.model == "gpt-4"
        assert session.status == "running"
        assert session.session_id
        assert session.tmux_window.startswith("agentcli-")

        # Verify tmux was called to create a new session
        calls = [str(c) for c in mock_tmux.call_args_list]
        assert any("new-session" in c for c in calls)

    @patch("agentcli.sessions._run_tmux")
    @patch("agentcli.sessions.is_tmux_available", return_value=True)
    def test_create_session_default_name(
        self, _mock_avail: MagicMock, mock_tmux: MagicMock
    ) -> None:
        mock_tmux.return_value = MagicMock(returncode=0, stdout="", stderr="")
        manager = SessionManager()

        session = manager.create_session()
        # Name should fall back to session_id
        assert session.name == session.session_id

    @patch("agentcli.sessions._run_tmux")
    @patch("agentcli.sessions.is_tmux_available", return_value=True)
    def test_list_sessions_empty(
        self, _mock_avail: MagicMock, mock_tmux: MagicMock
    ) -> None:
        mock_tmux.return_value = MagicMock(returncode=1, stdout="", stderr="")
        manager = SessionManager()
        assert manager.list_sessions() == []

    @patch("agentcli.sessions._run_tmux")
    @patch("agentcli.sessions.is_tmux_available", return_value=True)
    def test_list_sessions_with_data(
        self, _mock_avail: MagicMock, mock_tmux: MagicMock
    ) -> None:
        # First call: list-sessions returns our tmux session as active
        def side_effect(*args: str, **kwargs: object) -> MagicMock:
            if "list-sessions" in args:
                return MagicMock(returncode=0, stdout="agentcli-abc123\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_tmux.side_effect = side_effect

        # Save a session meta manually
        _save_session_meta("abc123", "test", "model-a", time.time(), "agentcli-abc123")

        manager = SessionManager()
        sessions = manager.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].session_id == "abc123"
        assert sessions[0].status == "running"

    @patch("agentcli.sessions._run_tmux")
    @patch("agentcli.sessions.is_tmux_available", return_value=True)
    def test_get_session(self, _mock_avail: MagicMock, mock_tmux: MagicMock) -> None:
        def side_effect(*args: str, **kwargs: object) -> MagicMock:
            if "list-sessions" in args:
                return MagicMock(returncode=0, stdout="agentcli-abc123\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_tmux.side_effect = side_effect

        _save_session_meta("abc123", "test", "model-a", time.time(), "agentcli-abc123")

        manager = SessionManager()
        session = manager.get_session("abc123")
        assert session is not None
        assert session.name == "test"

    @patch("agentcli.sessions._run_tmux")
    @patch("agentcli.sessions.is_tmux_available", return_value=True)
    def test_get_session_nonexistent(
        self, _mock_avail: MagicMock, mock_tmux: MagicMock
    ) -> None:
        mock_tmux.return_value = MagicMock(returncode=1, stdout="", stderr="")
        manager = SessionManager()
        assert manager.get_session("nonexistent") is None

    @patch("agentcli.sessions._run_tmux")
    @patch("agentcli.sessions.is_tmux_available", return_value=True)
    def test_kill_session(self, _mock_avail: MagicMock, mock_tmux: MagicMock) -> None:
        def side_effect(*args: str, **kwargs: object) -> MagicMock:
            if "list-sessions" in args:
                return MagicMock(returncode=0, stdout="agentcli-abc123\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_tmux.side_effect = side_effect

        _save_session_meta("abc123", "test", None, time.time(), "agentcli-abc123")
        manager = SessionManager()
        manager.kill_session("abc123")

        # Meta should be removed
        assert _load_session_meta("abc123") is None
        # tmux kill-session should have been called
        calls = [str(c) for c in mock_tmux.call_args_list]
        assert any("kill-session" in c for c in calls)

    @patch("agentcli.sessions._run_tmux")
    @patch("agentcli.sessions.is_tmux_available", return_value=True)
    def test_kill_all_sessions(
        self, _mock_avail: MagicMock, mock_tmux: MagicMock
    ) -> None:
        def side_effect(*args: str, **kwargs: object) -> MagicMock:
            if "list-sessions" in args:
                return MagicMock(
                    returncode=0, stdout="agentcli-aaa\nagentcli-bbb\n", stderr=""
                )
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_tmux.side_effect = side_effect

        _save_session_meta("aaa", "s1", None, 0.0, "agentcli-aaa")
        _save_session_meta("bbb", "s2", None, 0.0, "agentcli-bbb")

        manager = SessionManager()
        count = manager.kill_all_sessions()
        assert count == 2
        assert _load_session_meta("aaa") is None
        assert _load_session_meta("bbb") is None

    @patch("agentcli.sessions._run_tmux")
    @patch("agentcli.sessions.is_tmux_available", return_value=True)
    def test_send_message(self, _mock_avail: MagicMock, mock_tmux: MagicMock) -> None:
        def side_effect(*args: str, **kwargs: object) -> MagicMock:
            if "list-sessions" in args:
                return MagicMock(returncode=0, stdout="agentcli-abc123\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_tmux.side_effect = side_effect

        _save_session_meta("abc123", "test", None, time.time(), "agentcli-abc123")
        manager = SessionManager()
        manager.send_message("abc123", "Hello from test!")

        calls = [str(c) for c in mock_tmux.call_args_list]
        assert any("send-keys" in c and "Hello from test!" in c for c in calls)

    @patch("agentcli.sessions._run_tmux")
    @patch("agentcli.sessions.is_tmux_available", return_value=True)
    def test_send_message_not_running(
        self, _mock_avail: MagicMock, mock_tmux: MagicMock
    ) -> None:
        mock_tmux.return_value = MagicMock(returncode=1, stdout="", stderr="")
        manager = SessionManager()

        with pytest.raises(TmuxError, match="not running"):
            manager.send_message("nonexistent", "test")

    @patch("agentcli.sessions._run_tmux")
    @patch("agentcli.sessions.is_tmux_available", return_value=True)
    def test_get_logs(self, _mock_avail: MagicMock, mock_tmux: MagicMock) -> None:
        def side_effect(*args: str, **kwargs: object) -> MagicMock:
            if "list-sessions" in args:
                return MagicMock(returncode=0, stdout="agentcli-abc123\n", stderr="")
            if "capture-pane" in args:
                return MagicMock(returncode=0, stdout="line1\nline2\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_tmux.side_effect = side_effect

        _save_session_meta("abc123", "test", None, time.time(), "agentcli-abc123")
        manager = SessionManager()
        logs = manager.get_logs("abc123")
        assert "line1" in logs

    @patch("agentcli.sessions._run_tmux")
    @patch("agentcli.sessions.is_tmux_available", return_value=True)
    def test_get_logs_stopped_session(
        self, _mock_avail: MagicMock, mock_tmux: MagicMock
    ) -> None:
        mock_tmux.return_value = MagicMock(returncode=1, stdout="", stderr="")

        _save_session_meta("abc123", "test", None, time.time(), "agentcli-abc123")
        manager = SessionManager()
        logs = manager.get_logs("abc123")
        assert "stopped" in logs

    @patch("agentcli.sessions._run_tmux")
    @patch("agentcli.sessions.is_tmux_available", return_value=True)
    def test_session_exists(self, _mock_avail: MagicMock, mock_tmux: MagicMock) -> None:
        mock_tmux.return_value = MagicMock(returncode=0, stdout="", stderr="")
        manager = SessionManager()

        assert manager.session_exists("nonexistent") is False
        _save_session_meta("abc123", "test", None, 0.0, "w")
        assert manager.session_exists("abc123") is True


class TestSessionInfo:
    def test_creation(self) -> None:
        info = SessionInfo(
            session_id="abc123",
            name="test",
            status="running",
            model="gpt-4",
            created_at=1234567890.0,
            tmux_window="agentcli-abc123",
        )
        assert info.session_id == "abc123"
        assert info.name == "test"
        assert info.status == "running"
        assert info.model == "gpt-4"
        assert info.tmux_window == "agentcli-abc123"
