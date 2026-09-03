"""Multi-session manager using tmux for parallel AI chat sessions."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SESSION_PREFIX = "agentcli"
SESSION_DATA_DIR = Path.home() / ".config" / "agentcli" / "sessions"


@dataclass
class SessionInfo:
    """Metadata for a managed AI session."""

    session_id: str
    name: str
    status: str  # "running", "stopped"
    model: str | None
    created_at: float
    tmux_window: str


class TmuxError(Exception):
    """Raised when a tmux operation fails."""


def _run_tmux(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a tmux command and return the result."""
    cmd = ["tmux", *args]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if check and result.returncode != 0:
            raise TmuxError(f"tmux {' '.join(args)} failed: {result.stderr.strip()}")
        return result
    except FileNotFoundError:
        raise TmuxError(
            "tmux is not installed. Install it:\n"
            "  Ubuntu/Debian: sudo apt install tmux\n"
            "  macOS: brew install tmux\n"
            "  Arch: sudo pacman -S tmux"
        )
    except subprocess.TimeoutExpired:
        raise TmuxError(f"tmux command timed out: tmux {' '.join(args)}")


def is_tmux_available() -> bool:
    """Check if tmux is installed and the server is running."""
    try:
        result = _run_tmux("list-sessions", check=False)
        return result.returncode == 0
    except TmuxError:
        return False


def _ensure_data_dir() -> None:
    """Create the sessions data directory if it doesn't exist."""
    SESSION_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _session_meta_path(session_id: str) -> Path:
    """Path to the session metadata JSON file."""
    return SESSION_DATA_DIR / f"{session_id}.json"


def _save_session_meta(
    session_id: str,
    name: str,
    model: str | None,
    created_at: float,
    tmux_window: str,
) -> None:
    """Persist session metadata to disk."""
    _ensure_data_dir()
    meta = {
        "session_id": session_id,
        "name": name,
        "model": model,
        "created_at": created_at,
        "tmux_window": tmux_window,
    }
    _session_meta_path(session_id).write_text(json.dumps(meta), encoding="utf-8")


def _load_session_meta(session_id: str) -> dict[str, Any] | None:
    """Load session metadata from disk."""
    path = _session_meta_path(session_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _remove_session_meta(session_id: str) -> None:
    """Remove session metadata from disk."""
    path = _session_meta_path(session_id)
    if path.exists():
        path.unlink()


class SessionManager:
    """Manages multiple AI chat sessions running in tmux windows.

    Each session runs ``agentcli chat`` inside its own tmux window, so the
    user can switch between sessions with ``tmux`` key bindings or use the
    ``attach`` / ``send`` helpers to interact with them.

    Sessions are identified by a short ID and an optional human-readable name.
    """

    def __init__(self) -> None:
        self._ensure_tmux()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _ensure_tmux(self) -> None:
        """Make sure the tmux server is running."""
        if not is_tmux_available():
            # Start a detached tmux server
            _run_tmux("start-server", check=False)

    @staticmethod
    def _generate_id() -> str:
        """Generate a short random session ID."""
        import secrets

        return secrets.token_hex(4)

    def _tmux_session_name(self, session_id: str) -> str:
        """Full tmux session name including prefix."""
        return f"{SESSION_PREFIX}-{session_id}"

    def _list_active_tmux_windows(self) -> set[str]:
        """Return the set of agentcli tmux session names that exist."""
        result = _run_tmux(
            "list-sessions",
            "-F",
            "#{session_name}",
            check=False,
        )
        if result.returncode != 0:
            return set()
        names = set()
        for line in result.stdout.strip().splitlines():
            name = line.strip()
            if name.startswith(f"{SESSION_PREFIX}-"):
                names.add(name)
        return names

    def _send_keys(self, tmux_session: str, text: str) -> None:
        """Send literal text to a tmux pane."""
        # Use send-keys with -l (literal) to avoid special character interpretation
        _run_tmux("send-keys", "-t", tmux_session, "-l", text)
        _run_tmux("send-keys", "-t", tmux_session, "Enter")

    # ── public API ───────────────────────────────────────────────────────────

    def list_sessions(self) -> list[SessionInfo]:
        """List all tracked sessions and their current status."""
        sessions: list[SessionInfo] = []
        active_windows = self._list_active_tmux_windows()

        for meta_file in SESSION_DATA_DIR.glob("*.json"):
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                session_id = meta["session_id"]
                tmux_name = self._tmux_session_name(session_id)
                status = "running" if tmux_name in active_windows else "stopped"
                sessions.append(
                    SessionInfo(
                        session_id=session_id,
                        name=meta.get("name", session_id),
                        status=status,
                        model=meta.get("model"),
                        created_at=meta.get("created_at", 0),
                        tmux_window=tmux_name,
                    )
                )
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Skipping corrupt session file %s: %s", meta_file, exc)
                continue

        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions

    def create_session(
        self,
        name: str | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> SessionInfo:
        """Create a new AI session in a separate tmux window.

        The session runs ``agentcli chat`` (with optional ``--model``) inside a
        new tmux window.  The user can later attach to it, send input, or view
        its logs.

        Parameters
        ----------
        name:
            Human-readable name.  Falls back to the session ID.
        model:
            Optional model override passed to ``agentcli chat --model``.
        system_prompt:
            Optional path to a custom system prompt file.  If provided the
            ``--system-prompt`` flag is forwarded to ``agentcli chat``.

        Returns
        -------
        SessionInfo
            Metadata about the newly created session.
        """
        session_id = self._generate_id()
        display_name = name or session_id
        tmux_session_name = self._tmux_session_name(session_id)

        # Build the command to run inside tmux
        cmd_parts = ["agentcli", "chat"]
        if model:
            cmd_parts.extend(["--model", model])
        if system_prompt:
            cmd_parts.extend(["--system-prompt", system_prompt])
        chat_cmd = " ".join(cmd_parts)

        # Create a new detached tmux session
        _run_tmux(
            "new-session",
            "-d",
            "-s", tmux_session_name,
            "-n", display_name,
            chat_cmd,
        )

        # Give tmux a moment to start
        time.sleep(0.3)

        created_at = time.time()
        _save_session_meta(session_id, display_name, model, created_at, tmux_session_name)

        return SessionInfo(
            session_id=session_id,
            name=display_name,
            status="running",
            model=model,
            created_at=created_at,
            tmux_window=tmux_session_name,
        )

    def attach_session(self, session_id: str) -> None:
        """Attach the current terminal to a running session.

        This switches the user into the tmux session interactively.  The
        user can detach with ``Ctrl-B d`` to return to the main terminal.
        """
        tmux_name = self._tmux_session_name(session_id)
        active = self._list_active_tmux_windows()
        if tmux_name not in active:
            raise TmuxError(f"Session '{session_id}' is not running")

        # Use exec so we replace the current shell with the tmux client
        # When the user detaches, they return to the original shell.
        subprocess.run(
            ["tmux", "attach-session", "-t", tmux_name],
            check=False,
        )

    def send_message(self, session_id: str, message: str) -> None:
        """Send a message to a running session without attaching.

        The message is typed into the tmux pane as if the user had
        entered it in the chat REPL.  This enables parallel control.
        """
        tmux_name = self._tmux_session_name(session_id)
        active = self._list_active_tmux_windows()
        if tmux_name not in active:
            raise TmuxError(f"Session '{session_id}' is not running")

        # First, make sure the pane is ready for input by sending an empty
        # enter (in case the REPL is waiting), then the actual message.
        self._send_keys(tmux_name, message)

    def get_logs(self, session_id: str, lines: int = 50) -> str:
        """Capture the recent visible output from a session's tmux pane."""
        tmux_name = self._tmux_session_name(session_id)
        active = self._list_active_tmux_windows()
        if tmux_name not in active:
            # Try to get logs from the meta file or return a message
            meta = _load_session_meta(session_id)
            if not meta:
                raise TmuxError(f"Session '{session_id}' not found")
            return f"[Session '{session_id}' is stopped]"

        result = _run_tmux(
            "capture-pane",
            "-t", tmux_name,
            "-p",
            "-S", f"-{lines}",
            check=False,
        )
        if result.returncode != 0:
            return f"[Could not capture output from session '{session_id}']"
        return result.stdout

    def kill_session(self, session_id: str) -> None:
        """Terminate a running session and remove its metadata."""
        tmux_name = self._tmux_session_name(session_id)
        active = self._list_active_tmux_windows()

        if tmux_name in active:
            _run_tmux("kill-session", "-t", tmux_name, check=False)

        _remove_session_meta(session_id)

    def kill_all_sessions(self) -> int:
        """Kill all tracked agentcli sessions.  Returns count killed."""
        count = 0
        for session in self.list_sessions():
            self.kill_session(session.session_id)
            count += 1
        return count

    def session_exists(self, session_id: str) -> bool:
        """Check whether a session ID is tracked."""
        return _load_session_meta(session_id) is not None

    def get_session(self, session_id: str) -> SessionInfo | None:
        """Get metadata for a single session."""
        meta = _load_session_meta(session_id)
        if meta is None:
            return None

        tmux_name = self._tmux_session_name(session_id)
        active = self._list_active_tmux_windows()
        status = "running" if tmux_name in active else "stopped"

        return SessionInfo(
            session_id=meta["session_id"],
            name=meta.get("name", session_id),
            status=status,
            model=meta.get("model"),
            created_at=meta.get("created_at", 0),
            tmux_window=tmux_name,
        )
