"""Setup utilities for Freebuff CLI — install, auth, and token management.

The freebuff CLI stores its auth token in a JSON config file at:
  - ~/.config/freebuff/config.json   (Linux)
  - ~/Library/Application Support/freebuff/config.json  (macOS)
  - %APPDATA%/freebuff/config.json   (Windows)

This module detects and manages that token, bridging it into agentcli's
.env-based configuration.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


# ── Paths ────────────────────────────────────────────────────────────────

def _freebuff_config_dir() -> Path:
    """Return the platform-appropriate freebuff config directory."""
    # Check platformdirs first (if available)
    try:
        from platformdirs import user_config_dir
        return Path(user_config_dir("freebuff"))
    except ImportError:
        pass

    # Fallback to manual detection
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "freebuff"
    elif sys.platform == "win32":
        appdata = Path.home() / "AppData" / "Roaming"
        return appdata / "freebuff"
    else:
        # Linux / other
        xdg = Path.home() / ".config"
        return xdg / "freebuff"


def freebuff_config_path() -> Path:
    """Return the path to freebuff's config.json."""
    return _freebuff_config_dir() / "config.json"


# ── Detection ────────────────────────────────────────────────────────────

def is_freebuff_installed() -> bool:
    """Check if the freebuff CLI binary is available on PATH."""
    return shutil.which("freebuff") is not None


def get_freebuff_version() -> str | None:
    """Return the freebuff CLI version string, or None if not installed."""
    freebuff_bin = shutil.which("freebuff")
    if not freebuff_bin:
        return None
    try:
        result = subprocess.run(
            [freebuff_bin, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


# ── Token Management ─────────────────────────────────────────────────────

def get_token_from_config() -> str | None:
    """Read the auth token from freebuff's config.json file.

    Checks multiple possible key names: 'token', 'api_key', 'auth_token'.
    """
    config_file = freebuff_config_path()
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


def get_token_from_env() -> str | None:
    """Read the FREEBUFF_TOKEN from the environment or .env file."""
    import os
    # Check environment first
    token = os.environ.get("FREEBUFF_TOKEN")
    if token:
        return token

    # Check .env files
    env_locations = [
        Path(".env"),
        Path.home() / ".config" / "agentcli" / ".env",
    ]
    for env_path in env_locations:
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("FREEBUFF_TOKEN=") and not line.startswith("#"):
                    val = line.split("=", 1)[1].strip().strip("'\"")
                    if val:
                        return val
    return None


def get_freebuff_token() -> str | None:
    """Get the freebuff auth token from any available source.

    Priority:
      1. FREEBUFF_TOKEN environment variable
      2. freebuff config.json file
    """
    token = get_token_from_env()
    if token:
        return token
    return get_token_from_config()


def save_token_to_env(token: str, env_path: Path | None = None) -> Path:
    """Save the freebuff token to the agentcli .env file.

    Creates or updates the .env file with FREEBUFF_ENABLED=true and
    FREEBUFF_TOKEN=<token>.
    """
    if env_path is None:
        env_path = Path(".env")

    lines: list[str] = []
    token_written = False
    enabled_written = False

    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("FREEBUFF_TOKEN="):
                lines.append(f"FREEBUFF_TOKEN={token}")
                token_written = True
            elif stripped.startswith("FREEBUFF_ENABLED="):
                lines.append("FREEBUFF_ENABLED=true")
                enabled_written = True
            else:
                lines.append(line)

    if not token_written:
        lines.append(f"FREEBUFF_TOKEN={token}")
    if not enabled_written:
        lines.append("FREEBUFF_ENABLED=true")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env_path


# ── Installation ─────────────────────────────────────────────────────────

def install_freebuff() -> bool:
    """Install the freebuff CLI globally via npm.

    Returns True on success, False on failure.
    """
    if is_freebuff_installed():
        console.print("[green]✓ freebuff is already installed[/green]")
        return True

    # Check for npm
    if not shutil.which("npm"):
        console.print("[red]✗ npm is not installed. Please install Node.js first.[/red]")
        console.print(
            "  [cyan]curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -[/cyan]\n"
            "  [cyan]sudo apt-get install -y nodejs[/cyan]"
        )
        return False

    console.print("[yellow]→ Installing freebuff CLI globally via npm...[/yellow]")
    try:
        result = subprocess.run(
            ["npm", "install", "-g", "freebuff"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            version = get_freebuff_version() or "installed"
            console.print(f"[green]✓ freebuff {version} installed successfully[/green]")
            return True
        else:
            console.print(f"[red]✗ npm install failed:[/red]")
            if result.stderr:
                console.print(f"  {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        console.print("[red]✗ npm install timed out[/red]")
        return False
    except Exception as e:
        console.print(f"[red]✗ Installation failed: {e}[/red]")
        return False


def run_auth_flow() -> bool:
    """Run the freebuff auth flow interactively.

    Returns True if authentication succeeds.
    """
    freebuff_bin = shutil.which("freebuff")
    if not freebuff_bin:
        console.print("[red]✗ freebuff CLI not found. Install it first.[/red]")
        return False

    console.print("[cyan]→ Starting freebuff authentication...[/cyan]")
    try:
        result = subprocess.run(
            [freebuff_bin, "auth"],
            timeout=120,
        )
        if result.returncode == 0:
            console.print("[green]✓ Authentication successful[/green]")
            # Try to detect the saved token and write it to .env
            token = get_token_from_config()
            if token:
                env_path = save_token_to_env(token)
                console.print(f"[green]✓ Token saved to {env_path}[/green]")
            return True
        else:
            console.print("[yellow]⚠ Auth flow was cancelled or failed.[/yellow]")
            return False
    except subprocess.TimeoutExpired:
        console.print("[yellow]⚠ Auth flow timed out.[/yellow]")
        return False
    except KeyboardInterrupt:
        console.print("\n[yellow]Auth flow cancelled.[/yellow]")
        return False


# ── Full Setup ───────────────────────────────────────────────────────────

def setup_freebuff(auto_auth: bool = False) -> dict[str, str | bool]:
    """Run the full freebuff setup flow.

    1. Check if installed → install if missing
    2. Check if authenticated → optionally run auth flow
    3. Return status summary

    Returns a dict with keys:
      - installed: bool
      - version: str | None
      - authenticated: bool
      - token_source: str | None ("env", "config", or None)
    """
    result: dict[str, str | bool] = {
        "installed": False,
        "version": None,
        "authenticated": False,
        "token_source": None,
    }

    # Step 1: Install
    if not is_freebuff_installed():
        if not install_freebuff():
            return result

    result["installed"] = True
    result["version"] = get_freebuff_version()

    # Step 2: Check auth
    token = get_freebuff_token()
    if token:
        result["authenticated"] = True
        result["token_source"] = "env" if get_token_from_env() else "config"
    elif auto_auth:
        if run_auth_flow():
            token = get_freebuff_token()
            if token:
                result["authenticated"] = True
                result["token_source"] = "env" if get_token_from_env() else "config"

    return result
