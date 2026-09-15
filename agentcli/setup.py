"""Setup utilities for Freebuff CLI and OpenCode — install, auth, and token management.

The freebuff CLI stores its auth token in:
  - ~/.config/manicode/credentials.json (actual location, under 'authToken')
  - ~/.config/freebuff/config.json       (legacy fallback)

OpenCode uses API key authentication via Bearer token in .env.

This module detects and manages tokens, bridging them into agentcli's
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
    if sys.platform == "win32":
        appdata = Path.home() / "AppData" / "Roaming"
        return appdata / "freebuff"
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

def _manicode_credentials_path() -> Path:
    """Return the path to manicode's credentials.json (where freebuff stores tokens)."""
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "manicode" / "credentials.json"
    if sys.platform == "win32":
        appdata = Path.home() / "AppData" / "Roaming"
        return appdata / "manicode" / "credentials.json"
    return home / ".config" / "manicode" / "credentials.json"


def _read_token_from_json(path: Path) -> str | None:
    """Read a token value from a JSON file, trying common key names."""
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.debug(f"Failed to read {path}: {e}")
        return None

    for key in ("authToken", "token", "api_key", "auth_token", "apiKey", "accessToken"):
        val = cfg.get(key)
        if val and isinstance(val, str) and len(val) > 0:
            return val

    return None


def get_token_from_config() -> str | None:
    """Read the auth token from freebuff's credentials file.

    Checks both the manicode credentials.json (primary) and the
    legacy freebuff config.json (fallback).
    """
    # Primary: ~/.config/manicode/credentials.json → authToken
    token = _read_token_from_json(_manicode_credentials_path())
    if token:
        return token

    # Fallback: ~/.config/freebuff/config.json
    token = _read_token_from_json(freebuff_config_path())
    if token:
        return token

    # Fallback: any profile inside credentials.json
    creds_path = _manicode_credentials_path()
    if creds_path.is_file():
        try:
            with creds_path.open(encoding="utf-8") as f:
                creds = json.load(f)
            for profile in creds.values():
                if isinstance(profile, dict):
                    val = profile.get("authToken")
                    if val and isinstance(val, str) and len(val) > 0:
                        return val
        except (json.JSONDecodeError, OSError):
            pass

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
        console.print("[red]✗ npm install failed:[/red]")
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
            [freebuff_bin, "login"],
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
        console.print("[yellow]⚠ Auth flow was cancelled or failed.[/yellow]")
        return False
    except subprocess.TimeoutExpired:
        console.print("[yellow]⚠ Auth flow timed out.[/yellow]")
        return False
    except KeyboardInterrupt:
        console.print("\n[yellow]Auth flow cancelled.[/yellow]")
        return False


# ── Full Freebuff Setup ──────────────────────────────────────────────────

def setup_freebuff(auto_auth: bool = False) -> dict[str, str | bool | None]:
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
    result: dict[str, str | bool | None] = {
        "installed": False,
        "version": None,
        "authenticated": False,
        "token_source": None,
    }

    # Step 1: Install
    if not is_freebuff_installed() and not install_freebuff():
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


# ── OpenCode Setup ──────────────────────────────────────────────────────

def _find_opencode_env_path() -> Path:
    """Find or create the .env file path for OpenCode config."""
    local = Path(".env")
    if local.is_file():
        return local
    config_dir = Path.home() / ".config" / "agentcli"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / ".env"


def get_opencode_config() -> dict[str, str | None]:
    """Get current OpenCode configuration from .env."""
    from agentcli.config import get_settings
    s = get_settings()
    return {
        "api_key": s.opencode_api_key or None,
        "tier": s.opencode_tier or "zen",
        "base_url": s.opencode_base_url or None,
    }


def save_opencode_config(
    api_key: str,
    tier: str = "zen",
    env_path: Path | None = None,
) -> Path:
    """Save OpenCode API key and tier to .env file."""
    from agentcli.providers.opencode import OPENCODE_GO_BASE_URL, OPENCODE_ZEN_BASE_URL

    if env_path is None:
        env_path = _find_opencode_env_path()

    base_url = OPENCODE_GO_BASE_URL if tier == "go" else OPENCODE_ZEN_BASE_URL

    lines: list[str] = []
    key_written = tier_written = url_written = False

    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("OPENCODE_API_KEY="):
                lines.append(f"OPENCODE_API_KEY={api_key}")
                key_written = True
            elif stripped.startswith("OPENCODE_TIER="):
                lines.append(f"OPENCODE_TIER={tier}")
                tier_written = True
            elif stripped.startswith("OPENCODE_BASE_URL="):
                lines.append(f"OPENCODE_BASE_URL={base_url}")
                url_written = True
            else:
                lines.append(line)

    if not key_written:
        lines.append(f"OPENCODE_API_KEY={api_key}")
    if not tier_written:
        lines.append(f"OPENCODE_TIER={tier}")
    if not url_written:
        lines.append(f"OPENCODE_BASE_URL={base_url}")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env_path


def verify_opencode_key(api_key: str, tier: str = "zen") -> tuple[bool, str | None]:
    """Verify an OpenCode API key by calling /models endpoint."""
    import httpx

    from agentcli.providers.opencode import OPENCODE_GO_BASE_URL, OPENCODE_ZEN_BASE_URL

    base_url = OPENCODE_GO_BASE_URL if tier == "go" else OPENCODE_ZEN_BASE_URL

    try:
        with httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        ) as client:
            resp = client.get("/models")
            if resp.status_code == 200:
                models_data = resp.json()
                model_count = len(models_data.get("data", []))
                return True, f"Valid key — {model_count} models available"
            if resp.status_code == 401:
                return False, "Invalid API key (401 Unauthorized)"
            return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return False, f"Verification failed: {e}"


# ── Unified Multi-Provider Setup ────────────────────────────────────────

def setup_provider(
    provider: str,
    *,
    api_key: str | None = None,
    tier: str | None = None,
    auth: bool = False,
    auto_install: bool = True,  # noqa: ARG001 reserved for future per-provider install control
) -> dict[str, str | bool | None]:
    """Unified setup for LLM providers.

    Args:
        provider: "freebuff" or "opencode"
        api_key: API key for OpenCode (required for opencode)
        tier: "zen" or "go" for OpenCode
        auth: Run interactive auth flow for Freebuff
        auto_install: Auto-install CLI for Freebuff

    Returns:
        Dict with setup status: installed, authenticated, version, token_source, etc.
    """
    result: dict[str, str | bool | None] = {
        "provider": provider,
        "installed": False,
        "authenticated": False,
        "version": None,
        "token_source": None,
    }

    if provider == "freebuff":
        # Step 1: Install
        if not is_freebuff_installed() and not install_freebuff():
            return result

        result["installed"] = True
        result["version"] = get_freebuff_version()

        # Step 2: Check/configure auth
        token = get_freebuff_token()
        if token:
            result["authenticated"] = True
            result["token_source"] = "env" if get_token_from_env() else "config"
        elif auth:
            if run_auth_flow():
                token = get_freebuff_token()
                if token:
                    result["authenticated"] = True
                    result["token_source"] = "env" if get_token_from_env() else "config"

    elif provider == "opencode":
        if not api_key:
            result["error"] = "OpenCode API key is required"
            return result

        tier = tier or "zen"
        if tier not in ("zen", "go"):
            result["error"] = "Invalid tier. Use 'zen' or 'go'"
            return result

        # Verify key
        valid, msg = verify_opencode_key(api_key, tier)
        if not valid:
            result["error"] = f"OpenCode key verification failed: {msg}"
            return result

        # Save to .env
        env_path = save_opencode_config(api_key, tier)
        result["installed"] = True
        result["authenticated"] = True
        result["token_source"] = "env"
        result["version"] = tier
        result["env_path"] = str(env_path)

    else:
        result["error"] = f"Unknown provider: {provider}"

    return result


def setup_all(
    *,
    opencode_api_key: str | None = None,
    opencode_tier: str | None = None,
    freebuff_auth: bool = False,
    freebuff_auto_install: bool = True,
) -> dict[str, dict]:
    """Set up all configured providers in one call.

    Args:
        opencode_api_key: OpenCode API key (optional)
        opencode_tier: "zen" or "go" for OpenCode
        freebuff_auth: Run Freebuff auth flow
        freebuff_auto_install: Auto-install Freebuff CLI

    Returns:
        Dict mapping provider name to its setup result.
    """
    results = {}

    # OpenCode setup
    if opencode_api_key:
        results["opencode"] = setup_provider(
            "opencode",
            api_key=opencode_api_key,
            tier=opencode_tier,
        )
    else:
        # Check if already configured
        from agentcli.config import get_settings
        s = get_settings()
        if s.opencode_api_key:
            results["opencode"] = {
                "provider": "opencode",
                "installed": True,
                "authenticated": True,
                "version": s.opencode_tier,
                "token_source": "env",
            }

    # Freebuff setup
    results["freebuff"] = setup_provider(
        "freebuff",
        auth=freebuff_auth,
        auto_install=freebuff_auto_install,
    )

    return results