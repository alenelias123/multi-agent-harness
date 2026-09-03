#!/usr/bin/env bash
# install_freebuff.sh — Install and configure the freebuff CLI for agentcli
#
# Usage:
#   bash scripts/install_freebuff.sh          # install only
#   bash scripts/install_freebuff.sh --auth   # install + run auth flow
#
# This script:
#   1. Checks for node/npm availability
#   2. Installs freebuff globally if not already installed
#   3. Optionally runs the auth flow to save the API token
#
# The freebuff CLI stores its auth token in:
#   - ~/.freebuff/config.json   (Linux/macOS)
#   - %APPDATA%/freebuff/config.json  (Windows)

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

AUTH_MODE=false
if [[ "${1:-}" == "--auth" ]]; then
    AUTH_MODE=true
fi

echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Freebuff CLI Installer for AgentCLI         ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
echo ""

# ── Step 1: Check for Node.js ───────────────────────────────────────────

echo -e "${YELLOW}→ Checking for Node.js...${NC}"
if ! command -v node &>/dev/null; then
    echo -e "${RED}✗ Node.js is not installed.${NC}"
    echo -e "  Please install Node.js first:"
    echo -e "    ${CYAN}curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -${NC}"
    echo -e "    ${CYAN}sudo apt-get install -y nodejs${NC}"
    echo -e "  Or visit: https://nodejs.org/"
    exit 1
fi
NODE_VERSION=$(node --version)
echo -e "${GREEN}✓ Node.js ${NODE_VERSION} found${NC}"

# ── Step 2: Check for npm ───────────────────────────────────────────────

echo -e "${YELLOW}→ Checking for npm...${NC}"
if ! command -v npm &>/dev/null; then
    echo -e "${RED}✗ npm is not installed.${NC}"
    echo -e "  npm should come with Node.js. Please reinstall Node.js."
    exit 1
fi
NPM_VERSION=$(npm --version)
echo -e "${GREEN}✓ npm ${NPM_VERSION} found${NC}"

# ── Step 3: Check if freebuff is already installed ──────────────────────

echo -e "${YELLOW}→ Checking for freebuff CLI...${NC}"
if command -v freebuff &>/dev/null; then
    FREEBUFF_VERSION=$(freebuff --version 2>/dev/null || echo "unknown")
    echo -e "${GREEN}✓ freebuff ${FREEBUFF_VERSION} is already installed${NC}"
else
    echo -e "${YELLOW}  freebuff not found. Installing globally...${NC}"
    if npm install -g freebuff 2>/dev/null; then
        echo -e "${GREEN}✓ freebuff installed successfully${NC}"
        FREEBUFF_VERSION=$(freebuff --version 2>/dev/null || echo "installed")
        echo -e "  Version: ${FREEBUFF_VERSION}"
    else
        echo -e "${RED}✗ Failed to install freebuff.${NC}"
        echo -e "  Try running: ${CYAN}npm install -g freebuff${NC}"
        exit 1
    fi
fi

# ── Step 4: Locate config directory ─────────────────────────────────────

CONFIG_DIR=""
if [[ -n "${XDG_CONFIG_HOME:-}" ]]; then
    CONFIG_DIR="${XDG_CONFIG_HOME}/freebuff"
elif [[ -d "${HOME}/.config" ]]; then
    CONFIG_DIR="${HOME}/.config/freebuff"
else
    CONFIG_DIR="${HOME}/.freebuff"
fi

CONFIG_FILE="${CONFIG_DIR}/config.json"
ENV_FILE=""
if [[ -f ".env" ]]; then
    ENV_FILE=".env"
elif [[ -f "${HOME}/.config/agentcli/.env" ]]; then
    ENV_FILE="${HOME}/.config/agentcli/.env"
fi

echo ""
echo -e "${CYAN}Configuration locations:${NC}"
echo -e "  Config dir:  ${CONFIG_DIR}"
echo -e "  Config file: ${CONFIG_FILE}"
if [[ -n "${ENV_FILE}" ]]; then
    echo -e "  .env file:   ${ENV_FILE}"
else
    echo -e "  .env file:   ${YELLOW}not found (will create in CWD)${NC}"
fi

# ── Step 5: Check existing token ────────────────────────────────────────

echo ""
if [[ -f "${CONFIG_FILE}" ]]; then
    echo -e "${GREEN}✓ freebuff config exists at ${CONFIG_FILE}${NC}"
    # Check if token is set in the config
    if command -v python3 &>/dev/null; then
        HAS_TOKEN=$(python3 -c "
import json, sys
try:
    with open('${CONFIG_FILE}') as f:
        cfg = json.load(f)
    token = cfg.get('token', '') or cfg.get('api_key', '') or cfg.get('auth_token', '')
    sys.exit(0 if token else 1)
except:
    sys.exit(1)
" 2>/dev/null) && true
        if [[ $? -eq 0 ]]; then
            echo -e "  Token: ${GREEN}configured${NC}"
        else
            echo -e "  Token: ${YELLOW}not configured${NC}"
        fi
    fi
else
    echo -e "${YELLOW}⚠ No freebuff config found at ${CONFIG_FILE}${NC}"
fi

# Check for FREEBUFF_TOKEN in env
if [[ -n "${FREEBUFF_TOKEN:-}" ]]; then
    echo -e "  Env var ${CYAN}FREEBUFF_TOKEN${NC}: ${GREEN}set${NC}"
elif grep -q "^FREEBUFF_TOKEN=" "${ENV_FILE}" 2>/dev/null; then
    echo -e "  .env ${CYAN}FREEBUFF_TOKEN${NC}: ${GREEN}set${NC}"
else
    echo -e "  ${YELLOW}No FREEBUFF_TOKEN in environment${NC}"
fi

# ── Step 6: Run auth if requested ───────────────────────────────────────

if [[ "${AUTH_MODE}" == "true" ]]; then
    echo ""
    echo -e "${CYAN}→ Running freebuff auth flow...${NC}"
    if freebuff auth; then
        echo -e "${GREEN}✓ Authentication successful${NC}"
    else
        echo -e "${YELLOW}⚠ Auth flow may have been cancelled or failed.${NC}"
        echo -e "  You can manually set your token later:"
        echo -e "    ${CYAN}export FREEBUFF_TOKEN=your_token_here${NC}"
    fi
fi

# ── Step 7: Write to .env if needed ─────────────────────────────────────

echo ""
if [[ -n "${FREEBUFF_TOKEN:-}" || -f "${CONFIG_FILE}" ]]; then
    echo -e "${GREEN}✓ Freebuff is ready to use with agentcli${NC}"
    echo ""
    echo -e "${CYAN}To enable in agentcli, add to your .env file:${NC}"
    echo -e "  ${GREEN}FREEBUFF_ENABLED=true${NC}"
else
    echo -e "${YELLOW}⚠ Freebuff is installed but not yet authenticated.${NC}"
    echo ""
    echo -e "Options:"
    echo -e "  1. Run: ${CYAN}bash scripts/install_freebuff.sh --auth${NC}"
    echo -e "  2. Or set the token manually:"
    echo -e "     ${CYAN}export FREEBUFF_TOKEN=your_freebuff_api_token${NC}"
    echo ""
    echo -e "Then add to your .env file:"
    echo -e "  ${GREEN}FREEBUFF_ENABLED=true${NC}"
fi

echo ""
echo -e "${CYAN}Done.${NC}"
