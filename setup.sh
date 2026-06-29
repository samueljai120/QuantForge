#!/usr/bin/env bash
set -euo pipefail

# QuantForge — First-time setup
# Usage: ./setup.sh

echo "=== QuantForge Setup ==="

# Check prerequisites
command -v python3 >/dev/null 2>&1 || { echo "Error: python3 is required. Install Python 3.10+ and try again."; exit 1; }
command -v pip3 >/dev/null 2>&1 || command -v pip >/dev/null 2>&1 || { echo "Error: pip is required. Install pip and try again."; exit 1; }

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; }; then
  echo "Error: Python 3.10+ is required (found $PYTHON_VERSION)."
  exit 1
fi
echo "Python $PYTHON_VERSION found."

# Virtual environment
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment at .venv ..."
  python3 -m venv .venv
fi

# Activate
# shellcheck disable=SC1091
source .venv/bin/activate

# Upgrade pip quietly
pip install --upgrade pip --quiet

# Install dependencies
echo "Installing dependencies from requirements.txt ..."
pip install -r requirements.txt --quiet
echo "Dependencies installed."

# Environment file
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo ""
  echo "Created .env from .env.example — edit it with your configuration before running scripts."
fi

# Smoke test — verify all scripts parse cleanly
echo ""
echo "Running syntax check on scripts/ ..."
FAIL=0
for f in scripts/*.py; do
  if ! python3 -m py_compile "$f" 2>/dev/null; then
    echo "  SYNTAX ERROR: $f"
    FAIL=1
  fi
done
if [ "$FAIL" = "0" ]; then
  echo "  All scripts passed syntax check."
else
  echo "  Some scripts failed syntax check — check the output above."
  exit 1
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your configuration (API keys, QF_BASE_DIR, etc.)"
echo "  2. Activate the virtualenv:  source .venv/bin/activate"
echo "  3. Run the test suite:       python3 -m pytest tests/ -q"
echo "  4. Run a market scan:        QF_ALLOW_LOCAL_RUNTIME=1 python3 scripts/quantforge_paper.py scan"
echo "  5. Read the honest verdict:  cat docs/QUANTFORGE_VERDICT.md"
echo ""
echo "Using Claude Code? CLAUDE.md has all the context."
