#!/usr/bin/env bash
# Activate the project virtual environment. Optionally start Jupyter.
# Usage: source scripts/start.sh   OR   bash scripts/start.sh [--jupyter]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" && pwd)"
SCRIPT_DIR="${SCRIPT_DIR//$'\r'/}"
SCRIPT_DIR="${SCRIPT_DIR//$'\n'/}"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT//$'\r'/}"
PROJECT_ROOT="${PROJECT_ROOT//$'\n'/}"
VENV_DIR="$PROJECT_ROOT/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Virtual environment not found. Run first: bash scripts/setup_venv.sh"
    exit 1
fi

# Activate (when sourced, the caller's shell gets the env)
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    # Script was sourced (e.g. source scripts/start.sh)
    echo "Virtual environment activated. Run 'jupyter notebook' or open a notebook and select kernel 'Python (bbDM CMS DAS)'."
    return 0
fi

# Script was executed (e.g. bash scripts/start.sh)
if [[ "$1" == "--jupyter" || "$1" == "--notebook" ]]; then
    cd "$PROJECT_ROOT"
    echo "Starting Jupyter from $PROJECT_ROOT ..."
    exec jupyter notebook
fi

echo "Virtual environment activated. Run 'jupyter notebook' or open a notebook and select kernel 'Python (bbDM CMS DAS)'."
