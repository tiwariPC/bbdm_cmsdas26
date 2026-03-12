#!/usr/bin/env bash
# Create a virtual environment and install dependencies for the bbDM CMS DAS exercise.
# Run from any directory: bash scripts/setup_venv.sh

set -e
# Resolve project root (strip carriage return in case of CRLF line endings)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" && pwd)"
SCRIPT_DIR="${SCRIPT_DIR//$'\r'/}"
SCRIPT_DIR="${SCRIPT_DIR//$'\n'/}"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT//$'\r'/}"
PROJECT_ROOT="${PROJECT_ROOT//$'\n'/}"
VENV_DIR="$PROJECT_ROOT/.venv"
REQUIREMENTS="$PROJECT_ROOT/requirements.txt"

cd "$PROJECT_ROOT"

if ! command -v python3 &>/dev/null; then
    echo "python3 not found. Please install Python 3.8+ or use a different environment."
    exit 1
fi

echo "Project root: $PROJECT_ROOT"
echo "Creating virtual environment at $VENV_DIR ..."
python3 -m venv "$VENV_DIR"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "Upgrading pip ..."
pip install --upgrade pip -q

if [[ -f "$REQUIREMENTS" ]]; then
    echo "Installing dependencies from requirements.txt ..."
    pip install -r "$REQUIREMENTS"
else
    echo "requirements.txt not found; installing core packages ..."
    pip install coffea matplotlib hist uproot numpy awkward ipykernel
fi

echo "Registering Jupyter kernel (for SWAN / JupyterLab) ..."
python -m ipykernel install --user --name=bbdm-cmsdas26 --display-name="Python (bbDM CMS DAS)"

echo ""
echo "Done. Next steps:"
echo "  Activate the environment:  source $VENV_DIR/bin/activate"
echo "  In a notebook (e.g. on SWAN): Kernel → Change kernel → Python (bbDM CMS DAS)"
