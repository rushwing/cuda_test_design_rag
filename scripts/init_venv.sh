#!/bin/bash
# Initialize cuda_venv virtual environment using uv
# For Windows 11 (run via Git Bash, WSL, or compatible shell)

set -e

VENV_NAME="cuda_venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== CUDA Test RAG - Virtual Environment Setup ==="
echo "Project root: $PROJECT_ROOT"
echo "Virtual environment: $VENV_NAME"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    echo "Install uv with: pip install uv"
    echo "Or on Windows: winget install astral-sh.uv"
    echo "Or: powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\""
    exit 1
fi

echo "uv version: $(uv --version)"
echo ""

cd "$PROJECT_ROOT"

# Remove existing venv if it exists
if [ -d "$VENV_NAME" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV_NAME"
fi

# Create virtual environment with uv
echo "Creating virtual environment '$VENV_NAME'..."
uv venv "$VENV_NAME" --python 3.11

# Determine activation script path based on OS
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]] || [[ -n "$WINDIR" ]]; then
    # Windows (Git Bash / MSYS)
    ACTIVATE_SCRIPT="$VENV_NAME/Scripts/activate"
else
    # Linux / macOS / WSL
    ACTIVATE_SCRIPT="$VENV_NAME/bin/activate"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$ACTIVATE_SCRIPT"

# Install dependencies using uv
echo ""
echo "Installing project dependencies..."
uv pip install -e ".[dev]"

# Create .env from example if it doesn't exist
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo ""
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "NOTE: Please edit .env and add your OPENAI_API_KEY"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the virtual environment:"
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]] || [[ -n "$WINDIR" ]]; then
    echo "  Git Bash:    source $VENV_NAME/Scripts/activate"
    echo "  PowerShell:  .\\$VENV_NAME\\Scripts\\Activate.ps1"
    echo "  CMD:         $VENV_NAME\\Scripts\\activate.bat"
else
    echo "  source $VENV_NAME/bin/activate"
fi
echo ""
echo "Then run:"
echo "  cuda-test-rag --help"
echo ""
