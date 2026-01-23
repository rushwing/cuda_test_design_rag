# Initialize cuda_venv virtual environment using uv
# For Windows 11 (PowerShell)

$ErrorActionPreference = "Stop"

$VENV_NAME = "cuda_venv"
$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot

Write-Host "=== CUDA Test RAG - Virtual Environment Setup ===" -ForegroundColor Cyan
Write-Host "Project root: $PROJECT_ROOT"
Write-Host "Virtual environment: $VENV_NAME"
Write-Host ""

# Check if uv is installed
try {
    $uvVersion = uv --version
    Write-Host "uv version: $uvVersion"
} catch {
    Write-Host "Error: uv is not installed." -ForegroundColor Red
    Write-Host "Install uv with: winget install astral-sh.uv"
    Write-Host "Or: irm https://astral.sh/uv/install.ps1 | iex"
    exit 1
}

Set-Location $PROJECT_ROOT

# Remove existing venv if it exists
if (Test-Path $VENV_NAME) {
    Write-Host "Removing existing virtual environment..."
    Remove-Item -Recurse -Force $VENV_NAME
}

# Create virtual environment with uv
Write-Host ""
Write-Host "Creating virtual environment '$VENV_NAME'..."
uv venv $VENV_NAME --python 3.11

# Activate virtual environment
Write-Host "Activating virtual environment..."
& ".\$VENV_NAME\Scripts\Activate.ps1"

# Install dependencies using uv
Write-Host ""
Write-Host "Installing project dependencies..."
uv pip install -e ".[dev]"

# Create .env from example if it doesn't exist
if ((-not (Test-Path ".env")) -and (Test-Path ".env.example")) {
    Write-Host ""
    Write-Host "Creating .env from .env.example..."
    Copy-Item .env.example .env
    Write-Host "NOTE: Please edit .env and add your OPENAI_API_KEY" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the virtual environment:"
Write-Host "  PowerShell:  .\$VENV_NAME\Scripts\Activate.ps1"
Write-Host "  CMD:         $VENV_NAME\Scripts\activate.bat"
Write-Host "  Git Bash:    source $VENV_NAME/Scripts/activate"
Write-Host ""
Write-Host "Then run:"
Write-Host "  cuda-test-rag --help"
Write-Host ""
