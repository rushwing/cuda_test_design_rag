#!/bin/bash
# CUDA Test Case Generation Script Wrapper
#
# Usage:
#   ./scripts/gen_test_cases.sh --auto                    # Auto mode (both stages)
#   ./scripts/gen_test_cases.sh --intent                  # Stage 1 only
#   ./scripts/gen_test_cases.sh --approve REQ-xxx         # Approve intents
#   ./scripts/gen_test_cases.sh --case -r REQ-xxx         # Stage 2 only
#   ./scripts/gen_test_cases.sh --list                    # List requests

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default config file
DEFAULT_CONFIG="$PROJECT_ROOT/config/test_request_sample.json"

# Activate virtual environment if it exists
if [ -f "$PROJECT_ROOT/cuda_venv/bin/activate" ]; then
    source "$PROJECT_ROOT/cuda_venv/bin/activate"
elif [ -f "$PROJECT_ROOT/cuda_venv/Scripts/activate" ]; then
    source "$PROJECT_ROOT/cuda_venv/Scripts/activate"
fi

# Run the Python script
cd "$PROJECT_ROOT"

# If no config specified and not a list/approve/case-only command, use default
if [[ ! " $@ " =~ " --config " ]] && [[ ! " $@ " =~ " -c " ]] && \
   [[ ! " $@ " =~ " --list " ]] && [[ ! " $@ " =~ " --approve " ]] && \
   [[ ! " $@ " =~ " --case " ]]; then
    python scripts/gen_test_cases.py --config "$DEFAULT_CONFIG" "$@"
else
    python scripts/gen_test_cases.py "$@"
fi
