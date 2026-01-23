#!/bin/bash
# Clean intermediate and generated files
#
# Usage:
#   ./scripts/clean.sh          # Clean generated files only
#   ./scripts/clean.sh --all    # Clean everything including vectorstore
#   ./scripts/clean.sh --dry-run # Show what would be deleted

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

DRY_RUN=false
CLEAN_ALL=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            ;;
        --all)
            CLEAN_ALL=true
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run    Show what would be deleted without actually deleting"
            echo "  --all        Clean everything including vectorstore and knowledge base"
            echo "  -h, --help   Show this help message"
            exit 0
            ;;
    esac
done

cd "$PROJECT_ROOT"

echo -e "${GREEN}CUDA Test RAG - Cleanup Script${NC}"
echo "================================"
echo ""

if $DRY_RUN; then
    echo -e "${YELLOW}[DRY RUN] No files will be deleted${NC}"
    echo ""
fi

# Function to remove files/directories
clean_path() {
    local path="$1"
    local description="$2"

    if [ -e "$path" ]; then
        if $DRY_RUN; then
            echo -e "  ${YELLOW}[WOULD DELETE]${NC} $path"
        else
            rm -rf "$path"
            echo -e "  ${GREEN}[DELETED]${NC} $path"
        fi
    fi
}

# Function to clean directory contents but keep .gitkeep
clean_dir_contents() {
    local dir="$1"
    local description="$2"

    if [ -d "$dir" ]; then
        echo -e "${GREEN}Cleaning $description...${NC}"
        find "$dir" -mindepth 1 -not -name '.gitkeep' -print0 2>/dev/null | while IFS= read -r -d '' item; do
            if $DRY_RUN; then
                echo -e "  ${YELLOW}[WOULD DELETE]${NC} $item"
            else
                rm -rf "$item"
                echo -e "  ${GREEN}[DELETED]${NC} $item"
            fi
        done
    fi
}

# 1. Python cache files
echo -e "${GREEN}Cleaning Python cache...${NC}"
find . -type d -name "__pycache__" -print0 2>/dev/null | while IFS= read -r -d '' dir; do
    clean_path "$dir" "pycache"
done
find . -type f -name "*.pyc" -print0 2>/dev/null | while IFS= read -r -d '' file; do
    clean_path "$file" "pyc"
done
find . -type f -name "*.pyo" -print0 2>/dev/null | while IFS= read -r -d '' file; do
    clean_path "$file" "pyo"
done

# 2. Test/coverage artifacts
echo -e "${GREEN}Cleaning test artifacts...${NC}"
clean_path ".pytest_cache" "pytest cache"
clean_path ".coverage" "coverage data"
clean_path "htmlcov" "coverage html"
clean_path ".mypy_cache" "mypy cache"
clean_path ".ruff_cache" "ruff cache"

# 3. Build artifacts
echo -e "${GREEN}Cleaning build artifacts...${NC}"
clean_path "build" "build directory"
clean_path "dist" "dist directory"
clean_path "*.egg-info" "egg-info"
find . -type d -name "*.egg-info" -print0 2>/dev/null | while IFS= read -r -d '' dir; do
    clean_path "$dir" "egg-info"
done

# 4. Generated request files
clean_dir_contents "data/requests" "generated requests"

# 5. Log files
echo -e "${GREEN}Cleaning log files...${NC}"
find . -type f -name "*.log" -print0 2>/dev/null | while IFS= read -r -d '' file; do
    clean_path "$file" "log file"
done
clean_path "logs" "logs directory"

# 6. SQLite databases
echo -e "${GREEN}Cleaning database files...${NC}"
find ./data -type f -name "*.db" -print0 2>/dev/null | while IFS= read -r -d '' file; do
    clean_path "$file" "database"
done
find ./data -type f -name "*.sqlite" -print0 2>/dev/null | while IFS= read -r -d '' file; do
    clean_path "$file" "sqlite"
done

# 7. Generated output files
echo -e "${GREEN}Cleaning generated files...${NC}"
find . -type f -name "*.generated.md" -print0 2>/dev/null | while IFS= read -r -d '' file; do
    clean_path "$file" "generated md"
done
find . -type f -name "*.generated.yaml" -print0 2>/dev/null | while IFS= read -r -d '' file; do
    clean_path "$file" "generated yaml"
done
clean_path "output" "output directory"

# 8. Optional: Clean vectorstore and knowledge base (--all flag)
if $CLEAN_ALL; then
    echo ""
    echo -e "${RED}Cleaning vectorstore and knowledge base (--all)...${NC}"
    clean_dir_contents "data/vectorstore" "vectorstore"
    clean_dir_contents "data/knowledge_base" "knowledge base"
    clean_dir_contents "data/docs" "docs"
fi

echo ""
if $DRY_RUN; then
    echo -e "${YELLOW}Dry run complete. No files were deleted.${NC}"
    echo "Run without --dry-run to actually delete files."
else
    echo -e "${GREEN}Cleanup complete!${NC}"
fi
