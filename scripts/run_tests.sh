#!/bin/bash
# scripts/run_tests.sh
# Test runner for NEXUS AI v2.2.0

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}  NEXUS AI v2.2.0 - Test Suite                             ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Default options
VERBOSE=false
COVERAGE=false
TYPE=""
PATTERN=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -t|--type)
            TYPE="$2"
            shift 2
            ;;
        -p|--pattern)
            PATTERN="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo
            echo "Options:"
            echo "  -v, --verbose      Verbose output"
            echo "  -c, --coverage     Generate coverage report"
            echo "  -t, --type         Test type: unit, integration, e2e"
            echo "  -p, --pattern      Test pattern to match"
            echo "  -h, --help         Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${YELLOW}Installing pytest...${NC}"
    pip install pytest pytest-asyncio pytest-cov
fi

echo -e "${YELLOW}Running tests...${NC}"
echo

# Build pytest arguments
PYTEST_ARGS=("tests/")

if [ -n "$PATTERN" ]; then
    PYTEST_ARGS+=("-k" "$PATTERN")
fi

if [ "$VERBOSE" = true ]; then
    PYTEST_ARGS+=("-v" "--tb=short")
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_ARGS+=("--cov=." "--cov-report=term-missing" "--cov-report=html")
fi

if [ -n "$TYPE" ]; then
    case $TYPE in
        unit)
            PYTEST_ARGS+=("tests/test_*.py")
            ;;
        integration)
            PYTEST_ARGS+=("tests/integration/")
            ;;
        e2e)
            PYTEST_ARGS+=("tests/e2e/")
            ;;
    esac
fi

echo -e "${BLUE}pytest ${PYTEST_ARGS[*]}${NC}"
echo

# Run tests
set +e  # Don't exit on test failure
pytest "${PYTEST_ARGS[@]}"
TEST_EXIT_CODE=$?
set -e

echo

# Check results
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
else
    echo -e "${RED}════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}✗ Some tests failed (exit code: $TEST_EXIT_CODE)${NC}"
    echo -e "${RED}════════════════════════════════════════════════════════════${NC}"
fi

# Show coverage if requested
if [ "$COVERAGE" = true ]; then
    echo
    echo -e "${YELLOW}Coverage report generated in htmlcov/${NC}"
fi

exit $TEST_EXIT_CODE
