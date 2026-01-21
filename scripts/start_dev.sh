#!/bin/bash
# scripts/start_dev.sh
# Development startup script for NEXUS AI v2.2.0

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}  NEXUS AI v2.2.0 - Development Mode                        ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python_version=$(python --version 2>&1)
echo -e "${GREEN}$python_version${NC}"
echo

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python -m venv .venv
    echo -e "${GREEN}Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -q -e ".[dev]" 2>/dev/null || pip install -q -r requirements.txt
echo -e "${GREEN}Dependencies installed${NC}"
echo

# Check environment variables
echo -e "${YELLOW}Checking environment variables...${NC}"
if [ -f ".env" ]; then
    source .env
    echo -e "${GREEN}.env file found${NC}"
else
    echo -e "${YELLOW}Warning: .env file not found, using defaults${NC}"
fi
echo

# Start services (PostgreSQL, Redis)
echo -e "${YELLOW}Starting infrastructure services...${NC}"
if command -v docker &> /dev/null; then
    if docker compose ps --status running 2>/dev/null | grep -q postgres; then
        echo -e "${GREEN}PostgreSQL is already running${NC}"
    else
        echo -e "${YELLOW}Starting PostgreSQL...${NC}"
        docker compose up -d postgres redis 2>/dev/null || echo -e "${YELLOW}Docker not available, skipping${NC}"
    fi
else
    echo -e "${YELLOW}Docker not available, skipping service startup${NC}"
fi
echo

# Run database migrations
echo -e "${YELLOW}Running database migrations...${NC}"
python scripts/init_db.py --verify-only 2>/dev/null || echo -e "${YELLOW}Database check skipped${NC}"
echo

# Start backend API
echo -e "${YELLOW}Starting NEXUS AI Backend API...${NC}"
echo -e "${GREEN}API will be available at http://localhost:8000${NC}"
echo -e "${GREEN}API docs at http://localhost:8000/docs${NC}"
echo

# Start in development mode
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir api --reload-dir core --reload-dir agents --reload-dir config &

# Store backend PID
BACKEND_PID=$!
echo $BACKEND_PID > .backend.pid

echo
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}NEXUS AI Development Server Started Successfully!${NC}"
echo
echo -e "  Backend: ${BLUE}http://localhost:8000${NC}"
echo -e "  API Docs: ${BLUE}http://localhost:8000/docs${NC}"
echo -e "  WebSocket: ${BLUE}ws://localhost:8000/api/ws${NC}"
echo
echo -e "To stop: kill \$(cat .backend.pid) && rm .backend.pid"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"

# Wait for backend
wait $BACKEND_PID
