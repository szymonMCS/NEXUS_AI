#!/bin/bash
# scripts/start_prod.sh
# Production startup script for NEXUS AI v2.2.0

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}  NEXUS AI v2.2.0 - Production Mode                        ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Set production environment
export NEXUS_ENV=production
export PYTHONOPTIMIZE=1

# Check for .env.production
if [ -f ".env.production" ]; then
    echo -e "${YELLOW}Loading production environment variables...${NC}"
    set -a
    source .env.production
    set +a
    echo -e "${GREEN}Production environment loaded${NC}"
else
    echo -e "${RED}Error: .env.production not found${NC}"
    echo "Please create .env.production with your production settings."
    exit 1
fi

echo

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down NEXUS AI...${NC}"
    if [ -f ".nexus.pid" ]; then
        kill $(cat .nexus.pid) 2>/dev/null || true
        rm .nexus.pid
    fi
    echo -e "${GREEN}NEXUS AI stopped${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Check if already running
if [ -f ".nexus.pid" ]; then
    OLD_PID=$(cat .nexus.pid)
    if kill -0 $OLD_PID 2>/dev/null; then
        echo -e "${RED}NEXUS AI is already running (PID: $OLD_PID)${NC}"
        exit 1
    else
        rm .nexus.pid
    fi
fi

# Verify database connection
echo -e "${YELLOW}Verifying database connection...${NC}"
if python -c "from database.db import engine; engine.connect()" 2>/dev/null; then
    echo -e "${GREEN}Database connection verified${NC}"
else
    echo -e "${RED}Error: Cannot connect to database${NC}"
    exit 1
fi
echo

# Build frontend (if needed)
if [ -d "frontend/app/dist" ]; then
    echo -e "${YELLOW}Frontend build found${NC}"
else
    echo -e "${YELLOW}Building frontend...${NC}"
    cd frontend/app
    npm run build 2>/dev/null || echo -e "${YELLOW}Frontend build skipped (npm not available)${NC}"
    cd ../..
fi
echo

# Start the application using gunicorn
echo -e "${YELLOW}Starting NEXUS AI Production Server...${NC}"

# Use gunicorn for production with multiple workers
if command -v gunicorn &> /dev/null; then
    echo -e "${GREEN}Using Gunicorn with 4 workers${NC}"
    gunicorn main:app \
        --bind 0.0.0.0:8000 \
        --workers 4 \
        --worker-class uvicorn.workers.UvicornWorker \
        --access-logfile - \
        --error-logfile - \
        --capture-output \
        --daemon \
        --pid .nexus.pid
else
    echo -e "${YELLOW}Gunicorn not available, using uvicorn${NC}"
    uvicorn main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 4 \
        --access-logfile - \
        --error-logfile - \
        --daemon \
        --pid .nexus.pid
fi

sleep 2

# Verify startup
if [ -f ".nexus.pid" ]; then
    PID=$(cat .nexus.pid)
    if kill -0 $PID 2>/dev/null; then
        echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}NEXUS AI Production Server Started Successfully!${NC}"
        echo
        echo -e "  PID: ${BLUE}$PID${NC}"
        echo -e "  API: ${BLUE}http://localhost:8000${NC}"
        echo -e "  Docs: ${BLUE}http://localhost:8000/docs${NC}"
        echo
        echo -e "  Metrics: ${BLUE}http://localhost:8000/metrics${NC}"
        echo -e "  Health: ${BLUE}http://localhost:8000/api/status${NC}"
        echo
        echo -e "${GREEN}Monitoring Services:${NC}"
        echo -e "  Prometheus: ${BLUE}http://localhost:9090${NC}"
        echo -e "  Grafana: ${BLUE}http://localhost:3030${NC}"
        echo
        echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
        echo
        echo "To stop: kill \$(cat .nexus.pid)"
        echo
        
        # Keep script running
        while true; do
            sleep 10
            if ! kill -0 $PID 2>/dev/null; then
                echo -e "${RED}Server process died unexpectedly${NC}"
                exit 1
            fi
        done
    else
        echo -e "${RED}Error: Server failed to start${NC}"
        exit 1
    fi
else
    echo -e "${RED}Error: Could not start server (no PID file)${NC}"
    exit 1
fi
