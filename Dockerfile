# NEXUS AI Dockerfile
# Multi-stage build for optimized production image

# ==========================
# Stage 1: Builder
# ==========================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Install Playwright browsers (optional, for scraping)
RUN pip install --no-cache-dir --user playwright \
    && python -m playwright install chromium --with-deps || true


# ==========================
# Stage 2: Production
# ==========================
FROM python:3.11-slim as production

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -r nexus && \
    chown -R nexus:nexus /app

USER nexus

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NEXUS_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000

# Default command (FastAPI backend)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# ==========================
# Stage 3: Development
# ==========================
FROM python:3.11-slim as development

WORKDIR /app

# Install all dependencies including dev
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt requirements-dev.txt* ./
RUN pip install --no-cache-dir -r requirements.txt && \
    ([ -f requirements-dev.txt ] && pip install --no-cache-dir -r requirements-dev.txt || true)

# Install Playwright for scraping
RUN pip install --no-cache-dir playwright && \
    playwright install chromium --with-deps || true

# Copy application code
COPY . .

# Environment
ENV PYTHONUNBUFFERED=1 \
    NEXUS_ENV=development

# Dev server with auto-reload
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# ==========================
# Stage 4: MCP Servers
# ==========================
FROM python:3.11-slim as mcp

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy only MCP-related code
COPY config/ ./config/
COPY mcp_servers/ ./mcp_servers/
COPY core/ ./core/

ENV PYTHONUNBUFFERED=1

# Default: run all MCP servers
CMD ["python", "-m", "scripts.setup_mcp", "--mode", "both"]
