# Dockerfile
# Frontend build stage
FROM oven/bun:1 AS frontend-builder

WORKDIR /app

# Clone and build LightRAG frontend
RUN git clone https://github.com/HKUDS/LightRAG.git . && \
    cd lightrag_webui && \
    bun install --frozen-lockfile && \
    bun run build

# Main application stage
FROM python:3.12-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    pkg-config \
    curl \
    git \
    # LibreOffice for Office document processing (required by RAG-Anything)
    libreoffice \
    libreoffice-writer \
    libreoffice-calc \
    libreoffice-impress \
    # Image processing libraries
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # For MinerU
    libmagic1 \
    tesseract-ocr \
    tesseract-ocr-eng \
    # Additional dependencies
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create necessary directories
RUN mkdir -p /app/data/rag_storage \
    /app/data/inputs \
    /app/data/outputs \
    /app/data/tiktoken \
    /app/cache

# Clone and install LightRAG with API support
RUN git clone https://github.com/HKUDS/LightRAG.git /tmp/lightrag && \
    cd /tmp/lightrag && \
    uv pip install --system -e ".[api,offline]" && \
    # Remove git directory but keep the source
    rm -rf /tmp/lightrag/.git && \
    # Copy LightRAG to app directory for frontend integration
    cp -r /tmp/lightrag/lightrag /app/lightrag

# Copy pre-built frontend from builder stage
COPY --from=frontend-builder /app/lightrag/api/webui /app/lightrag/api/webui

# Clone and install RAG-Anything
RUN git clone https://github.com/HKUDS/RAG-Anything.git /tmp/raganything && \
    cd /tmp/raganything && \
    uv pip install --system -e ".[all]" && \
    rm -rf /tmp/raganything/.git

# Install MinerU with all dependencies
RUN uv pip install --system "magic-pdf[full]>=0.7.0"

# Install additional dependencies for Neo4j and Qdrant
RUN uv pip install --system \
    neo4j \
    qdrant-client \
    pillow \
    reportlab

# Pre-download tiktoken cache
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" || true

# Create non-root user
RUN useradd -m -u 1000 lightrag && \
    chown -R lightrag:lightrag /app

USER lightrag

# Expose ports
EXPOSE 9621

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9621/health || exit 1

# Start LightRAG server
CMD ["python", "-m", "lightrag.api.lightrag_server", \
     "--host", "0.0.0.0", \
     "--port", "9621", \
     "--working-dir", "/app/data/rag_storage", \
     "--input-dir", "/app/data/inputs"]