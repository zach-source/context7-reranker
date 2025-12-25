# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install package with all dependencies
RUN pip install --no-cache-dir --user ".[all]"

# Production image
FROM python:3.12-slim

WORKDIR /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash context7

# Copy installed packages from builder
COPY --from=builder /root/.local /home/context7/.local

# Copy source (for reference/debugging)
COPY --from=builder /app/src ./src

# Set PATH for installed packages
ENV PATH="/home/context7/.local/bin:$PATH"

# Default environment variables
ENV LLM_ENDPOINT="https://api.openai.com/v1" \
    LLM_MODEL="gpt-4o-mini" \
    LLM_TEMPERATURE="0" \
    RERANKER_FORMAT="cohere" \
    CHUNKER_MODE="regex" \
    HOST="0.0.0.0" \
    PORT="8000"

# Switch to non-root user
USER context7

# Expose server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

# Default command: run as server
ENTRYPOINT ["context7-reranker"]
CMD ["server", "--host", "0.0.0.0", "--port", "8000"]
