# Multi-stage build for OCR Engine Microservices - Optimized for Railway
FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libmagic1 \
    libmagic-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /build

# Copy all requirements files
COPY gateway_requirements.txt .
COPY services/surya/requirements.txt surya_requirements.txt
COPY services/qwen/requirements.txt qwen_requirements.txt

# Install all Python dependencies in one environment (for container efficiency)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r gateway_requirements.txt && \
    pip install --no-cache-dir -r surya_requirements.txt && \
    pip install --no-cache-dir -r qwen_requirements.txt

# Final stage
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libmagic1 \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 appuser

# Create app directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy microservices code
COPY services/ /app/services/
COPY api_gateway.py /app/
COPY start_api.sh /app/
COPY docker_entrypoint.sh /app/

# Make scripts executable
RUN chmod +x /app/start_api.sh /app/docker_entrypoint.sh

# Create temp and cache directories
RUN mkdir -p /tmp/surya_ocr_api \
             /app/.cache/huggingface \
             /app/.cache/torch \
             /app/.cache/datalab/models \
             /home/appuser/.cache/datalab/models && \
    chown -R appuser:appuser /app /tmp/surya_ocr_api /home/appuser

# Switch to non-root user
USER appuser

# Expose port (Railway will set PORT env var)
EXPOSE ${PORT:-8080}

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch \
    SURYA_MODEL_CACHE_DIR=/home/appuser/.cache/datalab/models \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    HF_HUB_OFFLINE=0 \
    SURYA_SERVICE_URL=http://localhost:8001 \
    QWEN_SERVICE_URL=http://localhost:8002

# Health check - use Railway's PORT
HEALTHCHECK --interval=30s --timeout=30s --start-period=300s --retries=10 \
    CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

# Run the microservices
CMD ["/app/docker_entrypoint.sh"]