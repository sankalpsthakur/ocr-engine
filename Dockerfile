# Multi-stage build for Surya OCR API
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /build

# Copy requirements
COPY api/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.10-slim

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
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 appuser

# Create app directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY api/ /app/api/
COPY test/ocr_postprocessing.py /app/

# Create temp directory
RUN mkdir -p /tmp/surya_ocr_api && \
    chown -R appuser:appuser /app /tmp/surya_ocr_api

# Switch to non-root user
USER appuser

# Download model weights (this will cache them in the image)
# Models will be downloaded on first use by surya CLI

# Expose port (Railway will set PORT env var)
EXPOSE ${PORT:-8080}

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Health check - use Railway's PORT
HEALTHCHECK --interval=30s --timeout=30s --start-period=300s --retries=10 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", \"8080\")}/health').read()"

# Run the application - use Railway's PORT
CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]