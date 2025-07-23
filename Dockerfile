# Multi-stage build for Surya OCR API - Optimized for Railway
FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /build

# Copy requirements
COPY requirements.txt .

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
COPY qwen_vl_integration/ /app/qwen_vl_integration/
COPY test/ocr_postprocessing.py /app/

# Ensure __init__.py files exist for proper module imports
RUN touch /app/__init__.py && \
    find /app/qwen_vl_integration -type d -exec touch {}/__init__.py \;

# Create temp and cache directories
RUN mkdir -p /tmp/surya_ocr_api /app/.cache/huggingface /app/.cache/torch && \
    chown -R appuser:appuser /app /tmp/surya_ocr_api

# Switch to non-root user
USER appuser

# Create a minimal model validation script that doesn't download
RUN echo '#!/usr/bin/env python3\n\
import os\n\
import sys\n\
os.environ["HF_HOME"] = "/app/.cache/huggingface"\n\
os.environ["TRANSFORMERS_CACHE"] = "/app/.cache/huggingface"\n\
os.environ["TORCH_HOME"] = "/app/.cache/torch"\n\
print("Environment configured for model downloads")\n\
print("Models will be downloaded on first request to keep image size small")\n\
# Validate imports only\n\
try:\n\
    import surya\n\
    import transformers\n\
    import torch\n\
    print("✓ All required packages installed successfully")\n\
except ImportError as e:\n\
    print(f"ERROR: Missing package: {e}")\n\
    sys.exit(1)\n\
' > /app/validate_install.py && \
    python /app/validate_install.py

# Expose port (Railway will set PORT env var)
EXPOSE ${PORT:-8080}

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/api \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    HF_HUB_OFFLINE=0

# Health check - use Railway's PORT
HEALTHCHECK --interval=30s --timeout=30s --start-period=300s --retries=10 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", \"8080\")}/health').read()"

# Run the application - use Railway's PORT
CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]