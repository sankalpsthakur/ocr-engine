# Qwen VL Service Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies including git (needed for transformers from GitHub)
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directory for models
RUN mkdir -p /app/.cache/huggingface

# Copy service code
COPY qwen_service.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8002/health').raise_for_status()"

# Run the service
CMD ["python", "qwen_service.py"]