FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY qwen_integration/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Qwen model weights during build (optional - can be done at runtime)
# This helps with faster startup but increases image size
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    print('Downloading Qwen3-0.6B model...'); \
    AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True); \
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True, low_cpu_mem_usage=True)"

# Copy application code
COPY qwen_integration/ ./qwen_integration/
COPY api/ ./api/
COPY test_bills/ ./test_bills/
COPY benchmark_output_ground_truth/ ./benchmark_output_ground_truth/
COPY DEWA_Schema.json SEWA_Schema.json ./

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Create cache directory
RUN mkdir -p /app/.cache/huggingface

# Expose port for API
EXPOSE 8080

# Default command - can be overridden
CMD ["python", "-m", "api.main"]