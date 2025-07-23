# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OCR Engine combining Surya OCR (99.9% accuracy) with Qwen Vision-Language models for structured data extraction from utility bills. Achieves <2% Character Error Rate (CER) on DEWA bills with enhanced spatial reasoning capabilities.

## Common Commands

### Development Setup (Microservices Architecture)

The OCR Engine uses a microservices architecture with separate virtual environments due to conflicting dependencies:

```bash
# 1. Clone repository and setup
git clone https://github.com/sankalpsthakur/ocr-engine.git
cd ocr-engine
git pull origin main

# 2. Create separate virtual environments for microservices
python3 -m venv gateway_env
python3 -m venv surya_env  
python3 -m venv qwen_env

# 3. Install dependencies in each environment
source gateway_env/bin/activate && pip install -r gateway_requirements.txt && deactivate
source surya_env/bin/activate && pip install -r services/surya/requirements.txt && deactivate
source qwen_env/bin/activate && pip install -r services/qwen/requirements.txt && deactivate

# 4. Create required temp directory
mkdir -p /tmp/surya_ocr_api
```

### Running the Microservices

**Option 1: Automated startup (Recommended)**
```bash
# Start all microservices automatically
./start_api.sh
```

**Option 2: Manual startup in separate terminals**
```bash
# Terminal 1: Start Surya OCR service (port 8001)
source surya_env/bin/activate
python services/surya/surya_service.py

# Terminal 2: Start Qwen VL service (port 8002)  
source qwen_env/bin/activate
python services/qwen/qwen_service.py

# Terminal 3: Start API Gateway (port 8080)
source gateway_env/bin/activate
python api_gateway.py
```

### Docker Deployment
```bash
# Build Docker image for microservices
docker build -t ocr-engine .

# Run with docker-compose
docker compose up
```

### Testing
```bash
# Run comprehensive CER evaluation
python test/comprehensive_evaluation.py

# Test all API endpoints
python test/test_api.py

# Test deployment readiness on port 8080
python test/test_deployment.py

# Test Qwen VL
python test/test_qwen_vl.py

# Run a single test function
python -m pytest test/test_api.py::test_single_ocr -v
```

### API Testing with curl
```bash
# Health check
curl http://localhost:8080/health

# Basic OCR (raw text extraction)
curl -X POST -F 'file=@test_bills/DEWA.png' http://localhost:8080/ocr

# Qwen VL processing with spatial reasoning (PRIMARY TEST)
curl -X POST -F 'file=@test_bills/DEWA.png' http://localhost:8080/ocr/qwen-vl/process

# Process bill with specific resource type and reasoning
curl -X POST -F 'file=@test_bills/DEWA.png' \
"http://localhost:8080/ocr/qwen-vl/process?resource_type=water&enable_reasoning=true"

# Process with energy schema (includes carbon emissions)
curl -X POST -F 'file=@test_bills/DEWA.png' \
"http://localhost:8080/ocr/qwen-vl/process?resource_type=energy&enable_reasoning=false"

# Get DEWA/SEWA schema
curl http://localhost:8080/ocr/qwen-vl/schema/DEWA
```

### Complete Setup & Testing Flow
```bash
# 1. Clone and setup
git clone https://github.com/sankalpsthakur/ocr-engine.git
cd ocr-engine
git pull origin main

# 2. Create separate virtual environments for microservices
python3 -m venv gateway_env
python3 -m venv surya_env  
python3 -m venv qwen_env

# 3. Install dependencies in each environment
source gateway_env/bin/activate && pip install -r gateway_requirements.txt && deactivate
source surya_env/bin/activate && pip install -r services/surya/requirements.txt && deactivate
source qwen_env/bin/activate && pip install -r services/qwen/requirements.txt && deactivate

# 4. Start all microservices locally
./start_api.sh

# 5. Test Docker build and deployment
docker build -t ocr-engine .
docker compose up

# 6. Test API endpoints (local or Docker)
# Basic OCR (raw text extraction)
curl -X POST -F 'file=@test_bills/DEWA.png' http://localhost:8080/ocr

# Qwen VL processing with spatial reasoning (PRIMARY TEST)
curl -X POST -F 'file=@test_bills/DEWA.png' http://localhost:8080/ocr/qwen-vl/process

# Process bill as water resource type
curl -X POST -F 'file=@test_bills/DEWA.png' \
"http://localhost:8080/ocr/qwen-vl/process?resource_type=water&enable_reasoning=true"

# Process with energy schema (includes carbon emissions)
curl -X POST -F 'file=@test_bills/DEWA.png' \
"http://localhost:8080/ocr/qwen-vl/process?resource_type=energy&enable_reasoning=false"
```

### Cleanup
```bash
# Remove junk files
./cleanup_junk.sh
```

## Architecture Overview

### Three-Stage Processing Pipeline

1. **Surya OCR** - High-accuracy text extraction
   - Achieves 99.9% accuracy on utility bills
   - Handles Arabic and English text
   - Post-processing fixes common OCR artifacts

2. **Qwen Vision-Language Model** - Spatial understanding
   - Analyzes document layout and structure
   - Identifies tables, sections, and hierarchies
   - Detects and corrects OCR errors using visual context

3. **Structured Extraction** - Schema mapping
   - Maps unstructured text to Pydantic models
   - Validates data types and formats
   - Falls back to regex when VLM fails

### API Endpoints

```
Port 8080:
├── /health                           # API health check
├── /ocr                             # Basic OCR endpoint
├── /ocr/batch                       # Batch processing (max 10 files)
├── /ocr/qwen-vl/
│   ├── /health                      # Qwen VL service health
│   ├── /process                     # Full pipeline with spatial reasoning
│   ├── /extract-text                # OCR with post-processing only
│   └── /schema/{provider}           # Get DEWA/SEWA schemas
└── Legacy endpoints (deprecated):
    └── /ocr/qwen/*                  # Use /ocr/qwen-vl/* instead
```

### Key Components

- **API Server** (`api/main.py`): FastAPI with auto-loaded extensions
- **Qwen VL Processor** (`api/extractors/qwen_vl_processor.py`): Vision-language processing
- **Spatial Reasoner** (`api/extractors/spatial_reasoner.py`): Layout understanding
- **OCR Post-processor** (`api/utils/ocr_postprocessor.py`): Text cleanup
- **Pydantic Models** (`api/models/`): Type-safe bill schemas
- **Ground Truth** (`benchmark_output_ground_truth/`): Evaluation data

## Performance & Requirements

### Performance Metrics
- **CER**: 0.00% on DEWA bills, <2% target maintained
- **Processing Time**: 3-10s (GPU), 30-180s (CPU)
- **Memory**: ~3GB with 4-bit quantization
- **File Size Limit**: 50MB
- **Batch Limit**: 10 files

### System Requirements
- **Python**: 3.10+ required
- **GPU**: 4-8GB VRAM recommended (optional)
- **RAM**: 8-16GB minimum
- **Port**: Must use 8080 for deployment

### Environment Variables
```bash
PORT=8080
PYTHONUNBUFFERED=1
TRANSFORMERS_CACHE=/app/.cache/huggingface
HF_HOME=/app/.cache/huggingface
TORCH_HOME=/app/.cache/torch
HF_HUB_DISABLE_SYMLINKS_WARNING=1
HF_HUB_OFFLINE=0
```

## Known Issues & Solutions

1. **SEWA OCR Hanging**: Surya hangs at ~52% on SEWA files
   - Workaround: Use image preprocessing
   - Root cause: Model-specific issue with SEWA format

2. **First Request Timeout**: Initial model loading can timeout
   - Solution: Models are pre-loaded on startup
   - Alternative: Increase health check start period

3. **Temp Directory Error**: OCR fails if `/tmp/surya_ocr_api` missing
   - Solution: Directory created automatically on startup

## Critical Implementation Details

### Ground Truth Alignment
- Text must match Surya's natural extraction order
- Post-processing essential for <2% CER
- Evaluation uses normalized text comparison

### Deployment Configuration
- **Railway**: Configured with nixpacks.toml
- **Docker**: Multi-stage build optimized for size
- **Health Checks**: 5-minute start period for model loading

### Testing Strategy
- Always test against `benchmark_output_ground_truth/`
- Include both original and synthetic degraded images
- Track CER, WER, and field-level accuracy
- Verify deployment on port 8080 before pushing

### Bill Provider Details
- **DEWA**: Dubai Electricity & Water Authority (fully working)
- **SEWA**: Sharjah Electricity & Water Authority (OCR issues)
- Auto-detection based on filename patterns or content analysis
- Provider-specific validation and field mapping