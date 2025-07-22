# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OCR Engine combining Surya OCR (99.9% accuracy) with Qwen Vision-Language models for structured data extraction from utility bills. Achieves <2% Character Error Rate (CER) on DEWA bills with enhanced spatial reasoning capabilities.

## Common Commands

### Development Setup
```bash
# Create and activate virtual environment
python3 -m venv ocr_env
source ocr_env/bin/activate

# Install all dependencies (includes Qwen VL)
pip install -r requirements.txt

# Create required temp directory
mkdir -p /tmp/surya_ocr_api
```

### Running the API Server
```bash
# Start API server on port 8080
./start_api.sh

# Or manually:
source ocr_env/bin/activate
python api/main.py

# Stop API server
if test -f api.pid; then kill $(cat api.pid); rm api.pid; fi
```

### Testing
```bash
# Run comprehensive CER evaluation
python test/comprehensive_evaluation.py

# Test all API endpoints
python test/test_api.py

# Test deployment readiness on port 8080
python test/test_deployment.py

# Test Qwen VL integration
python test/test_qwen_vl_integration.py

# Run a single test function
python -m pytest test/test_api.py::test_single_ocr -v
```

### API Testing with curl
```bash
# Health check
curl http://localhost:8080/health

# Basic OCR (raw text extraction)
curl -X POST -F 'file=@test_bills/DEWA.png' http://localhost:8080/ocr

# Qwen VL processing with spatial reasoning
curl -X POST -F 'file=@test_bills/DEWA.png' http://localhost:8080/ocr/qwen-vl/process

# Get DEWA/SEWA schema
curl http://localhost:8080/ocr/qwen-vl/schema/DEWA
```

### Docker Commands
```bash
# Build Docker image
docker build -t ocr-engine .

# Run container
docker run -p 8080:8080 -e PORT=8080 ocr-engine

# Run with docker-compose
docker-compose up
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
- **Qwen VL Processor** (`qwen_vl_integration/src/extractors/qwen_vl_processor.py`): Vision-language processing
- **Spatial Reasoner** (`qwen_vl_integration/src/extractors/spatial_reasoner.py`): Layout understanding
- **OCR Post-processor** (`qwen_vl_integration/src/utils/ocr_postprocessor.py`): Text cleanup
- **Pydantic Models** (`qwen_vl_integration/src/models/`): Type-safe bill schemas
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