# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an OCR Engine that combines Surya OCR for text extraction with Qwen3-0.6B LLM for structured data extraction from utility bills (DEWA and SEWA). The system achieves <2% Character Error Rate (CER) on DEWA bills.

## Common Commands

### Development Setup
```bash
# Create and activate virtual environment
python3 -m venv ocr_env
source ocr_env/bin/activate

# Install dependencies
pip install -r qwen_integration/requirements.txt

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
# Run comprehensive evaluation (CER testing)
python test/comprehensive_evaluation.py

# Test API endpoints
python test_api.py

# Test OCR processing
python test_ocr_endpoint.py

# Test deployment on port 8080
python test/test_deployment.py

# Generate clean OCR outputs
python test/generate_clean_ocr_outputs.py
```

### API Endpoints Testing
```bash
# Health check
curl http://localhost:8080/health

# Basic OCR (raw text)
curl -X POST -F 'file=@test_bills/DEWA.png' http://localhost:8080/ocr

# Qwen-enhanced processing (structured extraction)
curl -X POST -F 'file=@test_bills/DEWA.png' http://localhost:8080/ocr/qwen/process

# Get DEWA/SEWA schema
curl http://localhost:8080/ocr/qwen/schema/DEWA
```

## Architecture Overview

### Two-Stage OCR Pipeline
1. **Surya OCR**: Extracts raw text from images/PDFs
   - Located in: `qwen_integration/src/extractors/surya_extractor.py`
   - Handles post-processing to fix common OCR artifacts
   - Includes image preprocessing for complex files

2. **Qwen3-0.6B LLM**: Extracts structured data from raw text
   - Located in: `qwen_integration/src/extractors/qwen_processor.py`
   - Uses Pydantic models for type-safe validation
   - Includes fallback regex extraction when LLM fails

### Key Components
- **API Server** (`api/main.py`): FastAPI server on port 8080
- **OCR Pipeline** (`qwen_integration/src/main.py`): Main orchestration
- **Pydantic Models** (`qwen_integration/src/models/`): DEWA/SEWA bill schemas
- **Ground Truth** (`benchmark_output_ground_truth/`): Manually aligned test data

### API Structure
```
Port 8080:
├── /health                    # Health check
├── /ocr                      # Basic OCR endpoint
├── /ocr/batch               # Batch processing (max 10 files)
└── /ocr/qwen/
    ├── /health              # Qwen service health
    ├── /process             # Full pipeline with structured extraction
    ├── /extract-text        # OCR with optional post-processing
    └── /schema/{provider}   # Get DEWA/SEWA schemas
```

## Important Technical Details

### Performance Targets
- **CER Target**: <2% (achieved 0.00% on DEWA bills)
- **Processing Time**: 30-180 seconds per image
- **File Size Limit**: 50MB
- **Batch Limit**: 10 files

### Known Issues
1. **SEWA OCR Hanging**: Surya OCR hangs at ~52% when processing SEWA files
   - Preprocessing implemented but insufficient
   - Appears to be model-specific issue with SEWA bill format

2. **Account Number Extraction**: Fixed - now correctly extracts customer account number instead of invoice number

3. **Temp Directory**: Must exist at `/tmp/surya_ocr_api` or OCR will fail

### Critical Requirements
- **Port 8080**: Deployment must use port 8080
- **Python 3.10+**: Required for compatibility
- **Post-processing**: Essential for achieving <2% CER
- **Ground Truth Alignment**: Text must match Surya's extraction order

### Deployment Configuration
- **Docker**: Dockerfile with multi-stage build
- **Railway**: `railway.toml` and `nixpacks.toml` configured
- **Environment Variables**:
  ```
  PORT=8080
  PYTHONUNBUFFERED=1
  TRANSFORMERS_CACHE=/app/.cache/huggingface
  ```

## Working with the Codebase

### File Organization Best Practices
- **Do not create multiple junk files** - work with existing files
- Clean up temporary files after use
- Keep one comprehensive test file in `/test` subdirectory

### Testing Approach
1. Always test against ground truth in `benchmark_output_ground_truth/`
2. Test both original and synthetic degraded images
3. Track CER, WER, and field accuracy metrics
4. Run deployment testing to verify port 8080

### Handling Utility Bills
- **DEWA**: Dubai Electricity & Water Authority (working perfectly)
- **SEWA**: Sharjah Electricity & Water Authority (has OCR issues)
- Auto-detection based on filename or content
- Provider-specific Pydantic models for validation

### Important Files
- `test/ocr_postprocessing.py`: Core post-processing logic
- `test/comprehensive_evaluation.py`: Main evaluation script
- `qwen_integration/src/extractors/qwen_processor.py`: Fixed account number extraction
- `benchmark_output_ground_truth/fields_ground_truth.json`: Field-level ground truth

## Surya OCR Installation

```bash
pip install surya-ocr
```

Model weights will automatically download the first time you run surya.