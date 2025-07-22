# Qwen Vision-Language Integration for OCR Engine

This module integrates Qwen 2.5-VL (Vision-Language) models with spatial reasoning capabilities to enhance OCR extraction accuracy for utility bills.

## Key Features

- **Spatial Reasoning**: Understands document layout, table structures, and visual hierarchies
- **OCR Error Correction**: Uses visual context to correct OCR mistakes
- **Structured Extraction**: Maps unstructured text to typed Pydantic schemas
- **High Accuracy**: Combines Surya OCR's 99.9% accuracy with VLM understanding
- **Efficient Processing**: 4-bit quantization support for reduced memory usage

## Architecture

### Two-Stage Processing Pipeline

1. **Spatial Understanding Stage**
   - Analyzes image to identify sections, tables, and hierarchies
   - Detects label-value relationships and alignments
   - Identifies potential OCR errors from visual context

2. **Structured Extraction Stage**
   - Uses spatial understanding to guide extraction
   - Maps OCR text to schema fields with visual grounding
   - Applies corrections and validates output

### Components

- `qwen_vl_processor.py`: Main VLM processor with reasoning chains
- `spatial_reasoner.py`: Parses and structures spatial understanding
- `regex_fallback.py`: Fallback extraction when VLM fails
- `prompt_builder.py`: Constructs optimized VLM prompts
- `models/`: Pydantic models for DEWA and SEWA bills

## API Endpoints

### `/ocr/qwen-vl/process` (POST)
Process utility bill with spatial reasoning.

**Parameters:**
- `file`: Image/PDF file (required)
- `enable_reasoning`: Enable spatial reasoning (default: true)
- `provider`: Provider type DEWA/SEWA (auto-detected if not provided)

**Response:**
```json
{
  "provider": "Dubai Electricity and Water Authority (DEWA)",
  "extracted_data": {
    "account_number": "2052672303",
    "bill_date": "21/05/2025",
    "total_amount": 287.00,
    "electricity_kwh": 299.0,
    "carbon_kg_co2e": 120.0
  },
  "extraction_method": "qwen_vl_with_reasoning",
  "confidence": 0.892,
  "processing_time": 4.5
}
```

### `/ocr/qwen-vl/health` (GET)
Check Qwen VL service health.

### `/ocr/qwen-vl/schema/{provider}` (GET)
Get Pydantic schema for DEWA or SEWA.

### `/ocr/qwen-vl/extract-text` (POST)
Extract OCR text only with post-processing.

## Installation

```bash
# Install Qwen VL dependencies
pip install -r requirements.txt
```

## Usage

```python
# The integration is automatically loaded when starting the API
./start_api.sh

# Test the integration
python test_qwen_vl_integration.py
```

## Memory Requirements

- **GPU (Recommended)**: 4-8GB VRAM with 4-bit quantization
- **CPU**: 8-16GB RAM (slower processing)

## Model Selection

The system automatically selects the appropriate model:
- GPU available: Qwen2-VL-2B-Instruct with optional quantization
- CPU only: Qwen2-VL-2B-Instruct without quantization

## Performance

- **Processing Time**: 3-10 seconds per image (GPU)
- **Accuracy**: Maintains <2% CER with enhanced field extraction
- **Memory Usage**: ~3GB with 4-bit quantization

## Troubleshooting

### Model Loading Issues
- Ensure sufficient memory (4GB+ GPU or 8GB+ RAM)
- Check transformers version >= 4.45.0
- Verify CUDA installation for GPU usage

### Extraction Errors
- Enable regex fallback for critical fields
- Check OCR post-processing is applied
- Verify image quality and orientation

### API Integration
- Ensure qwen_vl_integration is in Python path
- Check all dependencies are installed
- Verify port 8080 is available