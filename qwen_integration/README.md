# Qwen3-0.6B Integration for OCR Engine

This module enhances the Surya OCR engine with Qwen3-0.6B language model for structured data extraction from utility bills.

## Features

- **Structured Data Extraction**: Converts raw OCR text into validated JSON matching DEWA/SEWA schemas
- **Pydantic Validation**: Ensures data integrity with comprehensive field validation
- **High Accuracy**: Targets <2% Character Error Rate (CER)
- **API Integration**: REST endpoints for easy integration
- **Docker Support**: Containerized deployment with GPU/CPU support

## Installation

```bash
# Create virtual environment
python -m venv qwen_venv
source qwen_venv/bin/activate  # On Windows: qwen_venv\Scripts\activate

# Install dependencies
pip install -r qwen_integration/requirements.txt
```

## Quick Start

### Command Line Usage

```bash
# Process a single bill
python -m qwen_integration.src.main test_bills/DEWA.png

# Process multiple bills
python -m qwen_integration.src.main test_bills/*.png --output-dir results/

# Specify provider
python -m qwen_integration.src.main test_bills/bill.pdf --provider SEWA
```

### Python API

```python
from qwen_integration.src import OCRPipeline

# Create pipeline
pipeline = OCRPipeline()

# Process a bill
result = pipeline.process_bill("test_bills/DEWA.png")

# Access structured data
print(f"Account: {result.extracted_data.bill_info.account_number}")
print(f"Electricity: {result.extracted_data.consumption_data.electricity.value} kWh")
print(f"Confidence: {result.validation.confidence:.2%}")
```

## API Endpoints

Start the API server:

```bash
python -m api.main
```

### Qwen-Enhanced Endpoints

- `POST /ocr/qwen/process` - Process bill with structured extraction
- `POST /ocr/qwen/extract-text` - Extract only OCR text
- `GET /ocr/qwen/health` - Check Qwen service health
- `GET /ocr/qwen/schema/{provider}` - Get JSON schema for provider

### Example Request

```bash
curl -X POST "http://localhost:8080/ocr/qwen/process" \
  -F "file=@test_bills/DEWA.png"
```

## Docker Deployment

```bash
# Build image
docker-compose -f qwen_integration/docker-compose.yml build

# Run container
docker-compose -f qwen_integration/docker-compose.yml up

# With GPU support
docker-compose -f qwen_integration/docker-compose.yml up -e CUDA_VISIBLE_DEVICES=0
```

## Evaluation

Run comprehensive evaluation:

```bash
# Full evaluation
python qwen_integration/evaluate_qwen.py

# Quick test
python qwen_integration/evaluate_qwen.py --quick

# Specific files
python qwen_integration/evaluate_qwen.py --files test_bills/DEWA.png test_bills/SEWA.png
```

## Testing

```bash
# Run all tests
pytest qwen_integration/tests/ -v

# Run specific test module
pytest qwen_integration/tests/test_models.py -v

# With coverage
pytest qwen_integration/tests/ --cov=qwen_integration.src --cov-report=html
```

## Architecture

1. **Surya OCR**: Extracts raw text from images/PDFs
2. **Qwen3-0.6B**: Processes text to extract structured data
3. **Pydantic Models**: Validate and structure the output
4. **API Layer**: Provides REST endpoints for integration

## Performance

- **Processing Time**: <5 seconds per bill
- **Memory Usage**: ~2-3GB with Qwen3-0.6B
- **Accuracy**: <2% CER on test dataset
- **Confidence Scoring**: Automatic flagging for manual review

## Configuration

### Environment Variables

- `CUDA_VISIBLE_DEVICES`: GPU device ID (optional)
- `PORT`: API server port (default: 8080)
- `TRANSFORMERS_CACHE`: Model cache directory

### Model Selection

The default model is `Qwen/Qwen3-0.6B`. To use a different model:

```python
pipeline = OCRPipeline(model_name="Qwen/Qwen3-0.6B", device="cuda")
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Use CPU mode or reduce batch size
2. **Slow Processing**: Enable GPU support or use quantization
3. **Low Accuracy**: Check OCR quality, enable post-processing

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Create feature branch from `qwen-integration`
2. Add tests for new functionality
3. Ensure <2% CER is maintained
4. Update documentation

## License

Same as parent project