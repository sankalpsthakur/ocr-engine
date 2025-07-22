# Surya OCR Quick Start Guide

## Overview
This project provides a high-accuracy OCR service using Surya OCR with a FastAPI interface, Docker deployment, and comprehensive evaluation tools.

## Key Features
- **FastAPI Server**: RESTful API with single and batch OCR endpoints
- **Docker Support**: Production-ready containerization
- **WebP Handling**: Automatic detection and conversion of WebP images
- **Evaluation Tools**: Comprehensive CER (Character Error Rate) testing
- **Port 8080**: Configured for standard deployment

## Quick Start

### 1. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server
python api/main.py

# Test the API
python api/test_api.py
```

### 2. Docker Deployment

```bash
# Build and run with docker-compose
docker-compose up -d

# Check logs
docker-compose logs -f

# Run deployment tests
python test/test_deployment.py
```

### 3. API Endpoints

- `GET /health` - Health check
- `POST /ocr` - Process single image/PDF
- `POST /ocr/batch` - Process multiple files (max 10)

### 4. Evaluation Tools

```bash
# Fix image format issues (WebP → PNG)
python test/fix_image_formats.py

# Run comprehensive evaluation
python test/comprehensive_evaluation.py

# Run deployment tests
python test/test_deployment.py
```

## Important Notes

1. **SEWA.png Issue**: Fixed! Was a WebP file with .png extension causing segmentation faults
2. **Performance**: OCR processing can be slow (~30s per image) 
3. **Target**: Achieving <2% CER on utility bills
4. **Port**: Always configured to use port 8080

## Project Structure

```
surya/
├── api/
│   ├── main.py          # FastAPI server
│   └── test_api.py      # API tests
├── test/
│   ├── comprehensive_evaluation.py  # Full CER evaluation
│   ├── fix_image_formats.py        # WebP→PNG converter
│   └── test_deployment.py          # Deployment tests
├── test_bills/           # Original test images
├── synthetic_test_bills/ # Degraded test images
├── Dockerfile           # Container definition
└── docker-compose.yml   # Multi-container setup
```