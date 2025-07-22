#!/bin/bash

# Test script for production deployment on Railway

echo "=================================================================================="
echo "Testing OCR Engine on Railway Production"
echo "URL: https://ocr-engine-production.up.railway.app"
echo "=================================================================================="

URL="https://ocr-engine-production.up.railway.app"

# Test 1: Health check
echo -e "\n1. Testing /health endpoint..."
curl -s "$URL/health" | jq '.' || echo "Failed to get health status"

# Test 2: Qwen VL health check
echo -e "\n2. Testing /ocr/qwen-vl/health endpoint..."
curl -s "$URL/ocr/qwen-vl/health" | jq '.' || echo "Qwen VL health endpoint not found"

# Test 3: Schema endpoints
echo -e "\n3. Testing schema endpoints..."
echo "   - DEWA schema:"
curl -s "$URL/ocr/qwen-vl/schema/DEWA" | jq '.' | head -20 || echo "DEWA schema not available"

echo "   - SEWA schema:"
curl -s "$URL/ocr/qwen-vl/schema/SEWA" | jq '.' | head -20 || echo "SEWA schema not available"

# Test 4: Basic OCR (if test file exists)
if [ -f "test_bills/DEWA.png" ]; then
    echo -e "\n4. Testing basic OCR endpoint..."
    echo "   Sending DEWA.png..."
    time curl -X POST -F "file=@test_bills/DEWA.png" "$URL/ocr" | jq '.' | head -50
else
    echo -e "\n4. Skipping OCR test - no test file found"
fi

# Test 5: Qwen VL process endpoint (if test file exists)
if [ -f "test_bills/DEWA.png" ]; then
    echo -e "\n5. Testing Qwen VL process endpoint..."
    echo "   Sending DEWA.png..."
    time curl -X POST -F "file=@test_bills/DEWA.png" "$URL/ocr/qwen-vl/process" | jq '.' | head -50
else
    echo -e "\n5. Skipping Qwen VL test - no test file found"
fi

echo -e "\n=================================================================================="
echo "Test complete!"
echo "=================================================================================="