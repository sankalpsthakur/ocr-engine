#!/usr/bin/env python3
"""Test script for Qwen VL integration"""

import requests
import json
from pathlib import Path
import sys


def test_health_check():
    """Test Qwen VL health endpoint"""
    print("Testing Qwen VL health check...")
    response = requests.get("http://localhost:8080/ocr/qwen-vl/health")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Health check passed: {json.dumps(data, indent=2)}")
        return True
    else:
        print(f"✗ Health check failed: {response.status_code}")
        return False


def test_schema_endpoints():
    """Test schema retrieval endpoints"""
    print("\nTesting schema endpoints...")
    
    for provider in ["DEWA", "SEWA"]:
        response = requests.get(f"http://localhost:8080/ocr/qwen-vl/schema/{provider}")
        
        if response.status_code == 200:
            schema = response.json()
            print(f"✓ {provider} schema retrieved successfully")
            print(f"  Fields: {list(schema.get('properties', {}).keys())[:5]}...")
        else:
            print(f"✗ Failed to get {provider} schema: {response.status_code}")


def test_ocr_extraction(test_file: str):
    """Test OCR text extraction"""
    print(f"\nTesting OCR extraction on {test_file}...")
    
    with open(test_file, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            "http://localhost:8080/ocr/qwen-vl/extract-text",
            files=files,
            params={'apply_postprocessing': True}
        )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ OCR extraction successful")
        print(f"  Confidence: {data.get('confidence', 'N/A')}")
        print(f"  Processing time: {data.get('processing_time', 'N/A')}s")
        print(f"  Text preview: {data['text'][:100]}...")
        return data['text']
    else:
        print(f"✗ OCR extraction failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return None


def test_qwen_vl_processing(test_file: str, provider: str = None):
    """Test full Qwen VL processing"""
    print(f"\nTesting Qwen VL processing on {test_file}...")
    
    with open(test_file, 'rb') as f:
        files = {'file': f}
        params = {'enable_reasoning': True}
        if provider:
            params['provider'] = provider
        
        response = requests.post(
            "http://localhost:8080/ocr/qwen-vl/process",
            files=files,
            params=params
        )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Qwen VL processing successful")
        print(f"  Provider: {data['provider']}")
        print(f"  Extraction method: {data['extraction_method']}")
        print(f"  Processing time: {data.get('processing_time', 'N/A')}s")
        
        # Print key extracted fields
        extracted = data['extracted_data']
        print("\n  Extracted fields:")
        for field in ['account_number', 'bill_date', 'total_amount', 'electricity_kwh']:
            if field in extracted:
                print(f"    {field}: {extracted[field]}")
        
        return data
    else:
        print(f"✗ Qwen VL processing failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return None


def main():
    """Run all tests"""
    print("=" * 60)
    print("Qwen VL Integration Test Suite")
    print("=" * 60)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8080/health")
        if response.status_code != 200:
            print("ERROR: API server is not running on port 8080")
            print("Please run: ./start_api.sh")
            sys.exit(1)
    except requests.ConnectionError:
        print("ERROR: Cannot connect to API server on port 8080")
        print("Please run: ./start_api.sh")
        sys.exit(1)
    
    # Run tests
    test_health_check()
    test_schema_endpoints()
    
    # Test with DEWA bill
    dewa_file = Path("test_bills/DEWA.png")
    if dewa_file.exists():
        print("\n" + "=" * 40)
        print("Testing DEWA bill processing")
        print("=" * 40)
        
        ocr_text = test_ocr_extraction(str(dewa_file))
        if ocr_text:
            test_qwen_vl_processing(str(dewa_file), "DEWA")
    else:
        print(f"\nWarning: {dewa_file} not found")
    
    # Test with SEWA bill (if not hanging)
    sewa_file = Path("test_bills/SEWA.png")
    if sewa_file.exists():
        print("\n" + "=" * 40)
        print("Testing SEWA bill processing")
        print("=" * 40)
        print("Note: SEWA processing may hang due to known Surya OCR issue")
        
        # Only test if user confirms
        # test_qwen_vl_processing(str(sewa_file), "SEWA")
    
    print("\n" + "=" * 60)
    print("Test suite completed")
    print("=" * 60)


if __name__ == "__main__":
    main()