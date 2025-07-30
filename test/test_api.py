#!/usr/bin/env python3
"""Test script for the FastAPI OCR service"""

import requests
import time
from pathlib import Path

API_BASE = "http://localhost:8080"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{API_BASE}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_single_ocr():
    """Test single file OCR"""
    print("Testing single file OCR...")
    
    # Use DEWA.png as test file
    test_file = Path(__file__).parent.parent / "test_bills" / "DEWA.png"
    
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return
    
    with open(test_file, 'rb') as f:
        files = {'file': (test_file.name, f, 'image/png')}
        start_time = time.time()
        response = requests.post(f"{API_BASE}/ocr", files=files)
        end_time = time.time()
    
    print(f"Status: {response.status_code}")
    print(f"Processing time: {end_time - start_time:.2f}s")
    
    if response.status_code == 200:
        result = response.json()
        print(f"OCR Status: {result['status']}")
        if result['text']:
            print(f"Text length: {len(result['text'])} characters")
            print(f"First 200 chars: {result['text'][:200]}...")
        if result.get('confidence'):
            print(f"Confidence: {result['confidence']:.2f}")
    else:
        print(f"Error: {response.text}")
    print()

def test_pdf_ocr():
    """Test PDF file OCR"""
    print("Testing PDF file OCR...")
    
    # Use Bill-4.pdf as test file
    test_file = Path(__file__).parent.parent / "test_bills" / "Bill-4.pdf"
    
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return
    
    with open(test_file, 'rb') as f:
        files = {'file': (test_file.name, f, 'application/pdf')}
        start_time = time.time()
        response = requests.post(f"{API_BASE}/ocr", files=files)
        end_time = time.time()
    
    print(f"Status: {response.status_code}")
    print(f"Processing time: {end_time - start_time:.2f}s")
    
    if response.status_code == 200:
        result = response.json()
        print(f"OCR Status: {result['status']}")
        if result['text']:
            print(f"Text length: {len(result['text'])} characters")
            # Check for multi-page markers
            if "[Page " in result['text']:
                print("Multi-page PDF detected")
                # Count pages
                page_count = result['text'].count("[Page ")
                print(f"Number of pages: {page_count}")
            print(f"First 200 chars: {result['text'][:200]}...")
        if result.get('confidence'):
            print(f"Confidence: {result['confidence']:.2f}")
    else:
        print(f"Error: {response.text}")
    print()

def test_batch_ocr():
    """Test batch OCR"""
    print("Testing batch OCR...")
    
    test_files = [
        Path(__file__).parent.parent / "test_bills" / "DEWA.png",
        Path(__file__).parent.parent / "test_bills" / "SEWA.png"
    ]
    
    files_data = []
    for test_file in test_files:
        if test_file.exists():
            files_data.append(('files', (test_file.name, open(test_file, 'rb'), 'image/png')))
    
    if not files_data:
        print("No test files found")
        return
    
    start_time = time.time()
    response = requests.post(f"{API_BASE}/ocr/batch", files=files_data)
    end_time = time.time()
    
    # Close files
    for _, (_, f, _) in files_data:
        f.close()
    
    print(f"Status: {response.status_code}")
    print(f"Processing time: {end_time - start_time:.2f}s")
    
    if response.status_code == 200:
        results = response.json()
        print(f"Processed {len(results)} files")
        for result in results:
            print(f"  - {result['filename']}: {result['status']}")
    else:
        print(f"Error: {response.text}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("FastAPI OCR Service Tests")
    print("=" * 60)
    print()
    
    # Check if API is running
    try:
        test_health()
        test_single_ocr()
        test_pdf_ocr()
        test_batch_ocr()
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to API at http://localhost:8080")
        print("Make sure the API is running: python api/main.py")

if __name__ == "__main__":
    main()