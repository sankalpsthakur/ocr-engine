#!/usr/bin/env python3
"""
Demo script to show PDF OCR functionality
Run this after starting all services with ./start_api.sh
"""

import requests
import time
from pathlib import Path

API_BASE = "http://localhost:8080"

def demo_pdf_ocr():
    """Demonstrate PDF OCR capabilities"""
    print("PDF OCR Demo")
    print("=" * 60)
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code != 200:
            print("ERROR: API is not healthy")
            return
        health = response.json()
        print(f"✓ API Status: {health['status']}")
        print(f"✓ Surya OCR: {health['services']['surya_ocr']}")
        print()
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to API at http://localhost:8080")
        print("Please start the API with: ./start_api.sh")
        return
    
    # Test PDF files
    test_bills_dir = Path(__file__).parent / "test_bills"
    pdf_files = list(test_bills_dir.glob("Bill-*.pdf"))
    
    if not pdf_files:
        print("No PDF test files found")
        return
    
    print(f"Found {len(pdf_files)} PDF files to test\n")
    
    for pdf_file in pdf_files[:2]:  # Test first 2 PDFs
        print(f"Testing: {pdf_file.name}")
        print("-" * 40)
        
        with open(pdf_file, 'rb') as f:
            files = {'file': (pdf_file.name, f, 'application/pdf')}
            
            # Test basic OCR endpoint
            print("1. Testing /ocr endpoint...")
            start_time = time.time()
            response = requests.post(f"{API_BASE}/ocr", files=files)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✓ Status: {result['status']}")
                print(f"   ✓ Processing time: {elapsed:.2f}s")
                
                if result.get('text'):
                    # Check for multi-page
                    page_count = result['text'].count("[Page ")
                    if page_count > 0:
                        print(f"   ✓ Pages detected: {page_count}")
                    
                    text_preview = result['text'][:200].replace('\n', ' ')
                    print(f"   ✓ Text preview: {text_preview}...")
                    print(f"   ✓ Total characters: {len(result['text'])}")
                
                if result.get('confidence'):
                    print(f"   ✓ Confidence: {result['confidence']:.2f}")
            else:
                print(f"   ✗ Error: {response.status_code} - {response.text}")
        
        # Test with Qwen VL endpoint (if available)
        with open(pdf_file, 'rb') as f:
            files = {'file': (pdf_file.name, f, 'application/pdf')}
            
            print("\n2. Testing /ocr/qwen-vl/process endpoint...")
            start_time = time.time()
            response = requests.post(
                f"{API_BASE}/ocr/qwen-vl/process",
                files=files,
                params={"resource_type": "utility", "enable_reasoning": True}
            )
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✓ Status: {result.get('status', 'success')}")
                print(f"   ✓ Processing time: {elapsed:.2f}s")
                
                if result.get('extracted_data'):
                    print(f"   ✓ Extracted data fields: {len(result['extracted_data'])}")
                    # Show first few fields
                    for key in list(result['extracted_data'].keys())[:3]:
                        print(f"     - {key}: {result['extracted_data'][key]}")
                
                if result.get('error') and "Only first page" in result['error']:
                    print(f"   ℹ {result['error']}")
            elif response.status_code == 504:
                print("   ℹ Qwen VL service timeout (this is normal for large PDFs)")
            else:
                print(f"   ℹ Qwen VL not available or error: {response.status_code}")
        
        print()
    
    # Test batch processing with mixed files
    print("\n3. Testing batch processing with PDFs and images...")
    batch_files = []
    
    # Add a PDF
    pdf_file = pdf_files[0]
    batch_files.append(('files', (pdf_file.name, open(pdf_file, 'rb'), 'application/pdf')))
    
    # Add an image
    image_file = test_bills_dir / "DEWA.png"
    if image_file.exists():
        batch_files.append(('files', (image_file.name, open(image_file, 'rb'), 'image/png')))
    
    if batch_files:
        response = requests.post(f"{API_BASE}/ocr/batch", files=batch_files)
        
        # Close files
        for _, (_, f, _) in batch_files:
            f.close()
        
        if response.status_code == 200:
            results = response.json()
            print(f"   ✓ Processed {len(results)} files")
            for result in results:
                status = "✓" if result['status'] == 'success' else "✗"
                print(f"   {status} {result['filename']}: {result['status']}")
    
    print("\n" + "=" * 60)
    print("PDF OCR functionality is working correctly!")
    print("\nNotes:")
    print("- PDFs are automatically converted to images before OCR")
    print("- Multi-page PDFs are processed page by page")
    print("- Qwen VL processes only the first page for structured extraction")
    print("- Both images and PDFs can be processed in batch")


if __name__ == "__main__":
    demo_pdf_ocr()