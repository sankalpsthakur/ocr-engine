#!/usr/bin/env python3
"""Deployment testing script for FastAPI on port 8080"""

import requests
import time
import sys
from pathlib import Path

def test_port_8080():
    """Test that the API is running on port 8080"""
    print("Testing API deployment on port 8080...")
    
    api_url = "http://localhost:8080"
    max_retries = 5
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✓ API is running on port 8080")
                health_data = response.json()
                print(f"  Status: {health_data['status']}")
                print(f"  Version: {health_data['version']}")
                return True
            else:
                print(f"✗ Unexpected status code: {response.status_code}")
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                print(f"  Connection failed, retrying in {retry_delay}s... ({attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"✗ Cannot connect to API on port 8080")
                return False
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    return False

def test_ocr_endpoint():
    """Test the OCR endpoint"""
    print("\nTesting OCR endpoint...")
    
    # Find a test image
    test_file = None
    possible_paths = [
        Path(__file__).parent.parent / "test_bills" / "DEWA.png",
        Path("/app/test_bills/DEWA.png"),  # Docker mount path
    ]
    
    for path in possible_paths:
        if path.exists():
            test_file = path
            break
    
    if not test_file:
        print("✗ No test file found")
        return False
    
    api_url = "http://localhost:8080"
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file.name, f, 'image/png')}
            response = requests.post(f"{api_url}/ocr", files=files, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if result['status'] == 'success':
                print(f"✓ OCR endpoint working")
                print(f"  File: {result['filename']}")
                print(f"  Text extracted: {len(result.get('text', ''))} characters")
                if result.get('processing_time'):
                    print(f"  Processing time: {result['processing_time']:.2f}s")
                return True
            else:
                print(f"✗ OCR failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"✗ HTTP {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_batch_endpoint():
    """Test the batch OCR endpoint"""
    print("\nTesting batch OCR endpoint...")
    
    api_url = "http://localhost:8080"
    
    # Create a simple test image in memory
    from PIL import Image
    import io
    
    # Create a simple white image
    img = Image.new('RGB', (100, 100), color='white')
    img_buffer = io.BytesIO()
    img.save(img_buffer, 'PNG')
    img_buffer.seek(0)
    
    try:
        files = [
            ('files', ('test1.png', img_buffer, 'image/png')),
            ('files', ('test2.png', img_buffer.getvalue(), 'image/png'))
        ]
        
        response = requests.post(f"{api_url}/ocr/batch", files=files, timeout=120)
        
        if response.status_code == 200:
            results = response.json()
            if len(results) == 2:
                print(f"✓ Batch endpoint working")
                print(f"  Processed {len(results)} files")
                return True
            else:
                print(f"✗ Expected 2 results, got {len(results)}")
                return False
        else:
            print(f"✗ HTTP {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all deployment tests"""
    print("=" * 60)
    print("SURYA OCR API DEPLOYMENT TESTS")
    print("=" * 60)
    print()
    
    tests = [
        ("Port 8080 availability", test_port_8080),
        ("OCR endpoint", test_ocr_endpoint),
        ("Batch OCR endpoint", test_batch_endpoint)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n[{len(results) + 1}/{len(tests)}] {test_name}")
        print("-" * 40)
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("DEPLOYMENT TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}")
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status} - {test_name}")
    
    if passed == total:
        print("\n✓ All deployment tests passed! API is ready for production.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the logs.")
        return 1

if __name__ == "__main__":
    sys.exit(main())