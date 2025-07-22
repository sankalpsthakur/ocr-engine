#!/usr/bin/env python3
"""Test script to verify deployment fixes for Qwen VL integration"""

import subprocess
import sys
import time
import requests
import os

def test_import_dependencies():
    """Test if all required dependencies can be imported"""
    print("Testing import of critical dependencies...")
    
    failed_imports = []
    
    # Test core dependencies
    dependencies = [
        ("qwen_vl_utils", "qwen-vl-utils"),
        ("transformers", "transformers>=4.45.0"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("bitsandbytes", "bitsandbytes"),
        ("surya_ocr", "surya-ocr"),
    ]
    
    for module_name, pip_name in dependencies:
        try:
            __import__(module_name)
            print(f"✓ {module_name} imported successfully")
        except ImportError:
            print(f"✗ {module_name} import failed - install with: pip install {pip_name}")
            failed_imports.append(pip_name)
    
    if failed_imports:
        print("\nMissing dependencies:")
        for dep in failed_imports:
            print(f"  - {dep}")
        return False
    
    print("\nAll dependencies imported successfully!")
    return True

def test_api_startup():
    """Test if API starts up with Qwen extensions"""
    print("\nTesting API startup...")
    
    # Start API server
    env = os.environ.copy()
    env["PORT"] = "8081"  # Use different port to avoid conflicts
    
    process = subprocess.Popen(
        ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8081"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        env=env
    )
    
    # Wait for startup and capture logs
    startup_logs = []
    start_time = time.time()
    qwen_loaded = False
    
    while time.time() - start_time < 30:  # 30 second timeout
        line = process.stdout.readline()
        if line:
            startup_logs.append(line.strip())
            print(f"  {line.strip()}")
            
            if "Qwen VL extensions loaded successfully" in line:
                qwen_loaded = True
            elif "Qwen VL endpoints will not be available" in line:
                print("\n✗ Qwen VL extensions failed to load")
                break
            elif "Application startup complete" in line or "Uvicorn running on" in line:
                break
    
    # Test health endpoint
    try:
        time.sleep(2)  # Give it a moment to be ready
        response = requests.get("http://localhost:8081/health", timeout=5)
        print(f"\nHealth check: {response.status_code}")
        
        if qwen_loaded:
            # Test Qwen health endpoint
            qwen_response = requests.get("http://localhost:8081/ocr/qwen-vl/health", timeout=5)
            print(f"Qwen VL health check: {qwen_response.status_code}")
            if qwen_response.status_code == 200:
                print("✓ Qwen VL endpoints are available")
            else:
                print("✗ Qwen VL health check failed")
    except Exception as e:
        print(f"Error checking endpoints: {e}")
    
    # Cleanup
    process.terminate()
    process.wait()
    
    return qwen_loaded

def main():
    """Run all tests"""
    print("=== Testing Deployment Fixes ===\n")
    
    # Test 1: Import dependencies
    imports_ok = test_import_dependencies()
    
    if not imports_ok:
        print("\nDependencies missing. Please run:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    # Test 2: API startup
    api_ok = test_api_startup()
    
    if api_ok:
        print("\n✓ All tests passed! API starts with Qwen VL extensions.")
        print("\nDeployment should now work correctly.")
    else:
        print("\n✗ API startup test failed. Check the logs above for errors.")
        sys.exit(1)

if __name__ == "__main__":
    main()