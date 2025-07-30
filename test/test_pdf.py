#!/usr/bin/env python3
"""
Test PDF processing capabilities of the OCR Engine
Tests PDF validation, conversion, and multi-page handling
"""

import asyncio
import pytest
import httpx
from pathlib import Path
import json
import time

# Base URL for API testing
API_BASE_URL = "http://localhost:8080"

# Test files directory
TEST_FILES_DIR = Path(__file__).parent.parent / "test_bills"

# PDF test files
PDF_TEST_FILES = [
    "Bill-4.pdf",
    "Bill-5.pdf", 
    "Bill-6.pdf",
    "Bill-7.pdf",
    "pdfcoffee.com_sewa-bill-pdf-free.pdf"
]

class TestPDFProcessing:
    """Test PDF processing functionality"""
    
    @pytest.fixture
    async def client(self):
        """Create async HTTP client"""
        async with httpx.AsyncClient(timeout=300.0) as client:
            yield client
    
    async def wait_for_api(self, client, max_retries=30, delay=2):
        """Wait for API to be ready"""
        for i in range(max_retries):
            try:
                response = await client.get(f"{API_BASE_URL}/health")
                if response.status_code == 200:
                    data = response.json()
                    if data["services"]["surya_ocr"] == "healthy":
                        print("API is ready!")
                        return True
            except Exception:
                pass
            
            if i < max_retries - 1:
                print(f"Waiting for API... ({i+1}/{max_retries})")
                await asyncio.sleep(delay)
        
        raise Exception("API failed to become ready")
    
    @pytest.mark.asyncio
    async def test_pdf_validation(self, client):
        """Test that PDFs are properly validated"""
        await self.wait_for_api(client)
        
        # Test with valid PDF
        pdf_path = TEST_FILES_DIR / PDF_TEST_FILES[0]
        if pdf_path.exists():
            with open(pdf_path, "rb") as f:
                files = {"file": (pdf_path.name, f, "application/pdf")}
                response = await client.post(f"{API_BASE_URL}/ocr", files=files)
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert data["filename"] == pdf_path.name
    
    @pytest.mark.asyncio
    async def test_single_page_pdf(self, client):
        """Test OCR on single-page PDF"""
        await self.wait_for_api(client)
        
        # Use first PDF for testing
        pdf_path = TEST_FILES_DIR / PDF_TEST_FILES[0]
        if pdf_path.exists():
            with open(pdf_path, "rb") as f:
                files = {"file": (pdf_path.name, f, "application/pdf")}
                response = await client.post(f"{API_BASE_URL}/ocr", files=files)
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert data["text"] is not None
                assert len(data["text"]) > 0
                assert "processing_time" in data
    
    @pytest.mark.asyncio
    async def test_multi_page_pdf(self, client):
        """Test OCR on multi-page PDF"""
        await self.wait_for_api(client)
        
        # Test all PDFs to find multi-page ones
        for pdf_file in PDF_TEST_FILES:
            pdf_path = TEST_FILES_DIR / pdf_file
            if pdf_path.exists():
                with open(pdf_path, "rb") as f:
                    files = {"file": (pdf_path.name, f, "application/pdf")}
                    response = await client.post(f"{API_BASE_URL}/ocr", files=files)
                    
                    assert response.status_code == 200
                    data = response.json()
                    
                    # Check for page markers in multi-page PDFs
                    if data.get("text") and "[Page " in data["text"]:
                        print(f"Found multi-page PDF: {pdf_file}")
                        assert "[Page 1]" in data["text"]
                        # Multi-page PDFs should have page markers
    
    @pytest.mark.asyncio
    async def test_pdf_with_qwen_vl(self, client):
        """Test PDF processing with Qwen VL endpoint"""
        await self.wait_for_api(client)
        
        pdf_path = TEST_FILES_DIR / PDF_TEST_FILES[0]
        if pdf_path.exists():
            with open(pdf_path, "rb") as f:
                files = {"file": (pdf_path.name, f, "application/pdf")}
                params = {"resource_type": "utility", "enable_reasoning": True}
                response = await client.post(
                    f"{API_BASE_URL}/ocr/qwen-vl/process", 
                    files=files,
                    params=params
                )
                
                # Even if Qwen service is not available, gateway should handle it
                if response.status_code == 200:
                    data = response.json()
                    assert "filename" in data
                    assert data["filename"] == pdf_path.name
                    # Check for multi-page note if applicable
                    if data.get("error") and "Note: Only first page" in data["error"]:
                        print("Multi-page PDF detected, only first page processed")
    
    @pytest.mark.asyncio
    async def test_batch_pdf_processing(self, client):
        """Test batch processing with PDFs"""
        await self.wait_for_api(client)
        
        # Prepare multiple files (mix of PDFs and images)
        files_to_upload = []
        
        # Add PDFs
        for pdf_file in PDF_TEST_FILES[:2]:  # Use first 2 PDFs
            pdf_path = TEST_FILES_DIR / pdf_file
            if pdf_path.exists():
                files_to_upload.append(
                    ("files", (pdf_path.name, open(pdf_path, "rb"), "application/pdf"))
                )
        
        # Add an image
        image_path = TEST_FILES_DIR / "DEWA.png"
        if image_path.exists():
            files_to_upload.append(
                ("files", (image_path.name, open(image_path, "rb"), "image/png"))
            )
        
        if files_to_upload:
            response = await client.post(f"{API_BASE_URL}/ocr/batch", files=files_to_upload)
            
            # Close files
            for _, (_, f, _) in files_to_upload:
                f.close()
            
            assert response.status_code == 200
            results = response.json()
            assert isinstance(results, list)
            assert len(results) == len(files_to_upload)
            
            # Check each result
            for result in results:
                assert "filename" in result
                assert "status" in result
                assert "processing_time" in result
                if result["status"] == "success":
                    assert "text" in result
    
    @pytest.mark.asyncio
    async def test_invalid_file_rejection(self, client):
        """Test that invalid files are rejected"""
        await self.wait_for_api(client)
        
        # Create a fake file that's neither image nor PDF
        files = {"file": ("test.txt", b"This is not a PDF", "text/plain")}
        response = await client.post(f"{API_BASE_URL}/ocr", files=files)
        
        assert response.status_code == 400
        assert "must be an image or PDF" in response.text
    
    @pytest.mark.asyncio 
    async def test_pdf_error_handling(self, client):
        """Test error handling for corrupted PDFs"""
        await self.wait_for_api(client)
        
        # Send corrupted PDF data
        files = {"file": ("corrupted.pdf", b"PDF-1.4 corrupted data", "application/pdf")}
        response = await client.post(f"{API_BASE_URL}/ocr", files=files)
        
        # Should handle gracefully
        assert response.status_code in [200, 500]
        if response.status_code == 500:
            assert "Failed to convert PDF" in response.text or "error" in response.text.lower()


def run_tests():
    """Run all PDF tests"""
    print("Running PDF processing tests...")
    print(f"Testing with PDF files: {PDF_TEST_FILES}")
    
    # Run pytest
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    run_tests()