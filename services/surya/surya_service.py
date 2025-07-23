#!/usr/bin/env python3
"""
Surya OCR Microservice
Isolated service running transformers 4.36.2 for Surya OCR compatibility
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import asyncio
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Surya OCR (just the main function)
try:
    from surya.ocr import run_ocr
    logger.info("✓ Surya OCR imported successfully")
except ImportError as e:
    logger.error(f"✗ Failed to import Surya OCR: {e}")
    sys.exit(1)

app = FastAPI(title="Surya OCR Service", version="1.0.0")

# Global models
detector = None
processor = None
tokenizer = None

class OCRResponse(BaseModel):
    filename: str
    status: str
    text: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
    processing_time: float

@app.on_event("startup")
async def startup_event():
    """Load Surya OCR models on startup"""
    logger.info("Surya OCR service ready - models will be loaded on first request")
    logger.info("✓ Surya OCR service started successfully")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "surya-ocr", "ready": True}

@app.post("/ocr", response_model=OCRResponse)
async def perform_ocr(file: UploadFile = File(...)):
    """Perform OCR on uploaded image using Surya OCR"""
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        content = await file.read()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Load image
            image = Image.open(tmp_path)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            # Run Surya OCR
            langs = [["en", "ar"]]  # For DEWA/SEWA bills - array of language lists
            results = run_ocr([image], langs)
            
            # Extract text from results
            if results and len(results) > 0:
                result = results[0]
                text_parts = []
                total_confidence = 0
                confidence_count = 0
                
                for text_line in result.text_lines:
                    if text_line.text.strip():
                        text_parts.append(text_line.text.strip())
                        if hasattr(text_line, 'confidence') and text_line.confidence:
                            total_confidence += text_line.confidence
                            confidence_count += 1
                
                extracted_text = '\n'.join(text_parts)
                avg_confidence = total_confidence / confidence_count if confidence_count > 0 else None
                
                processing_time = (time.time() - start_time) * 1000
                
                return OCRResponse(
                    filename=file.filename,
                    status="success",
                    text=extracted_text,
                    confidence=avg_confidence,
                    processing_time=processing_time
                )
            else:
                processing_time = (time.time() - start_time) * 1000
                return OCRResponse(
                    filename=file.filename,
                    status="error",
                    error="No text detected in image",
                    processing_time=processing_time
                )
                
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.error(f"OCR processing failed: {e}")
        return OCRResponse(
            filename=file.filename,
            status="error",
            error=str(e),
            processing_time=processing_time
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)