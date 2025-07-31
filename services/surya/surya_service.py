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

# Import Surya OCR components
try:
    from surya.ocr import run_ocr
    from surya.model.detection.segformer import load_model as load_det_model, load_processor as load_det_processor
    from surya.model.recognition.model import load_model as load_rec_model
    from surya.model.recognition.processor import load_processor as load_rec_processor

    logger.info("✓ Surya OCR imported successfully")
except ImportError as e:
    logger.error(f"✗ Failed to import Surya OCR: {e}")
    sys.exit(1)

app = FastAPI(title="Surya OCR Service", version="1.0.0")

# Global models
det_model = None
det_processor = None
rec_model = None
rec_processor = None

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
    global det_model, det_processor, rec_model, rec_processor
    
    logger.info("Loading Surya OCR models...")
    try:
        # Load detection models
        det_model = load_det_model()
        det_processor = load_det_processor()
        logger.info("✓ Detection models loaded successfully")
        
        # Load recognition models
        rec_model = load_rec_model()
        rec_processor = load_rec_processor()
        logger.info("✓ Recognition models loaded successfully")
        
        logger.info("✓ Surya OCR service started successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load Surya OCR models: {e}")
        raise

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
            
            # Handle EXIF orientation
            try:
                exif = image.getexif()
                orientation_tag = 274  # EXIF orientation tag
                if orientation_tag in exif:
                    orientation = exif[orientation_tag]
                    
                    # Rotate image based on EXIF orientation
                    if orientation == 2:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 4:
                        image = image.transpose(Image.FLIP_TOP_BOTTOM)
                    elif orientation == 5:
                        image = image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation == 6:
                        image = image.rotate(-90, expand=True)
                    elif orientation == 7:
                        image = image.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
                    
                    logger.info(f"Applied EXIF orientation correction: {orientation}")
            except Exception as e:
                logger.warning(f"Could not process EXIF orientation: {e}")
            
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            # Run Surya OCR with loaded models
            langs = [["en", "ar"]]  # For DEWA/SEWA bills - array of language lists
            results = run_ocr([image], langs, det_model, det_processor, rec_model, rec_processor)

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