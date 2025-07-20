#!/usr/bin/env python3
"""FastAPI server for Surya OCR service"""

import os
import io
import json
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import magic

app = FastAPI(
    title="Surya OCR API with Qwen Enhancement",
    description="OCR service powered by Surya with Qwen3-0.6B for structured data extraction",
    version="2.0.0"
)

# Configuration
TEMP_DIR = Path("/tmp/surya_ocr_api")
TEMP_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

class OCRResponse(BaseModel):
    """OCR response model"""
    filename: str
    status: str
    text: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str

def cleanup_temp_file(file_path: str):
    """Clean up temporary file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        # Also remove the output directory
        output_dir = Path(file_path).parent / Path(file_path).stem
        if output_dir.exists():
            shutil.rmtree(output_dir)
    except Exception as e:
        print(f"Error cleaning up {file_path}: {e}")

def validate_image_file(file_content: bytes, filename: str) -> bytes:
    """Validate and convert image file if needed"""
    # Check file size
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_FILE_SIZE/1024/1024}MB")
    
    # Detect actual file type
    mime = magic.from_buffer(file_content, mime=True)
    
    # Handle WebP files with wrong extension
    if mime == 'image/webp' and filename.lower().endswith('.png'):
        # Convert WebP to PNG
        img = Image.open(io.BytesIO(file_content))
        png_buffer = io.BytesIO()
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        img.save(png_buffer, 'PNG')
        return png_buffer.getvalue()
    
    # Validate supported formats
    supported_mimes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'application/pdf']
    if mime not in supported_mimes:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {mime}")
    
    return file_content

def run_surya_ocr(file_path: Path) -> Dict:
    """Run Surya OCR on a file"""
    start_time = datetime.now()
    
    try:
        # Create output directory
        output_dir = file_path.parent / file_path.stem
        output_dir.mkdir(exist_ok=True)
        
        # Run surya_ocr command
        cmd = ["surya_ocr", str(file_path), "--output_dir", str(output_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            return {
                "status": "error",
                "error": f"OCR process failed: {result.stderr}"
            }
        
        # Parse results
        results_file = output_dir / file_path.stem / "results.json"
        
        if not results_file.exists():
            return {
                "status": "error",
                "error": "OCR results file not found"
            }
        
        with open(results_file, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
        
        # Extract text
        text_parts = []
        total_confidence = 0
        confidence_count = 0
        
        for filename, pages in ocr_data.items():
            if isinstance(pages, list):
                for page in pages:
                    if 'text_lines' in page:
                        for line in page['text_lines']:
                            text_parts.append(line.get('text', ''))
                            if 'confidence' in line:
                                total_confidence += line['confidence']
                                confidence_count += 1
            elif isinstance(pages, dict) and 'text_lines' in pages:
                for line in pages['text_lines']:
                    text_parts.append(line.get('text', ''))
                    if 'confidence' in line:
                        total_confidence += line['confidence']
                        confidence_count += 1
        
        # Calculate average confidence
        avg_confidence = (total_confidence / confidence_count) if confidence_count > 0 else None
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "status": "success",
            "text": '\n'.join(text_parts),
            "confidence": avg_confidence,
            "processing_time": processing_time
        }
        
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "error": "OCR process timed out after 5 minutes"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/ocr", response_model=OCRResponse)
async def process_single_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Process a single image or PDF file"""
    temp_path = None
    
    try:
        # Read and validate file
        file_content = await file.read()
        validated_content = validate_image_file(file_content, file.filename)
        
        # Save to temporary file
        suffix = Path(file.filename).suffix or '.png'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TEMP_DIR) as tmp:
            tmp.write(validated_content)
            temp_path = Path(tmp.name)
        
        # Run OCR
        result = run_surya_ocr(temp_path)
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, str(temp_path))
        
        return OCRResponse(
            filename=file.filename,
            status=result["status"],
            text=result.get("text"),
            confidence=result.get("confidence"),
            error=result.get("error"),
            processing_time=result.get("processing_time")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        if temp_path:
            background_tasks.add_task(cleanup_temp_file, str(temp_path))
        return OCRResponse(
            filename=file.filename,
            status="error",
            error=str(e)
        )

@app.post("/ocr/batch", response_model=List[OCRResponse])
async def process_batch_images(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Process multiple images or PDF files"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    results = []
    
    for file in files:
        result = await process_single_image(background_tasks, file)
        results.append(result)
    
    return results

@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup"""
    print(f"Surya OCR API with Qwen starting on port {os.getenv('PORT', 8080)}")
    print(f"Temporary directory: {TEMP_DIR}")
    
    # Test surya_ocr availability
    try:
        result = subprocess.run(["surya_ocr", "--help"], capture_output=True)
        if result.returncode != 0:
            print("WARNING: surya_ocr command not found. Make sure it's installed.")
    except FileNotFoundError:
        print("WARNING: surya_ocr command not found. Make sure it's installed.")
    
    # Add Qwen routes
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from qwen_integration.api_extensions import add_qwen_routes
        add_qwen_routes(app)
        print("Qwen OCR extensions loaded successfully")
    except ImportError as e:
        print(f"WARNING: Could not load Qwen extensions: {e}")
        print("Qwen endpoints will not be available")
    except Exception as e:
        print(f"ERROR loading Qwen extensions: {e}")
        print("Qwen endpoints will not be available")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    # Clean up temp directory
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)