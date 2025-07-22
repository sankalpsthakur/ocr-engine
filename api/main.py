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
    """Run Surya OCR on a file using direct API"""
    start_time = datetime.now()
    
    try:
        # Load image
        image = Image.open(file_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Check if predictors are loaded
        if 'surya_predictors' not in globals():
            from surya.models import load_predictors
            global surya_predictors
            surya_predictors = load_predictors()
        
        # Get detection and recognition predictors
        det_predictor = surya_predictors['detection']
        rec_predictor = surya_predictors['recognition']
        
        # Run OCR
        # First run detection
        detection_results = det_predictor([image])
        
        # Then run recognition with detection results
        ocr_results = rec_predictor(
            [image],
            det_predictor=det_predictor,
            highres_images=None,
            math_mode=True
        )
        
        # Extract text from results
        text_parts = []
        total_confidence = 0
        confidence_count = 0
        
        if ocr_results and len(ocr_results) > 0:
            result = ocr_results[0]  # Get first (and only) result
            if hasattr(result, 'text_lines'):
                for line in result.text_lines:
                    text_parts.append(line.text)
                    if hasattr(line, 'confidence') and line.confidence is not None:
                        total_confidence += line.confidence
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
    import sys
    print("=" * 80)
    print("SURYA OCR API STARTUP")
    print("=" * 80)
    print(f"Port: {os.getenv('PORT', 8080)}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Python path: {sys.path}")
    print(f"Temporary directory: {TEMP_DIR}")
    
    # List directory contents
    print("\nDirectory contents:")
    for item in os.listdir('.'):
        print(f"  {item}")
    
    # Check environment variables
    print("\nEnvironment variables:")
    for key in ['PYTHONPATH', 'HF_HOME', 'TRANSFORMERS_CACHE', 'TORCH_HOME']:
        print(f"  {key}: {os.environ.get(key, 'Not set')}")
    
    # Test surya_ocr availability
    print("\nChecking surya_ocr command...")
    try:
        result = subprocess.run(["surya_ocr", "--help"], capture_output=True)
        if result.returncode != 0:
            print("WARNING: surya_ocr command not found. Make sure it's installed.")
    except FileNotFoundError:
        print("WARNING: surya_ocr command not found. Make sure it's installed.")
    
    # Pre-load Surya models to avoid timeout on first request
    print("\nPre-loading Surya OCR models...")
    print("  Attempting import surya...")
    try:
        import surya
        print(f"  ✓ surya imported from: {surya.__file__}")
    except ImportError as e:
        print(f"  ✗ Failed to import surya: {e}")
    
    try:
        # Import Surya predictors
        print("  Attempting to import surya.detection, surya.recognition, surya.models...")
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor
        from surya.models import load_predictors
        print("  ✓ Surya modules imported successfully")
        
        # Load all predictors (detection, recognition, etc.)
        print("  Loading predictors...")
        global surya_predictors
        surya_predictors = load_predictors()
        print("  ✓ Surya predictors loaded:", list(surya_predictors.keys()))
        print("✓ Surya OCR models pre-loaded successfully")
    except ImportError as e:
        print(f"✗ Import error for Surya: {e}")
        print("  Checking installed packages:")
        import subprocess
        result = subprocess.run(["pip", "list", "|", "grep", "-i", "surya"], 
                              shell=True, capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"✗ Failed to pre-load Surya models: {e}")
        import traceback
        traceback.print_exc()
        print("Models will be loaded on first use (may cause timeout)")
    
    # Add Qwen routes
    print("\nLoading Qwen VL extensions...")
    try:
        # Add parent directory to path
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            print(f"  Added to sys.path: {parent_dir}")
        
        # Debug: Check if qwen_vl_integration exists
        qwen_path = Path(parent_dir) / "qwen_vl_integration"
        print(f"  Checking for qwen_vl_integration at: {qwen_path}")
        if qwen_path.exists():
            print("  ✓ qwen_vl_integration directory exists")
            # List contents
            for item in qwen_path.iterdir():
                print(f"    - {item.name}")
        else:
            print("  ✗ qwen_vl_integration directory NOT found")
        
        # Check for critical dependencies first
        try:
            import qwen_vl_utils
            print("  ✓ qwen-vl-utils imported")
        except ImportError:
            print("  ✗ qwen-vl-utils not installed")
            print("    Install with: pip install qwen-vl-utils")
            print("  Qwen VL endpoints will not be available")
            return
        
        # Try to import the extension
        print("  Attempting to import qwen_vl_integration.api_extensions...")
        try:
            from qwen_vl_integration.api_extensions import add_qwen_routes
            print("  ✓ Import successful (using wrapper)")
        except ImportError:
            print("  Wrapper import failed, trying direct import...")
            from qwen_vl_integration.src.api_extensions import add_qwen_routes
            print("  ✓ Import successful (direct import)")
        
        add_qwen_routes(app)
        print("✓ Qwen VL extensions loaded successfully")
        print("  Available endpoints:")
        print("    - /ocr/qwen-vl/health")
        print("    - /ocr/qwen-vl/process")
        print("    - /ocr/qwen-vl/extract-text")
        print("    - /ocr/qwen-vl/schema/{provider}")
    except ImportError as e:
        print(f"✗ Import error for Qwen extensions: {e}")
        print("  Qwen VL endpoints will not be available")
        print("  Make sure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"✗ ERROR loading Qwen extensions: {e}")
        import traceback
        traceback.print_exc()
        print("  Qwen VL endpoints will not be available")
    
    print("\n" + "=" * 80)
    print("STARTUP COMPLETE")
    print("=" * 80)

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