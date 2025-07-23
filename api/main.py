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
import time
import sys
import psutil
import platform

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import magic

# Import logging configuration
try:
    from .logging_config import setup_logging, get_logger, set_request_id, log_performance, log_memory_usage
    from .monitoring import record_request_start, record_request_end, get_application_metrics, ResourceMonitor
except ImportError:
    # When running as a script
    from logging_config import setup_logging, get_logger, set_request_id, log_performance, log_memory_usage
    from monitoring import record_request_start, record_request_end, get_application_metrics, ResourceMonitor

# Setup logging
setup_logging("ocr-engine")
logger = get_logger(__name__)

# Add parent directory to sys.path for qwen_vl_integration imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    logger.debug(f"Added to sys.path early: {parent_dir}")

app = FastAPI(
    title="Surya OCR API with Qwen Enhancement",
    description="OCR service powered by Surya with Qwen3-0.6B for structured data extraction",
    version="2.0.0"
)

# Configuration
TEMP_DIR = Path("/tmp/surya_ocr_api")
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Log startup configuration
logger.info("Starting OCR Engine API", extra={
    'temp_dir': str(TEMP_DIR),
    'max_file_size_mb': MAX_FILE_SIZE / 1024 / 1024,
    'python_version': sys.version,
    'platform': platform.platform(),
    'pid': os.getpid(),
    'port': os.getenv('PORT', 8080)
})

# Create temp directory with logging
try:
    TEMP_DIR.mkdir(exist_ok=True)
    logger.info(f"Temp directory ready: {TEMP_DIR}")
except Exception as e:
    logger.error(f"Failed to create temp directory: {e}", exc_info=True)
    raise

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
    start_time = time.time()
    logger.info(f"Starting Surya OCR processing", extra={
        'file_path': str(file_path),
        'file_size': os.path.getsize(file_path) / 1024,  # KB
    })
    
    try:
        # Load image
        logger.debug("Loading image...")
        image = Image.open(file_path)
        logger.debug(f"Image loaded", extra={
            'mode': image.mode,
            'size': image.size,
            'format': image.format
        })
        
        if image.mode != 'RGB':
            logger.debug(f"Converting image from {image.mode} to RGB")
            image = image.convert('RGB')
        
        # Import Surya functions
        logger.debug("Importing Surya functions...")
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor
        
        # Load models if not already loaded
        logger.debug("Loading Surya models...")
        model_start = time.time()
        
        # Use the global predictors if available
        if 'surya_predictors' in globals() and 'detection' in surya_predictors:
            logger.debug("Using pre-loaded predictors")
            det_predictor = surya_predictors['detection']
            rec_predictor = surya_predictors['recognition']
        else:
            logger.debug("Loading models fresh")
            # Load predictors
            det_predictor = DetectionPredictor()
            rec_predictor = RecognitionPredictor()
        
        model_load_time = (time.time() - model_start) * 1000
        logger.debug(f"Models loaded", extra={'load_time_ms': round(model_load_time, 2)})
        
        # Run OCR using Surya's run_ocr function
        logger.info("Starting OCR processing...")
        ocr_start = time.time()
        
        # Run detection and recognition
        # First detect text regions
        det_results = det_predictor([image])
        
        # Then run recognition on detected regions
        rec_results = rec_predictor([image], det_predictor=det_predictor)
        
        # Get the first result (we only process one image)
        results = rec_results
        
        ocr_time = (time.time() - ocr_start) * 1000
        logger.info(f"OCR processing complete", extra={'ocr_time_ms': round(ocr_time, 2)})
        
        # Extract text from results
        text_parts = []
        total_confidence = 0
        confidence_count = 0
        
        if results and len(results) > 0:
            result = results[0]  # Get first (and only) result
            logger.debug(f"Processing {len(result.text_lines) if hasattr(result, 'text_lines') else 0} text lines")
            
            if hasattr(result, 'text_lines'):
                for line in result.text_lines:
                    text_parts.append(line.text)
                    if hasattr(line, 'confidence') and line.confidence is not None:
                        total_confidence += line.confidence
                        confidence_count += 1
        
        # Calculate average confidence
        avg_confidence = (total_confidence / confidence_count) if confidence_count > 0 else None
        
        # Calculate total processing time
        total_time = (time.time() - start_time)
        
        logger.info(f"Surya OCR completed successfully", extra={
            'total_time_s': round(total_time, 2),
            'text_length': len('\n'.join(text_parts)),
            'line_count': len(text_parts),
            'avg_confidence': round(avg_confidence, 3) if avg_confidence else None
        })
        
        return {
            "status": "success",
            "text": '\n'.join(text_parts),
            "confidence": avg_confidence,
            "processing_time": total_time
        }
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"Surya OCR failed", extra={
            'error': str(e),
            'error_type': type(e).__name__,
            'time_before_error_s': round(error_time, 2)
        }, exc_info=True)
        
        return {
            "status": "error",
            "error": str(e),
            "processing_time": error_time
        }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    # Check system health
    health = ResourceMonitor.check_resource_health()
    
    return HealthResponse(
        status="healthy" if health['healthy'] else "degraded",
        timestamp=datetime.now().isoformat(),
        version="2.0.0"
    )

@app.get("/debug/metrics")
async def debug_metrics():
    """Get application metrics (development only)"""
    if os.getenv('ENVIRONMENT') == 'production':
        raise HTTPException(status_code=403, detail="Debug endpoints disabled in production")
    
    return get_application_metrics()

@app.get("/debug/resources")
async def debug_resources():
    """Get current resource usage (development only)"""
    if os.getenv('ENVIRONMENT') == 'production':
        raise HTTPException(status_code=403, detail="Debug endpoints disabled in production")
    
    return ResourceMonitor.get_current_resources()

@app.post("/ocr", response_model=OCRResponse)
async def process_single_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Process a single image or PDF file"""
    request_start = time.time()
    temp_path = None
    
    logger.info(f"OCR request received", extra={
        'file_name': file.filename,
        'content_type': file.content_type
    })
    
    try:
        # Read and validate file
        file_content = await file.read()
        logger.debug(f"File read complete", extra={'size_bytes': len(file_content)})
        
        validated_content = validate_image_file(file_content, file.filename)
        logger.debug("File validation passed")
        
        # Save to temporary file
        suffix = Path(file.filename).suffix or '.png'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TEMP_DIR) as tmp:
            tmp.write(validated_content)
            temp_path = Path(tmp.name)
        
        # Run OCR with timeout
        ocr_start = time.time()
        
        # Import timeout utility
        from .api_utils import run_with_timeout, TimeoutException
        
        try:
            # Run OCR with 60-second timeout
            result = run_with_timeout(run_surya_ocr, args=(temp_path,), timeout_seconds=60)
        except TimeoutException:
            result = {
                "status": "error",
                "error": "OCR processing timed out after 60 seconds",
                "processing_time": 60.0
            }
            logger.error(f"OCR timed out for file: {file.filename}")
        
        ocr_duration = (time.time() - ocr_start) * 1000
        
        log_performance(logger, "surya_ocr", ocr_duration, result.get('status') == 'success', 
                       file_name=file.filename)
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, str(temp_path))
        
        # Calculate total processing time
        total_duration = (time.time() - request_start) * 1000
        
        response = OCRResponse(
            filename=file.filename,
            status=result["status"],
            text=result.get("text"),
            confidence=result.get("confidence"),
            error=result.get("error"),
            processing_time=result.get("processing_time")
        )
        
        logger.info(f"OCR request completed", extra={
            'file_name': file.filename,
            'status': result["status"],
            'total_duration_ms': round(total_duration, 2),
            'text_length': len(result.get("text", "")) if result.get("text") else 0,
            'confidence': result.get("confidence")
        })
        
        return response
        
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

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with timing"""
    request_id = set_request_id()
    start_time = time.time()
    
    # Start monitoring
    metric = record_request_start(request.url.path)
    
    # Log request
    logger.info(f"Request started: {request.method} {request.url.path}", extra={
        'method': request.method,
        'path': request.url.path,
        'query_params': dict(request.query_params),
        'client_host': request.client.host if request.client else None
    })
    
    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        
        # Record metrics
        record_request_end(metric, response.status_code)
        
        # Log response
        logger.info(f"Request completed: {request.method} {request.url.path}", extra={
            'status_code': response.status_code,
            'duration_ms': round(duration_ms, 2)
        })
        
        return response
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        
        # Record metrics
        record_request_end(metric, 500, str(e))
        
        logger.error(f"Request failed: {request.method} {request.url.path}", extra={
            'duration_ms': round(duration_ms, 2),
            'error': str(e)
        }, exc_info=True)
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup with comprehensive logging"""
    startup_start = time.time()
    
    logger.info("=" * 80)
    logger.info("SURYA OCR API STARTUP INITIATED")
    logger.info("=" * 80)
    
    # System information
    system_info = {
        'port': os.getenv('PORT', 8080),
        'working_directory': os.getcwd(),
        'python_version': sys.version,
        'python_executable': sys.executable,
        'platform': platform.platform(),
        'cpu_count': psutil.cpu_count(),
        'total_memory_gb': round(psutil.virtual_memory().total / 1024 / 1024 / 1024, 2),
        'available_memory_gb': round(psutil.virtual_memory().available / 1024 / 1024 / 1024, 2),
        'temp_directory': str(TEMP_DIR)
    }
    logger.info("System information", extra=system_info)
    
    # List directory contents
    try:
        dir_contents = os.listdir('.')
        logger.info(f"Directory contents ({len(dir_contents)} items)", extra={
            'files': [f for f in dir_contents if os.path.isfile(f)],
            'directories': [d for d in dir_contents if os.path.isdir(d)]
        })
    except Exception as e:
        logger.error(f"Failed to list directory contents: {e}", exc_info=True)
    
    # Check environment variables
    env_vars = {}
    critical_vars = ['PYTHONPATH', 'HF_HOME', 'TRANSFORMERS_CACHE', 'TORCH_HOME', 'PORT', 'ENVIRONMENT']
    for key in critical_vars:
        env_vars[key] = os.environ.get(key, 'Not set')
    logger.info("Environment variables", extra=env_vars)
    
    # Test surya_ocr availability
    logger.info("Checking surya_ocr command availability...")
    try:
        result = subprocess.run(["surya_ocr", "--help"], capture_output=True, timeout=5)
        if result.returncode == 0:
            logger.info("✓ surya_ocr command available")
        else:
            logger.warning("surya_ocr command returned non-zero exit code", extra={
                'returncode': result.returncode,
                'stderr': result.stderr.decode() if result.stderr else None
            })
    except FileNotFoundError:
        logger.error("surya_ocr command not found in PATH")
    except subprocess.TimeoutExpired:
        logger.error("surya_ocr command timed out")
    except Exception as e:
        logger.error(f"Error checking surya_ocr: {e}", exc_info=True)
    
    # Pre-load Surya models to avoid timeout on first request
    logger.info("Pre-loading Surya OCR models...")
    surya_start = time.time()
    
    # Log memory before loading
    log_memory_usage(logger, "Before Surya model loading")
    
    try:
        import surya
        logger.info(f"✓ surya module imported", extra={'module_path': surya.__file__})
    except ImportError as e:
        logger.error(f"Failed to import surya module: {e}", exc_info=True)
    
    try:
        # Import Surya modules
        logger.info("Importing Surya modules...")
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor
        logger.info("✓ Surya modules imported successfully")
        
        # Load models properly
        logger.info("Loading Surya models...")
        predictor_start = time.time()
        
        global surya_predictors
        surya_predictors = {}
        
        # Load detection and recognition predictors
        logger.debug("Loading detection predictor...")
        det_predictor = DetectionPredictor()
        logger.debug("Loading recognition predictor...")
        rec_predictor = RecognitionPredictor()
        
        # Store predictors
        surya_predictors['detection'] = det_predictor
        surya_predictors['recognition'] = rec_predictor
        
        predictor_time = (time.time() - predictor_start) * 1000
        logger.info("✓ Surya models loaded", extra={
            'models': list(surya_predictors.keys()),
            'load_time_ms': round(predictor_time, 2)
        })
        
        # Log memory after loading
        log_memory_usage(logger, "After Surya model loading")
        
        surya_total_time = (time.time() - surya_start) * 1000
        logger.info("✓ Surya OCR models pre-loaded successfully", extra={
            'total_time_ms': round(surya_total_time, 2)
        })
        
    except ImportError as e:
        logger.error(f"Import error for Surya modules: {e}", exc_info=True)
        # Check installed packages
        try:
            result = subprocess.run(["pip", "list"], capture_output=True, text=True, timeout=5)
            surya_packages = [line for line in result.stdout.split('\n') if 'surya' in line.lower()]
            logger.error("Surya-related packages installed", extra={'packages': surya_packages})
        except Exception as pkg_e:
            logger.error(f"Failed to check installed packages: {pkg_e}")
    except Exception as e:
        logger.error(f"Failed to pre-load Surya models: {e}", exc_info=True)
        logger.warning("Models will be loaded on first use (may cause timeout)")
    
    # Add Qwen routes
    logger.info("Loading Qwen VL extensions...")
    qwen_start = time.time()
    
    try:
        # Check for critical dependencies first
        try:
            import qwen_vl_utils
            logger.info("✓ qwen-vl-utils imported successfully")
        except ImportError as e:
            logger.error("qwen-vl-utils not installed", extra={
                'error': str(e),
                'install_command': 'pip install qwen-vl-utils'
            })
            logger.warning("Qwen VL endpoints will not be available")
            return
        
        # Import the Qwen endpoints
        logger.debug("Attempting to import Qwen endpoints...")
        try:
            from .qwen_endpoints import add_qwen_routes
            logger.debug("✓ Import successful")
        except ImportError as e:
            logger.error("Failed to import add_qwen_routes", extra={
                'error': str(e)
            })
            raise
        
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