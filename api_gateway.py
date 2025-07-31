#!/usr/bin/env python3
"""
OCR Engine API Gateway
Coordinates between Surya OCR and Qwen 2.5-VL 3B Instruct microservices
"""

import os
import sys
import time
import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
import uvicorn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from api.utils.pdf_handler import PDFHandler
from api.utils.image_compressor import compress_image_for_qwen, compress_image_for_ocr
from api.utils.orientation_handler import OrientationHandler


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service URLs
SURYA_SERVICE_URL = os.getenv("SURYA_SERVICE_URL", "http://localhost:8001")
QWEN_SERVICE_URL = os.getenv("QWEN_SERVICE_URL", "http://localhost:8002")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    logger.info("API Gateway starting up...")
    
    # In Docker environment, services might take longer to start
    max_retries = int(os.getenv("SERVICE_STARTUP_RETRIES", "10"))
    retry_delay = int(os.getenv("SERVICE_STARTUP_DELAY", "3"))
    
    for attempt in range(max_retries):
        # Check service health
        surya_health = await check_service_health(SURYA_SERVICE_URL, "Surya OCR")
        qwen_health = await check_service_health(QWEN_SERVICE_URL, "Qwen 2.5-VL")
        
        if surya_health == "healthy" and qwen_health == "healthy":
            logger.info(f"Surya OCR Service: {surya_health}")
            logger.info(f"Qwen 2.5-VL Service: {qwen_health}")
            logger.info("All services are healthy!")
            break
        
        if attempt < max_retries - 1:
            logger.info(f"Services not ready yet (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay}s...")
            await asyncio.sleep(retry_delay)
        else:
            logger.info(f"Surya OCR Service: {surya_health}")
            logger.info(f"Qwen 2.5-VL Service: {qwen_health}")
            
            if surya_health == "unreachable":
                logger.warning("Surya OCR service is not available")
            if qwen_health == "unreachable":
                logger.warning("Qwen 2.5-VL service is not available")
    
    logger.info("API Gateway startup complete")
    
    yield
    
    # Shutdown
    logger.info("API Gateway shutting down...")

app = FastAPI(
    title="OCR Engine API Gateway",
    description="Unified API for Surya OCR and Qwen 2.5-VL 3B Instruct processing",
    version="2.0.0",
    lifespan=lifespan
)

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    services: Dict[str, str]

class OCRResponse(BaseModel):
    filename: str
    status: str
    text: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
    processing_time: float

class QwenVLResponse(BaseModel):
    filename: str
    status: str
    provider: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    raw_text: Optional[str] = None
    extraction_method: str
    confidence: Optional[float] = None
    processing_time: float
    error: Optional[str] = None

async def check_service_health(service_url: str, service_name: str) -> str:
    """Check health of a microservice"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{service_url}/health")
            if response.status_code == 200:
                return "healthy"
            else:
                return "unhealthy"
    except Exception as e:
        logger.error(f"{service_name} health check failed: {e}")
        return "unreachable"

async def validate_file(file: UploadFile) -> None:
    """Validate uploaded file is an image or PDF"""
    content_type = file.content_type or ""
    filename = file.filename or ""
    
    # Check if it's an image
    if content_type.startswith('image/'):
        return
    
    # Check if it's a PDF
    if PDFHandler.is_pdf(content_type, filename):
        return
    
    # Invalid file type
    raise HTTPException(
        status_code=400, 
        detail=f"File must be an image or PDF. Received: {content_type}"
    )

async def preprocess_file(file: UploadFile, auto_orient: bool = True) -> List[Tuple[bytes, str, str]]:
    """
    Preprocess file - convert PDF to images if needed and fix orientation
    
    Args:
        file: Uploaded file
        auto_orient: Whether to automatically fix orientation
    
    Returns:
        List of tuples (content_bytes, filename, content_type)
    """
    content = await file.read()
    await file.seek(0)  # Reset file pointer
    
    # If it's already an image, apply orientation correction
    if file.content_type and file.content_type.startswith('image/'):
        if auto_orient:
            logger.info(f"Checking orientation for image {file.filename}...")
            # Check if this appears to be a utility bill
            is_utility = any(provider in (file.filename or "").upper() for provider in ['DEWA', 'SEWA'])
            
            # Apply orientation correction
            from PIL import Image
            import io
            
            image = Image.open(io.BytesIO(content))
            # Set Surya service URL
            OrientationHandler.SURYA_SERVICE_URL = SURYA_SERVICE_URL
            image = await OrientationHandler.auto_orient(image, is_utility_bill=is_utility)
            
            # Convert back to bytes
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG' if file.content_type == 'image/png' else 'JPEG')
            content = img_buffer.getvalue()
            
        return [(content, file.filename, file.content_type)]
    
    # If it's a PDF, convert to images (orientation handled in PDFHandler)
    if PDFHandler.is_pdf(file.content_type or "", file.filename or ""):
        logger.info(f"Converting PDF {file.filename} to images...")
        images = await PDFHandler.convert_to_images(
            content, 
            file.filename or "document.pdf",
            auto_orient=auto_orient
        )
        return [(img_bytes, img_name, "image/png") for img_bytes, img_name in images]
    
    # This shouldn't happen due to validation, but just in case
    raise HTTPException(status_code=400, detail="Unsupported file type")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Overall system health check"""
    from datetime import datetime
    
    # Check both services
    surya_health = await check_service_health(SURYA_SERVICE_URL, "Surya OCR")
    qwen_health = await check_service_health(QWEN_SERVICE_URL, "Qwen 2.5-VL")
    
    overall_status = "healthy" if surya_health == "healthy" and qwen_health == "healthy" else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        services={
            "surya_ocr": surya_health,
            "qwen_2_5_vl": qwen_health
        }
    )

@app.post("/ocr", response_model=OCRResponse)
async def basic_ocr(file: UploadFile = File(...)):
    """Basic OCR using Surya OCR service - supports images and PDFs"""
    start_time = time.time()
    
    try:
        # Validate file type
        await validate_file(file)
        
        # Preprocess file (convert PDF to images if needed)
        file_items = await preprocess_file(file)
        
        # Process single image or first page of PDF
        if len(file_items) == 1:
            # Single image or single-page PDF
            content, filename, content_type = file_items[0]
            
            # Compress image if needed (> 1MB)
            original_size_kb = len(content) / 1024
            content, content_type = compress_image_for_ocr(content, filename, content_type)
            compressed_size_kb = len(content) / 1024
            
            if compressed_size_kb < original_size_kb:
                logger.info(f"Image compressed for OCR from {original_size_kb:.1f}KB to {compressed_size_kb:.1f}KB")
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                files = {"file": (filename, content, content_type)}
                response = await client.post(f"{SURYA_SERVICE_URL}/ocr", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    result["processing_time"] = time.time() - start_time
                    return OCRResponse(**result)
                else:
                    raise HTTPException(status_code=response.status_code, detail=response.text)
        else:
            # Multi-page PDF - concatenate text from all pages
            all_text = []
            total_confidence = 0
            page_count = 0
            
            async with httpx.AsyncClient(timeout=300.0) as client:
                for content, filename, content_type in file_items:
                    # Compress image if needed (> 1MB)
                    original_size_kb = len(content) / 1024
                    content, content_type = compress_image_for_ocr(content, filename, content_type)
                    compressed_size_kb = len(content) / 1024
                    
                    if compressed_size_kb < original_size_kb:
                        logger.info(f"Page {page_count + 1} compressed for OCR from {original_size_kb:.1f}KB to {compressed_size_kb:.1f}KB")
                    
                    files = {"file": (filename, content, content_type)}
                    response = await client.post(f"{SURYA_SERVICE_URL}/ocr", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("text"):
                            all_text.append(f"[Page {page_count + 1}]\n{result['text']}")
                            if result.get("confidence"):
                                total_confidence += result["confidence"]
                        page_count += 1
                    else:
                        logger.warning(f"Failed to process page {page_count + 1}: {response.text}")
            
            # Return aggregated result
            return OCRResponse(
                filename=file.filename,
                status="success",
                text="\n\n".join(all_text) if all_text else None,
                confidence=total_confidence / page_count if page_count > 0 else None,
                processing_time=time.time() - start_time
            )
                
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Surya OCR service timeout")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR service error: {e}")
        raise HTTPException(status_code=500, detail=f"OCR service error: {str(e)}")

@app.post("/ocr/qwen-vl/process", response_model=QwenVLResponse)
async def qwen_vl_process(
    file: UploadFile = File(...),
    resource_type: str = Query(default="utility", description="Type of resource (water, energy, utility)"),
    enable_reasoning: bool = Query(default=True, description="Enable spatial reasoning")
):
    """Process document with Qwen 2.5-VL 3B Instruct - supports images and PDFs"""
    start_time = time.time()
    
    try:
        # Validate file type
        await validate_file(file)
        
        # Preprocess file (convert PDF to images if needed)
        file_items = await preprocess_file(file)
        
        # For Qwen VL, we'll process the first page only for now
        # (Multi-page structured extraction would require more complex logic)
        if file_items:
            content, filename, content_type = file_items[0]
            
            if len(file_items) > 1:
                logger.info(f"Processing first page of {len(file_items)}-page PDF with Qwen VL")
            
            # First, get OCR text from Surya
            logger.info(f"Getting OCR text for Qwen VL processing...")
            ocr_text = ""
            
            # Compress image if needed for OCR (> 1MB)
            ocr_content = content
            ocr_content_type = content_type
            original_size_kb = len(ocr_content) / 1024
            ocr_content, ocr_content_type = compress_image_for_ocr(ocr_content, filename, ocr_content_type)
            compressed_size_kb = len(ocr_content) / 1024
            
            if compressed_size_kb < original_size_kb:
                logger.info(f"Image compressed for OCR from {original_size_kb:.1f}KB to {compressed_size_kb:.1f}KB")
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                files = {"file": (filename, ocr_content, ocr_content_type)}
                ocr_response = await client.post(f"{SURYA_SERVICE_URL}/ocr", files=files)
                
                if ocr_response.status_code == 200:
                    ocr_result = ocr_response.json()
                    if ocr_result.get("status") == "success":
                        ocr_text = ocr_result.get("text", "")
                        logger.info(f"OCR text extracted: {len(ocr_text)} characters")
                    else:
                        logger.warning(f"OCR extraction failed: {ocr_result.get('error', 'Unknown error')}")
                else:
                    logger.warning(f"OCR service returned status {ocr_response.status_code}")
            
            # Compress image if needed (> 512KB)
            original_size_kb = len(content) / 1024
            content, content_type = compress_image_for_qwen(content, filename, content_type)
            compressed_size_kb = len(content) / 1024
            
            if compressed_size_kb < original_size_kb:
                logger.info(f"Image compressed from {original_size_kb:.1f}KB to {compressed_size_kb:.1f}KB")
            
            async with httpx.AsyncClient(timeout=600.0) as client:
                files = {"file": (filename, content, content_type)}
                data = {"ocr_text": ocr_text}  # Pass OCR text as form data
                params = {"resource_type": resource_type, "enable_reasoning": enable_reasoning}
                
                response = await client.post(
                    f"{QWEN_SERVICE_URL}/process", 
                    files=files,
                    data=data,
                    params=params
                )
                
                if response.status_code == 200:
                    result = response.json()
                    result["processing_time"] = time.time() - start_time
                    if len(file_items) > 1:
                        # Add note about partial processing
                        if "error" in result and result["error"]:
                            result["error"] += f" (Note: Only first page of {len(file_items)}-page PDF was processed)"
                        else:
                            result["error"] = f"Note: Only first page of {len(file_items)}-page PDF was processed"
                    return QwenVLResponse(**result)
                else:
                    raise HTTPException(status_code=response.status_code, detail=response.text)
                
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Qwen 2.5-VL service timeout")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Qwen VL service error: {e}")
        raise HTTPException(status_code=500, detail=f"Qwen VL service error: {str(e)}")

@app.get("/ocr/qwen-vl/health")
async def qwen_vl_health():
    """Check Qwen 2.5-VL service health"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{QWEN_SERVICE_URL}/health")
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qwen 2.5-VL service unavailable: {str(e)}")

@app.get("/ocr/qwen-vl/schema/{provider}")
async def get_qwen_schema(provider: str):
    """Get schema for specific provider from Qwen service"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{QWEN_SERVICE_URL}/schema/{provider}")
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
                
    except Exception as e:
        logger.error(f"Schema service error: {e}")
        raise HTTPException(status_code=500, detail=f"Schema service error: {str(e)}")

@app.post("/ocr/batch")
async def batch_ocr(files: List[UploadFile] = File(...)):
    """Batch OCR processing - supports images and PDFs"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    results = []
    
    for file in files:
        start_time = time.time()
        try:
            # Validate file type
            await validate_file(file)
            
            # Preprocess file (convert PDF to images if needed)
            file_items = await preprocess_file(file)
            
            # Process each file
            if len(file_items) == 1:
                # Single image or single-page PDF
                content, filename, content_type = file_items[0]
                
                # Compress image if needed (> 1MB)
                original_size_kb = len(content) / 1024
                content, content_type = compress_image_for_ocr(content, filename, content_type)
                compressed_size_kb = len(content) / 1024
                
                if compressed_size_kb < original_size_kb:
                    logger.info(f"Batch file {file.filename} compressed for OCR from {original_size_kb:.1f}KB to {compressed_size_kb:.1f}KB")
                
                async with httpx.AsyncClient(timeout=120.0) as client:
                    files_data = {"file": (filename, content, content_type)}
                    response = await client.post(f"{SURYA_SERVICE_URL}/ocr", files=files_data)
                    
                    if response.status_code == 200:
                        result = response.json()
                        result["processing_time"] = time.time() - start_time
                        results.append(result)
                    else:
                        results.append({
                            "filename": file.filename,
                            "status": "error", 
                            "error": response.text,
                            "processing_time": time.time() - start_time
                        })
            else:
                # Multi-page PDF - concatenate text from all pages
                all_text = []
                total_confidence = 0
                page_count = 0
                
                async with httpx.AsyncClient(timeout=300.0) as client:
                    for content, filename, content_type in file_items:
                        # Compress image if needed (> 1MB)
                        original_size_kb = len(content) / 1024
                        content, content_type = compress_image_for_ocr(content, filename, content_type)
                        compressed_size_kb = len(content) / 1024
                        
                        if compressed_size_kb < original_size_kb:
                            logger.info(f"Batch file {file.filename} page {page_count + 1} compressed for OCR from {original_size_kb:.1f}KB to {compressed_size_kb:.1f}KB")
                        
                        files_data = {"file": (filename, content, content_type)}
                        response = await client.post(f"{SURYA_SERVICE_URL}/ocr", files=files_data)
                        
                        if response.status_code == 200:
                            result = response.json()
                            if result.get("text"):
                                all_text.append(f"[Page {page_count + 1}]\n{result['text']}")
                                if result.get("confidence"):
                                    total_confidence += result["confidence"]
                            page_count += 1
                        else:
                            logger.warning(f"Failed to process page {page_count + 1} of {file.filename}: {response.text}")
                
                # Add aggregated result
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "text": "\n\n".join(all_text) if all_text else None,
                    "confidence": total_confidence / page_count if page_count > 0 else None,
                    "processing_time": time.time() - start_time
                })
                
        except HTTPException as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": e.detail,
                "processing_time": time.time() - start_time
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            })
    
    return results


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)