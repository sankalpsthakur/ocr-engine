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
from typing import Dict, Any, Optional, List
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service URLs
SURYA_SERVICE_URL = os.getenv("SURYA_SERVICE_URL", "http://localhost:8001")
QWEN_SERVICE_URL = os.getenv("QWEN_SERVICE_URL", "http://localhost:8002")

app = FastAPI(
    title="OCR Engine API Gateway",
    description="Unified API for Surya OCR and Qwen 2.5-VL 3B Instruct processing",
    version="2.0.0"
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
    """Basic OCR using Surya OCR service"""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            files = {"file": (file.filename, await file.read(), file.content_type)}
            response = await client.post(f"{SURYA_SERVICE_URL}/ocr", files=files)
            
            if response.status_code == 200:
                return OCRResponse(**response.json())
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
                
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Surya OCR service timeout")
    except Exception as e:
        logger.error(f"OCR service error: {e}")
        raise HTTPException(status_code=500, detail=f"OCR service error: {str(e)}")

@app.post("/ocr/qwen-vl/process", response_model=QwenVLResponse)
async def qwen_vl_process(
    file: UploadFile = File(...),
    resource_type: str = Query(default="utility", description="Type of resource (water, energy, utility)"),
    enable_reasoning: bool = Query(default=True, description="Enable spatial reasoning")
):
    """Process document with Qwen 2.5-VL 3B Instruct"""
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            files = {"file": (file.filename, await file.read(), file.content_type)}
            params = {"resource_type": resource_type, "enable_reasoning": enable_reasoning}
            
            response = await client.post(
                f"{QWEN_SERVICE_URL}/process", 
                files=files, 
                params=params
            )
            
            if response.status_code == 200:
                return QwenVLResponse(**response.json())
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
                
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Qwen 2.5-VL service timeout")
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
    """Batch OCR processing"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    results = []
    async with httpx.AsyncClient(timeout=300.0) as client:
        for file in files:
            try:
                files_data = {"file": (file.filename, await file.read(), file.content_type)}
                response = await client.post(f"{SURYA_SERVICE_URL}/ocr", files=files_data)
                
                if response.status_code == 200:
                    results.append(response.json())
                else:
                    results.append({
                        "filename": file.filename,
                        "status": "error", 
                        "error": response.text,
                        "processing_time": 0
                    })
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e),
                    "processing_time": 0
                })
    
    return results

@app.on_event("startup")
async def startup_event():
    """Check services on startup"""
    logger.info("API Gateway starting up...")
    
    # Wait a moment for services to be ready
    await asyncio.sleep(2)
    
    # Check service health
    surya_health = await check_service_health(SURYA_SERVICE_URL, "Surya OCR")
    qwen_health = await check_service_health(QWEN_SERVICE_URL, "Qwen 2.5-VL")
    
    logger.info(f"Surya OCR Service: {surya_health}")
    logger.info(f"Qwen 2.5-VL Service: {qwen_health}")
    
    if surya_health == "unreachable":
        logger.warning("Surya OCR service is not available")
    if qwen_health == "unreachable":
        logger.warning("Qwen 2.5-VL service is not available")
    
    logger.info("API Gateway startup complete")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)