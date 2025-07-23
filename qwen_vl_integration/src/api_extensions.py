"""API extensions for Qwen VL integration"""

import os
import tempfile
import asyncio
import logging
import time
import psutil
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import torch

from .extractors import QwenVLProcessor
from .models import DEWABill, SEWABill
from .utils import ImagePreprocessor

# Import centralized logging if available
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from api.logging_config import get_logger, log_performance, log_memory_usage, set_request_id
    logger = get_logger(__name__)
except ImportError:
    # Fallback to basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    def log_performance(logger, operation, duration_ms, success=True, **extra):
        logger.info(f"Performance: {operation} - {duration_ms}ms")
    
    def log_memory_usage(logger, context):
        logger.info(f"Memory usage: {context}")
    
    def set_request_id(request_id=None):
        return request_id or "fallback"


class QwenVLResponse(BaseModel):
    """Response model for Qwen VL processing"""
    provider: str
    extracted_data: Dict[str, Any]
    extraction_method: str
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    spatial_features_used: Optional[Dict[str, int]] = None
    

class QwenVLHealthResponse(BaseModel):
    """Health check response for Qwen VL service"""
    status: str
    model_loaded: bool
    device: str
    available_memory_gb: Optional[float] = None


# Global processor instance (lazy loaded)
_processor: Optional[QwenVLProcessor] = None


def get_processor() -> QwenVLProcessor:
    """Get or create the global processor instance"""
    global _processor
    if _processor is None:
        init_start = time.time()
        logger.info("Initializing Qwen VL processor for first time...")
        
        # Log system resources
        log_memory_usage(logger, "Before Qwen VL initialization")
        
        # Choose model based on available resources
        if torch.cuda.is_available():
            model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / 1024**3
            use_quantization = gpu_memory < 8 * 1024**3
            
            logger.info("GPU detected", extra={
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_gb': round(gpu_memory_gb, 2),
                'use_quantization': use_quantization,
                'cuda_version': torch.version.cuda
            })
        else:
            # Use smaller model for CPU
            model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
            use_quantization = False
            
            cpu_info = {
                'cpu_count': psutil.cpu_count(),
                'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 'unknown',
                'ram_gb': round(psutil.virtual_memory().total / 1024**3, 2)
            }
            logger.info("CPU mode - no GPU available", extra=cpu_info)
        
        logger.info(f"Loading model: {model_name}")
        
        try:
            _processor = QwenVLProcessor(
                model_name=model_name,
                use_quantization=use_quantization
            )
            
            # Log successful initialization
            init_duration = (time.time() - init_start) * 1000
            log_memory_usage(logger, "After Qwen VL initialization")
            
            logger.info("Qwen VL processor initialized successfully", extra={
                'init_duration_ms': round(init_duration, 2),
                'model_name': model_name,
                'quantized': use_quantization
            })
            
        except Exception as e:
            init_duration = (time.time() - init_start) * 1000
            logger.error(f"Failed to initialize processor", extra={
                'error': str(e),
                'init_duration_ms': round(init_duration, 2),
                'model_name': model_name
            }, exc_info=True)
            raise
            
    return _processor


def add_qwen_routes(app: FastAPI):
    """Add Qwen VL routes to existing FastAPI app"""
    
    @app.get("/ocr/qwen-vl/health", response_model=QwenVLHealthResponse)
    async def qwen_vl_health():
        """Check Qwen VL service health"""
        logger.info("Health check requested")
        try:
            processor = get_processor()
            
            # Get device info
            if torch.cuda.is_available():
                device = f"cuda:{torch.cuda.current_device()}"
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            else:
                device = "cpu"
                memory_gb = None
            
            logger.info(f"Health check successful - Device: {device}")
            
            return QwenVLHealthResponse(
                status="healthy",
                model_loaded=True,
                device=device,
                available_memory_gb=memory_gb
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return QwenVLHealthResponse(
                status="error",
                model_loaded=False,
                device="unknown",
                available_memory_gb=None
            )
    
    @app.post("/ocr/qwen-vl/process", response_model=QwenVLResponse)
    async def process_with_qwen_vl(
        file: UploadFile = File(...),
        enable_reasoning: bool = Query(True, description="Enable spatial reasoning"),
        provider: Optional[str] = Query(None, description="Provider (DEWA/SEWA)")
    ):
        """
        Process utility bill with Qwen Vision-Language model.
        
        This endpoint combines:
        1. Surya OCR for high-accuracy text extraction
        2. Qwen VL for spatial reasoning and structured extraction
        """
        request_start = time.time()
        request_id = set_request_id()
        temp_path = None
        
        logger.info(f"Qwen VL processing request started", extra={
            'request_id': request_id,
            'file_name': file.filename,
            'enable_reasoning': enable_reasoning,
            'provider': provider
        })
        
        try:
            
            # Validate file
            if not file.content_type or not file.content_type.startswith(('image/', 'application/pdf')):
                logger.error(f"Invalid file type: {file.content_type}")
                raise HTTPException(status_code=400, detail="File must be an image or PDF")
            
            # Save uploaded file
            file_content = await file.read()
            file_size_kb = len(file_content) / 1024
            logger.debug(f"File read complete", extra={
                'size_kb': round(file_size_kb, 2),
                'content_type': file.content_type
            })
            
            suffix = Path(file.filename).suffix or '.png'
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_content)
                temp_path = Path(tmp.name)
            logger.debug(f"Saved to temp file: {temp_path}")
            
            # First, run Surya OCR
            logger.info("Running Surya OCR...")
            ocr_start = time.time()
            
            from api.main import run_surya_ocr
            ocr_result = await asyncio.to_thread(run_surya_ocr, temp_path)
            
            ocr_duration = (time.time() - ocr_start) * 1000
            
            if ocr_result["status"] != "success":
                logger.error(f"OCR failed", extra={
                    'error': ocr_result.get('error'),
                    'duration_ms': round(ocr_duration, 2)
                })
                raise HTTPException(status_code=500, detail=f"OCR failed: {ocr_result.get('error')}")
            
            log_performance(logger, "surya_ocr", ocr_duration, True,
                          confidence=ocr_result.get('confidence'),
                          text_length=len(ocr_result.get('text', '')))
            
            # Apply post-processing to OCR text
            logger.debug("Applying OCR post-processing...")
            postprocess_start = time.time()
            
            from .utils.ocr_postprocessor import process_surya_output
            cleaned_text = process_surya_output(ocr_result["text"])
            
            postprocess_duration = (time.time() - postprocess_start) * 1000
            logger.debug(f"Post-processing complete", extra={
                'duration_ms': round(postprocess_duration, 2),
                'original_length': len(ocr_result["text"]),
                'cleaned_length': len(cleaned_text)
            })
            
            # Load and preprocess image
            preprocessor = ImagePreprocessor()
            image = preprocessor.preprocess(temp_path)
            
            # Process with Qwen VL
            logger.debug("Getting Qwen VL processor...")
            processor = get_processor()
            
            logger.info(f"Starting VLM processing", extra={
                'method': 'reasoning' if enable_reasoning else 'direct',
                'text_length': len(cleaned_text)
            })
            
            vlm_start = time.time()
            
            if enable_reasoning:
                result = await asyncio.to_thread(
                    processor.process_with_reasoning,
                    image,
                    cleaned_text,
                    provider
                )
            else:
                result = await asyncio.to_thread(
                    processor.process_direct,
                    image,
                    cleaned_text,
                    provider
                )
            
            # Check for errors
            if "error" in result:
                logger.error(f"VLM processing error: {result['error']}")
                raise HTTPException(status_code=500, detail=f"VLM processing failed: {result['error']}")
            
            logger.info(f"Qwen VL processing complete - Provider detected: {result.get('provider_name')}")
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare response
            response = QwenVLResponse(
                provider=result.get("provider_name", "Unknown"),
                extracted_data=result,
                extraction_method="qwen_vl_with_reasoning" if enable_reasoning else "qwen_vl_direct",
                confidence=ocr_result.get("confidence"),
                processing_time=processing_time
            )
            
            # Add spatial features info if reasoning was used
            if enable_reasoning and hasattr(processor, 'spatial_reasoner'):
                # This would be populated from the actual reasoning results
                response.spatial_features_used = {
                    "sections_identified": 0,
                    "tables_found": 0,
                    "corrections_made": 0
                }
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Cleanup
            if temp_path and temp_path.exists():
                temp_path.unlink()
                logger.info("Cleaned up temp file")
    
    @app.get("/ocr/qwen-vl/schema/{provider}")
    async def get_qwen_vl_schema(provider: str):
        """Get the Pydantic schema for a specific provider"""
        provider = provider.upper()
        
        if provider == "DEWA":
            schema = DEWABill.model_json_schema()
        elif provider == "SEWA":
            schema = SEWABill.model_json_schema()
        else:
            raise HTTPException(status_code=400, detail="Provider must be DEWA or SEWA")
        
        return JSONResponse(content=schema)
    
    @app.post("/ocr/qwen-vl/extract-text")
    async def extract_text_only(
        file: UploadFile = File(...),
        apply_postprocessing: bool = Query(True)
    ):
        """
        Extract only OCR text with optional post-processing.
        Useful for testing OCR quality.
        """
        temp_path = None
        
        try:
            # Save file
            file_content = await file.read()
            suffix = Path(file.filename).suffix or '.png'
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_content)
                temp_path = Path(tmp.name)
            
            # Run OCR
            from api.main import run_surya_ocr
            ocr_result = await asyncio.to_thread(run_surya_ocr, temp_path)
            
            if ocr_result["status"] != "success":
                raise HTTPException(status_code=500, detail=f"OCR failed: {ocr_result.get('error')}")
            
            # Apply post-processing if requested
            text = ocr_result["text"]
            if apply_postprocessing:
                from .utils.ocr_postprocessor import process_surya_output
                text = process_surya_output(text)
            
            return {
                "text": text,
                "confidence": ocr_result.get("confidence"),
                "processing_time": ocr_result.get("processing_time"),
                "postprocessing_applied": apply_postprocessing
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink()
    
    @app.get("/ocr/qwen-vl/info")
    async def get_qwen_vl_info():
        """Get information about the Qwen VL integration"""
        try:
            processor = get_processor()
            cache_info = processor.cache_manager.get_cache_info()
            
            return {
                "version": "1.0.0",
                "model": processor.model_name,
                "device": processor.device,
                "quantization_enabled": processor.use_quantization,
                "cache_info": cache_info,
                "capabilities": [
                    "spatial_reasoning",
                    "table_extraction",
                    "ocr_correction",
                    "structured_extraction"
                ]
            }
        except Exception as e:
            return {
                "version": "1.0.0",
                "status": "not_initialized",
                "error": str(e)
            }