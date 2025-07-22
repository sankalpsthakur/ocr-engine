"""API extensions for Qwen VL integration"""

import os
import tempfile
import asyncio
import logging
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        logger.info("Initializing Qwen VL processor...")
        
        # Choose model based on available resources
        if torch.cuda.is_available():
            model_name = "Qwen/Qwen2-VL-2B-Instruct"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            use_quantization = gpu_memory < 8 * 1024**3
            logger.info(f"CUDA available. GPU memory: {gpu_memory / 1024**3:.2f}GB")
            logger.info(f"Using quantization: {use_quantization}")
        else:
            # Use smaller model for CPU
            model_name = "Qwen/Qwen2-VL-2B-Instruct"
            use_quantization = False
            logger.info("CUDA not available, using CPU")
        
        logger.info(f"Loading model: {model_name}")
        
        try:
            _processor = QwenVLProcessor(
                model_name=model_name,
                use_quantization=use_quantization
            )
            logger.info("Qwen VL processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize processor: {e}", exc_info=True)
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
        start_time = datetime.now()
        temp_path = None
        
        try:
            logger.info(f"Processing request - File: {file.filename}, Reasoning: {enable_reasoning}, Provider: {provider}")
            
            # Validate file
            if not file.content_type or not file.content_type.startswith(('image/', 'application/pdf')):
                logger.error(f"Invalid file type: {file.content_type}")
                raise HTTPException(status_code=400, detail="File must be an image or PDF")
            
            # Save uploaded file
            file_content = await file.read()
            logger.info(f"File size: {len(file_content) / 1024:.2f}KB")
            suffix = Path(file.filename).suffix or '.png'
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_content)
                temp_path = Path(tmp.name)
            logger.info(f"Saved to temp file: {temp_path}")
            
            # First, run Surya OCR
            logger.info("Running Surya OCR...")
            from api.main import run_surya_ocr
            ocr_result = await asyncio.to_thread(run_surya_ocr, temp_path)
            
            if ocr_result["status"] != "success":
                logger.error(f"OCR failed: {ocr_result.get('error')}")
                raise HTTPException(status_code=500, detail=f"OCR failed: {ocr_result.get('error')}")
            
            logger.info(f"OCR successful - Confidence: {ocr_result.get('confidence')}, Text length: {len(ocr_result.get('text', ''))}")
            
            # Apply post-processing to OCR text
            logger.info("Applying OCR post-processing...")
            from .utils.ocr_postprocessor import process_surya_output
            cleaned_text = process_surya_output(ocr_result["text"])
            logger.info(f"Post-processing complete - Cleaned text length: {len(cleaned_text)}")
            
            # Load and preprocess image
            preprocessor = ImagePreprocessor()
            image = preprocessor.preprocess(temp_path)
            
            # Process with Qwen VL
            logger.info("Getting Qwen VL processor...")
            processor = get_processor()
            
            logger.info(f"Processing with Qwen VL - Method: {'reasoning' if enable_reasoning else 'direct'}")
            
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