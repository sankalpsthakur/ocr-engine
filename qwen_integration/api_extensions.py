"""API extensions for Qwen-enhanced OCR processing"""

import io
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from PIL import Image

# Import the Qwen pipeline
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from qwen_integration.src import OCRPipeline


# Create API router
router = APIRouter(prefix="/ocr/qwen", tags=["Qwen OCR"])

# Global pipeline instance
pipeline: Optional[OCRPipeline] = None


class QwenOCRResponse(BaseModel):
    """Response model for Qwen-enhanced OCR"""
    filename: str
    status: str
    provider: str = Field(description="Detected provider (DEWA/SEWA)")
    extracted_data: Optional[Dict[str, Any]] = Field(description="Structured extracted data")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence score")
    manual_verification_required: bool
    processing_time: float = Field(description="Total processing time in seconds")
    error: Optional[str] = None


class QwenHealthResponse(BaseModel):
    """Health check response for Qwen endpoints"""
    status: str
    qwen_model_loaded: bool
    model_name: str
    device: str


def get_pipeline() -> OCRPipeline:
    """Get or create the OCR pipeline instance"""
    global pipeline
    if pipeline is None:
        pipeline = OCRPipeline()
        pipeline.initialize()
    return pipeline


def cleanup_temp_file(file_path: str):
    """Clean up temporary file"""
    try:
        if Path(file_path).exists():
            Path(file_path).unlink()
    except Exception as e:
        print(f"Error cleaning up {file_path}: {e}")


@router.get("/health", response_model=QwenHealthResponse)
async def qwen_health_check():
    """Check Qwen OCR service health"""
    try:
        p = get_pipeline()
        return QwenHealthResponse(
            status="healthy",
            qwen_model_loaded=p.qwen_processor.is_loaded,
            model_name=p.qwen_processor.model_name,
            device=p.qwen_processor.device
        )
    except Exception as e:
        return QwenHealthResponse(
            status="error",
            qwen_model_loaded=False,
            model_name="",
            device="",
            error=str(e)
        )


@router.post("/process", response_model=QwenOCRResponse)
async def process_with_qwen(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    provider: Optional[str] = None
):
    """Process a utility bill with Surya OCR and Qwen extraction
    
    This endpoint:
    1. Extracts text using Surya OCR
    2. Uses Qwen3-0.6B to extract structured data
    3. Returns validated data matching DEWA/SEWA schemas
    """
    temp_path = None
    start_time = datetime.now()
    
    try:
        # Validate file
        if file.size > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=413, detail="File too large")
            
        # Save to temporary file
        suffix = Path(file.filename).suffix or '.png'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = Path(tmp.name)
            
        # Get pipeline
        p = get_pipeline()
        
        # Process the bill
        try:
            bill = p.process_bill(
                temp_path,
                provider=provider
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Schedule cleanup
            background_tasks.add_task(cleanup_temp_file, str(temp_path))
            
            return QwenOCRResponse(
                filename=file.filename,
                status="success",
                provider=bill.extracted_data.bill_info.provider_name.split()[0],  # DEWA or SEWA
                extracted_data=bill.model_dump()["extracted_data"],
                confidence=bill.validation.confidence,
                manual_verification_required=bill.validation.manual_verification_required,
                processing_time=processing_time
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        if temp_path:
            background_tasks.add_task(cleanup_temp_file, str(temp_path))
        
        return QwenOCRResponse(
            filename=file.filename,
            status="error",
            provider="UNKNOWN",
            confidence=0.0,
            manual_verification_required=True,
            processing_time=(datetime.now() - start_time).total_seconds(),
            error=str(e)
        )


@router.post("/extract-text")
async def extract_text_only(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    apply_postprocessing: bool = True
):
    """Extract only OCR text without structured extraction
    
    Useful for debugging or when you only need the raw text.
    """
    temp_path = None
    
    try:
        # Save to temporary file
        suffix = Path(file.filename).suffix or '.png'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = Path(tmp.name)
            
        # Get pipeline
        p = get_pipeline()
        
        # Extract text only
        if apply_postprocessing:
            text, metadata = p.surya_extractor.extract_with_postprocessing(temp_path)
        else:
            text, metadata = p.surya_extractor.extract_text(temp_path)
            
        # Detect provider
        provider = p.detect_provider(temp_path, text)
        
        # Validate extraction
        validations = p.surya_extractor.validate_extraction(text, provider)
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, str(temp_path))
        
        return {
            "filename": file.filename,
            "status": "success",
            "text": text,
            "provider": provider,
            "metadata": metadata,
            "validations": validations
        }
        
    except Exception as e:
        if temp_path:
            background_tasks.add_task(cleanup_temp_file, str(temp_path))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schema/{provider}")
async def get_provider_schema(provider: str):
    """Get the Pydantic schema for a specific provider
    
    Args:
        provider: Provider name (DEWA or SEWA)
        
    Returns:
        JSON schema for the provider's bill structure
    """
    provider = provider.upper()
    
    if provider == "DEWA":
        from qwen_integration.src.models import DEWABill
        return DEWABill.model_json_schema()
    elif provider == "SEWA":
        from qwen_integration.src.models import SEWABill
        return SEWABill.model_json_schema()
    else:
        raise HTTPException(status_code=400, detail="Invalid provider. Use DEWA or SEWA")


# Function to add these routes to the main FastAPI app
def add_qwen_routes(app):
    """Add Qwen routes to an existing FastAPI app"""
    app.include_router(router)
    
    # Add startup event to initialize pipeline
    @app.on_event("startup")
    async def initialize_qwen():
        """Initialize Qwen pipeline on startup"""
        print("Initializing Qwen OCR pipeline...")
        try:
            get_pipeline()
            print("Qwen pipeline initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize Qwen pipeline: {e}")
            print("Pipeline will be initialized on first request")
    
    # Add shutdown event to cleanup
    @app.on_event("shutdown")
    async def cleanup_qwen():
        """Cleanup Qwen resources on shutdown"""
        global pipeline
        if pipeline:
            pipeline.cleanup()
            pipeline = None