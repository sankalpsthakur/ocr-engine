#!/usr/bin/env python3
"""
Qwen 2.5-VL 3B Instruct Microservice
Isolated service running latest transformers for Qwen 2.5-VL compatibility
"""

import os
import sys
import time
import tempfile
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union
import asyncio
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Qwen 2.5-VL components
try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    from qwen_vl_utils import process_vision_info
    logger.info("✓ Qwen 2.5-VL 3B Instruct imported successfully")
except ImportError as e:
    logger.error(f"✗ Failed to import Qwen 2.5-VL: {e}")
    sys.exit(1)

app = FastAPI(title="Qwen 2.5-VL 3B Instruct Service", version="1.0.0")

# Global model components
model = None
processor = None

class QwenVLResponse(BaseModel):
    filename: str
    status: str
    provider: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    raw_text: Optional[str] = None
    extraction_method: str = "qwen-2.5-vl-3b-instruct"
    confidence: Optional[float] = None
    processing_time: float
    error: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Load Qwen 2.5-VL models on startup"""
    global model, processor
    
    logger.info("Loading Qwen 2.5-VL 3B Instruct models...")
    try:
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        
        # Load model with proper configuration (force CPU for stability)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        
        # Load processor
        processor = AutoProcessor.from_pretrained(model_name)
        
        logger.info("✓ Qwen 2.5-VL 3B Instruct models loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load Qwen 2.5-VL models: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "qwen-2.5-vl-3b-instruct",
        "models_loaded": model is not None
    }

@app.post("/process", response_model=QwenVLResponse)
async def process_document(
    file: UploadFile = File(...),
    resource_type: str = Query(default="utility", description="Type of resource (water, energy, utility)"),
    enable_reasoning: bool = Query(default=True, description="Enable spatial reasoning"),
    ocr_text: str = Query(default="", description="Pre-extracted OCR text from Surya")
):
    """Process document with Qwen 2.5-VL 3B Instruct for structured extraction"""
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    
    try:
        logger.info(f"[{request_id}] Starting Qwen VL processing for file: {file.filename}")
        logger.info(f"[{request_id}] Resource type: {resource_type}, Enable reasoning: {enable_reasoning}")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            logger.error(f"[{request_id}] Invalid file type: {file.content_type}")
            raise HTTPException(status_code=400, detail="File must be an image")
        
        logger.info(f"[{request_id}] File validation passed. Content type: {file.content_type}")
        
        # Read file content
        content = await file.read()
        logger.info(f"[{request_id}] File content read successfully. Size: {len(content)} bytes")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        logger.info(f"[{request_id}] Temporary file created: {tmp_path}")
        
        try:
            # Load and process image
            logger.info(f"[{request_id}] Loading image with PIL...")
            image = Image.open(tmp_path)
            logger.info(f"[{request_id}] Image loaded. Mode: {image.mode}, Size: {image.size}")
            
            if image.mode == 'RGBA':
                logger.info(f"[{request_id}] Converting RGBA to RGB...")
                image = image.convert('RGB')
                logger.info(f"[{request_id}] Conversion complete. New mode: {image.mode}")
            
            # Create prompt based on resource type
            logger.info(f"[{request_id}] Creating prompt for resource type: {resource_type}")
            logger.info(f"[{request_id}] OCR text provided: {len(ocr_text)} characters")
            
            # Include OCR text in prompt if available
            ocr_context = f"\n\nOCR extracted text:\n{ocr_text}\n\n" if ocr_text else ""
            
            if resource_type == "water":
                prompt = f"""{ocr_context}Extract water utility bill information from the image and OCR text including:
- Account number
- Billing period
- Water consumption in gallons/cubic meters
- Total amount due
- Due date
- Service address
Use both the visual information and OCR text to ensure accuracy. Return as structured JSON."""
            elif resource_type == "energy":
                prompt = f"""{ocr_context}Extract energy utility bill information from the image and OCR text including:
- Account number
- Billing period  
- Energy consumption in kWh
- Total amount due
- Due date
- Service address
- Carbon emissions data if available
Use both the visual information and OCR text to ensure accuracy. Return as structured JSON."""
            else:
                prompt = f"""{ocr_context}Extract all utility bill information from the image and OCR text including account details, consumption, charges, and dates. Use both the visual information and OCR text to ensure accuracy. Return as structured JSON."""
            
            logger.info(f"[{request_id}] Prompt created. Length: {len(prompt)} characters")
            
            # Prepare messages for Qwen 2.5-VL
            logger.info(f"[{request_id}] Preparing messages for Qwen 2.5-VL...")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": tmp_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            logger.info(f"[{request_id}] Messages prepared. Message count: {len(messages)}")
            
            # Process the input
            logger.info(f"[{request_id}] Applying chat template...")
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            logger.info(f"[{request_id}] Chat template applied. Text length: {len(text)}")
            
            logger.info(f"[{request_id}] Processing vision info...")
            image_inputs, video_inputs = process_vision_info(messages)
            logger.info(f"[{request_id}] Vision info processed. Image inputs type: {type(image_inputs)}")
            if image_inputs:
                logger.info(f"[{request_id}] Image inputs length: {len(image_inputs) if isinstance(image_inputs, list) else 'Not a list'}")
            
            logger.info(f"[{request_id}] Running processor...")
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            logger.info(f"[{request_id}] Processor completed. Inputs type: {type(inputs)}")
            logger.info(f"[{request_id}] Inputs keys: {list(inputs.keys()) if hasattr(inputs, 'keys') else 'No keys method'}")
            
            # Debug inputs structure
            if hasattr(inputs, '__dict__'):
                logger.info(f"[{request_id}] Inputs attributes: {list(inputs.__dict__.keys())}")
            
            # Check if inputs has the expected attributes
            for attr in ['input_ids', 'attention_mask', 'pixel_values']:
                has_attr = hasattr(inputs, attr)
                logger.info(f"[{request_id}] inputs.{attr}: {has_attr}")
                if has_attr:
                    attr_value = getattr(inputs, attr)
                    logger.info(f"[{request_id}] inputs.{attr} type: {type(attr_value)}")
                    if hasattr(attr_value, 'shape'):
                        logger.info(f"[{request_id}] inputs.{attr} shape: {attr_value.shape}")
            
            # Alternative approach - check if inputs is a dict
            if isinstance(inputs, dict):
                logger.info(f"[{request_id}] Inputs is a dict with keys: {list(inputs.keys())}")
                for key, value in inputs.items():
                    logger.info(f"[{request_id}] inputs['{key}'] type: {type(value)}")
                    if hasattr(value, 'shape'):
                        logger.info(f"[{request_id}] inputs['{key}'] shape: {value.shape}")
                    if hasattr(value, 'device'):
                        logger.info(f"[{request_id}] inputs['{key}'] device: {value.device}")
            
            # Ensure inputs are on the same device as the model (CUDA)
            logger.info(f"[{request_id}] Moving inputs to CUDA...")
            inputs = inputs.to('cuda')
            
            logger.info(f"[{request_id}] Starting model generation...")
            # Generate response
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            logger.info(f"[{request_id}] Model generation completed. Generated IDs shape: {generated_ids.shape}")
            
            logger.info(f"[{request_id}] Trimming generated IDs...")
            # Access input_ids correctly based on inputs structure
            if isinstance(inputs, dict):
                input_ids = inputs['input_ids']
            else:
                input_ids = inputs.input_ids
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
            ]
            logger.info(f"[{request_id}] Generated IDs trimmed. Count: {len(generated_ids_trimmed)}")
            
            logger.info(f"[{request_id}] Decoding output text...")
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            logger.info(f"[{request_id}] Output text decoded. Length: {len(output_text)}")
            logger.info(f"[{request_id}] Output preview: {output_text[:200]}...")
            
            # Try to parse as JSON, fallback to raw text
            logger.info(f"[{request_id}] Attempting to parse output as JSON...")
            try:
                extracted_data = json.loads(output_text)
                logger.info(f"[{request_id}] JSON parsing successful")
            except json.JSONDecodeError as json_err:
                logger.warning(f"[{request_id}] JSON parsing failed: {json_err}")
                extracted_data = {"raw_response": output_text}
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"[{request_id}] Processing completed successfully. Time: {processing_time:.2f}ms")
            
            return QwenVLResponse(
                filename=file.filename,
                status="success",
                provider="DEWA" if "dewa" in file.filename.lower() else "SEWA" if "sewa" in file.filename.lower() else "Unknown",
                extracted_data=extracted_data,
                raw_text=output_text,
                processing_time=processing_time
            )
                
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                logger.info(f"[{request_id}] Cleaning up temporary file: {tmp_path}")
                os.unlink(tmp_path)
    
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.error(f"[{request_id}] Qwen VL processing failed: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"[{request_id}] Full traceback:\n{traceback.format_exc()}")
        
        return QwenVLResponse(
            filename=file.filename,
            status="error",
            error=str(e),
            processing_time=processing_time
        )

@app.get("/schema/{provider}")
async def get_schema(provider: str):
    """Get schema for specific provider"""
    schemas = {
        "DEWA": {
            "account_number": "string",
            "billing_period": "string", 
            "consumption": "number",
            "total_amount": "number",
            "due_date": "string",
            "service_address": "string"
        },
        "SEWA": {
            "account_number": "string",
            "billing_period": "string",
            "consumption": "number", 
            "total_amount": "number",
            "due_date": "string",
            "service_address": "string"
        }
    }
    
    if provider.upper() not in schemas:
        raise HTTPException(status_code=404, detail=f"Schema for {provider} not found")
    
    return schemas[provider.upper()]

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)