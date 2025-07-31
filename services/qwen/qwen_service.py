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

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
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
device = None

def parse_json_response(text: str) -> dict:
    """Parse JSON response, handling markdown code blocks and other formatting"""
    # Remove markdown code blocks if present
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    
    # Strip whitespace
    text = text.strip()
    
    # Try to parse JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # If parsing fails, return the raw text
        return {"raw_response": text}

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
    global model, processor, device
    
    logger.info("Loading Qwen 2.5-VL 3B Instruct models...")
    try:
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        
        # Detect best available device
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("Using CUDA device")
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using MPS (Metal Performance Shaders) device")
        else:
            device = "cpu"
            logger.info("Using CPU device")
        
        # Load model with proper configuration
        if device == "cuda":
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="cuda"
            )
        elif device == "mps":
            # MPS requires special handling - load to CPU first then move
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            )
            model = model.to(device)
        else:
            # CPU requires float32
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            )
            model = model.to(device)
        
        # Load processor
        processor = AutoProcessor.from_pretrained(model_name)
        
        logger.info(f"✓ Qwen 2.5-VL 3B Instruct models loaded successfully on {device}")
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
    # ocr_text: Optional[str] = Form(None),
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
        logger.info(f"[{request_id}] OCR text provided: {len(ocr_text) if ocr_text else 0} characters")
        
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

                if ocr_text:
                    # Build prompt without f-string to avoid issues with curly braces in OCR text
                    prompt = """You are an expert at extracting structured data from water utility bills.

IMPORTANT: Extract information ONLY from the OCR text provided below. The image is provided for spatial context only - to help understand the layout and positioning of text elements.

OCR Text from the bill:
---
""" + ocr_text + """
---

Based on the OCR TEXT ABOVE, extract the following information:
- Account number
- Billing period
- Water consumption in gallons/cubic meters
- Total amount due
- Due date
- Service address

Return as structured JSON."""
                else:
                    prompt = """Extract water utility bill information including:
- Account number
- Billing period
- Water consumption in gallons/cubic meters
- Total amount due
- Due date
- Service address
Use both the visual information and OCR text to ensure accuracy. Return as structured JSON."""
            elif resource_type == "energy":
                if ocr_text:
                    # Build prompt without f-string to avoid issues with curly braces in OCR text
                    prompt = """You are an expert at extracting structured data from energy utility bills.

CRITICAL INSTRUCTIONS - RESEARCH MODE:

You are in RESEARCH MODE. Your task is to be EXTREMELY THOROUGH and ACCURATE.

GOLDEN RULES:
1. ONLY extract information that is EXPLICITLY written in the OCR text below
2. If a field is not clearly mentioned, return null - DO NOT GUESS
3. DO NOT HALLUCINATE - If you don't see "kWh" written, don't assume it
4. Be a detective - look for exact text matches, not assumptions
5. When in doubt, return null rather than guessing

IMPORTANT: I will verify every value you extract against the OCR text. Any hallucinated data will be considered a failure.

OCR Text from the bill:
---
""" + ocr_text + """
---

Based on the OCR TEXT ABOVE, extract and standardize the following information:

REQUIRED FIELDS (always include these):
1. consumption: The total energy consumed (numeric value only, no units)
2. start_date: Beginning of the billing period in "YYYY-MM-DD" format
3. end_date: End of the billing period in "YYYY-MM-DD" format  
4. unit_of_consumption: The unit of measurement (e.g., "kWh", "MWh", "units", "therms", "m³")
5. source_of_energy: Type of energy (e.g., "electricity", "gas", "solar", "mixed")
6. carbon_footprint: CO2 emissions in kg (numeric only), or null if not mentioned

IMPORTANT: Return ONLY valid JSON without markdown formatting, code blocks, or explanations.

DETAILED EXTRACTION RULES:

1. DATE DEDUCTION (Critical - bills rarely show dates in the format we need):
   
   When you see "Billing Period: 14/04/2025 to 13/05/2025":
   → Extract start_date: "2025-04-14", end_date: "2025-05-13"
   
   When you see only month/year like "FEB 2020" or "February 2020":
   → Assume full month: start_date: "2020-02-01", end_date: "2020-02-29" (check leap years!)
   → 2020 is a leap year, so February has 29 days
   → 2021 is not a leap year, so February has 28 days
   
   When you see "Billing Cycle: 15 JAN - 14 FEB 2023":
   → start_date: "2023-01-15", end_date: "2023-02-14"
   
   When you see quarters like "Q1 2020" or "First Quarter 2020":
   → Q1: start_date: "2020-01-01", end_date: "2020-03-31"
   → Q2: start_date: "2020-04-01", end_date: "2020-06-30"
   → Q3: start_date: "2020-07-01", end_date: "2020-09-30"
   → Q4: start_date: "2020-10-01", end_date: "2020-12-31"

2. CONSUMPTION EXTRACTION - NO HALLUCINATION:

   STEP 1: Search for EXACT consumption indicators in the OCR text:
   - Look for the word "Consumption" or "استهلاك"
   - Look for "Usage" or "Used"
   - Look for numeric values near these words

   STEP 2: Verify the number is actually consumption:
   - Is there a number next to "Consumption"? → Extract it
   - No clear consumption label? → Return null
   - See "Total Amount" or "Due"? → That's money, NOT consumption

   STEP 3: For MULTI-UTILITY bills (electricity + water + gas):
   - Look for "Meter & Reading Details" section
   - Find the consumption column in the table
   - For bills with multiple meters:
     * E-xxxxx meters = Electricity
     * W-xxxxx meters = Water
     * G-xxxxx meters = Gas
   - Extract the PRIMARY utility (usually electricity) or return null if unclear

   EXAMPLES:
   ✓ "Consumption: 92" → consumption: 92
   ✓ "Electricity Usage: 634 kWh" → consumption: 634
   ✗ "Total Amount Due: 90981" → This is money, NOT consumption (return null)
   ✗ No consumption mentioned → consumption: null

3. UNIT OF CONSUMPTION - ONLY FROM OCR TEXT:

   STRICT RULE: The unit MUST be written in the OCR text. DO NOT ASSUME.

   STEP 1: Find the consumption value first
   STEP 2: Look for units NEAR that value in the OCR text
   STEP 3: Common units to SEARCH FOR (not assume):
   - Electricity: "kWh", "KWH", "units", "يونت"
   - Water: "gallons", "m³", "cubic meters", "liters"
   - Gas: "therms", "m³", "cubic feet"

   EXAMPLES:
   ✓ OCR has "993 kWh" → unit_of_consumption: "kWh"
   ✓ OCR has "Consumption: 299 units" → unit_of_consumption: "units"
   ✗ OCR has just "932" with no unit → unit_of_consumption: null
   ✗ You think it should be kWh → NO! Return null if not in text

   SPECIAL CASE - Meter Reading Tables:
   If you see a table with "Consumption" column but NO unit specified → unit_of_consumption: null

4. SOURCE OF ENERGY DETECTION - FROM OCR TEXT:

   Look for EXPLICIT mentions:
   - "Electricity" or "كهرباء" → source_of_energy: "electricity"
   - "Gas" or "غاز" → source_of_energy: "gas"
   - "Water" or "مياه" → (not energy, but indicates mixed utility)
   - Multiple utilities (E+W+G meters) → source_of_energy: "mixed"
   - "Sharjah Electricity & Water Authority" → source_of_energy: "mixed"
   - No clear indication → source_of_energy: null

5. CARBON FOOTPRINT EXTRACTION:
   
   Look for EXPLICIT carbon/CO2 mentions:
   - "Carbon Emissions: 125.58 kg CO2" → carbon_footprint: 125.58
   - "CO2: 0.126 tonnes" → carbon_footprint: 126 (convert tonnes to kg)
   - Not mentioned → carbon_footprint: null

VERIFICATION CHECKLIST - BEFORE RETURNING JSON:

□ consumption: Can I point to the EXACT word "Consumption" or similar in OCR?
□ unit_of_consumption: Is this unit WRITTEN in the OCR text?
□ dates: Are these dates ACTUALLY written or am I deducing correctly?
□ source_of_energy: Is this type mentioned or clearly indicated?
□ carbon_footprint: Is CO2/carbon explicitly mentioned?

If you cannot find explicit text evidence → Set field to null

REAL-WORLD EXAMPLES:

Example 1 - Explicit dates:
Input: "Service Period: 01/03/2024 - 31/03/2024, Electricity Usage: 450 kWh, CO2 Emissions: 189 kg"
Output: {
  "consumption": 450,
  "start_date": "2024-03-01", 
  "end_date": "2024-03-31",
  "unit_of_consumption": "kWh",
  "source_of_energy": "electricity",
  "carbon_footprint": 189
}

Example 2 - Month only:
Input: "DEWA Bill for March 2024, Total Units Consumed: 523"
Output: {
  "consumption": 523,
  "start_date": "2024-03-01",
  "end_date": "2024-03-31",
  "unit_of_consumption": "units",
  "source_of_energy": "electricity",
  "carbon_footprint": null
}

Example 3 - Gas bill with therms:
Input: "Natural Gas Statement, Period: FEB 2023, Usage: 67 therms"
Output: {
  "consumption": 67,
  "start_date": "2023-02-01",
  "end_date": "2023-02-28",
  "unit_of_consumption": "therms",
  "source_of_energy": "gas",
  "carbon_footprint": null
}

Example 4 - Quarterly bill:
Input: "Q3 2023 Electric Bill, 3,456.78 kWh consumed, Carbon: 1.45 tons"
Output: {
  "consumption": 3456.78,
  "start_date": "2023-07-01",
  "end_date": "2023-09-30",
  "unit_of_consumption": "kWh",
  "source_of_energy": "electricity",
  "carbon_footprint": 1450
}

Example 5 - SEWA Multi-utility (CRITICAL):
Input: OCR shows table with "Meter No." and "Consumption" columns:
"E-56151545    472
 W-13A011272   1980
 G-60399       143"
Output: {
  "consumption": 472,  // Electricity consumption only
  "start_date": null,  // Extract from billing period if shown
  "end_date": null,
  "unit_of_consumption": null,  // No unit shown in table
  "source_of_energy": "mixed",  // Has E+W+G meters
  "carbon_footprint": null
}

Example 6 - Missing unit (DO NOT HALLUCINATE):
Input: "Meter Reading Details... Consumption: 472"
Output: {
  "consumption": 472,
  "unit_of_consumption": null,  // No unit mentioned! Don't assume kWh
  "source_of_energy": null,  // No type mentioned
  ...
}

Example 7 - No consumption found:
Input: "Total Amount Due: 582.03 AED, VAT: 25.87"
Output: {
  "consumption": null,  // No consumption mentioned!
  "unit_of_consumption": null,
  "source_of_energy": null,
  "carbon_footprint": null,
  ...
}"""
                else:
                    # Original prompt when no OCR text is provided
                    prompt = """You are an expert at extracting and intelligently deducing structured data from energy utility bills. Energy bills come in many formats - some show explicit date ranges, others only show month/year, and some use billing cycles or quarters.

Your task is to extract and standardize the following information:

REQUIRED FIELDS (always include these):
1. consumption: The total energy consumed (numeric value only, no units)
2. start_date: Beginning of the billing period in "YYYY-MM-DD" format
3. end_date: End of the billing period in "YYYY-MM-DD" format  
4. unit_of_consumption: The unit of measurement (e.g., "kWh", "MWh", "units", "therms", "m³")
5. source_of_energy: Type of energy (e.g., "electricity", "gas", "solar", "mixed")
6. carbon_footprint: CO2 emissions in kg (numeric only), or null if not mentioned

IMPORTANT: Return ONLY valid JSON without markdown formatting, code blocks, or explanations."""
            else:
                if ocr_text:
                    # Build prompt without f-string to avoid issues with curly braces in OCR text
                    prompt = """You are an expert at extracting structured data from utility bills.

IMPORTANT: Extract information ONLY from the OCR text provided below. The image is provided for spatial context only - to help understand the layout and positioning of text elements.

OCR Text from the bill:
---
""" + ocr_text + """
---

Based on the OCR TEXT ABOVE, extract all utility bill information including account details, consumption, charges, and dates. Return as structured JSON."""
                else:
                    prompt = """Extract all utility bill information including account details, consumption, charges, and dates. Return as structured JSON."""
            
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
            
            # Ensure inputs are on the same device as the model
            logger.info(f"[{request_id}] Moving inputs to {device}...")
            inputs = inputs.to(device)
            
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
            extracted_data = parse_json_response(output_text)
            if "raw_response" in extracted_data:
                logger.warning(f"[{request_id}] Could not parse JSON from output")
            else:
                logger.info(f"[{request_id}] JSON parsing successful")
            
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