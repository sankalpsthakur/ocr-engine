"""Main Qwen Vision-Language processor with spatial reasoning capabilities"""

import json
import torch
import time
import psutil
import os
from typing import Dict, Any, Optional, Union, Tuple, List
from PIL import Image
import logging
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from pydantic import BaseModel
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None
    import warnings
    warnings.warn(
        "qwen_vl_utils not installed. This is required for Qwen2-VL models. "
        "Please install with: pip install qwen-vl-utils",
        ImportWarning
    )

from qwen_vl_integration.src.models import DEWABill, SEWABill, EnergyBill, WaterBill, WasteBill
from qwen_vl_integration.src.utils.prompt_builder import PromptBuilder
from qwen_vl_integration.src.utils.cache_manager import ModelCacheManager
from qwen_vl_integration.src.extractors.spatial_reasoner import SpatialReasoner

# Import centralized logging if available, otherwise use basic logging
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from api.logging_config import get_logger, log_performance, log_memory_usage
    logger = get_logger(__name__)
except ImportError:
    # Fallback to basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Define fallback functions
    def log_performance(logger, operation, duration_ms, success=True, **extra):
        logger.info(f"Performance: {operation} - {duration_ms}ms")
    
    def log_memory_usage(logger, context):
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage ({context}): {memory_mb:.2f}MB")


class QwenVLProcessor:
    """
    Combines Surya OCR text with visual understanding for superior extraction.
    
    Key capabilities:
    1. Visual grounding of OCR text (match text to image locations)
    2. Table/structure understanding (rows, columns, alignments)
    3. Semantic field mapping (understand "Total" vs line items)
    4. Error correction using visual context
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", 
                 device: str = None,
                 use_quantization: bool = True):
        """
        Initialize Qwen VL processor.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use (cuda/cpu/auto)
            use_quantization: Whether to use 4-bit quantization for efficiency
        """
        init_start = time.time()
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_quantization = use_quantization and torch.cuda.is_available()
        
        logger.info(f"Initializing QwenVLProcessor", extra={
            'model_name': model_name,
            'device': self.device,
            'use_quantization': self.use_quantization,
            'cuda_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        })
        
        # Initialize components
        self.cache_manager = ModelCacheManager()
        self.prompt_builder = PromptBuilder()
        self.spatial_reasoner = SpatialReasoner()
        
        # Check if qwen_vl_utils is available
        if process_vision_info is None:
            logger.error("qwen_vl_utils not available")
            raise ImportError("qwen_vl_utils is required. Install with: pip install qwen-vl-utils")
        
        # Load model
        self._load_model()
        
        init_duration = (time.time() - init_start) * 1000
        logger.info(f"QwenVLProcessor initialized", extra={
            'init_duration_ms': round(init_duration, 2)
        })
        
    def _load_model(self):
        """Load Qwen VL model with caching and optimization"""
        load_start = time.time()
        
        try:
            logger.info(f"Starting model load process")
            
            # Log memory before loading
            log_memory_usage(logger, "Before model loading")
            
            # Check disk space
            import shutil
            cache_dir = Path.home() / ".cache" / "huggingface"
            if cache_dir.exists():
                free_space_gb = shutil.disk_usage(cache_dir).free / 1024 / 1024 / 1024
                logger.info(f"Free disk space for cache: {free_space_gb:.2f}GB")
            
            # Check cache first
            cached_model = self.cache_manager.get_model(self.model_name)
            if cached_model:
                self.model, self.processor = cached_model
                logger.info(f"Model loaded from cache", extra={
                    'cache_hit': True,
                    'model_name': self.model_name
                })
                return
            
            logger.info(f"Model not in cache, downloading {self.model_name}...")
            logger.info("This may take several minutes on first run...")
            
            # Log cache directory info
            cache_info = self.cache_manager.get_cache_info()
            logger.info("Cache directory info", extra=cache_info)
            
            # Model loading kwargs
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                "device_map": "auto" if self.device == "auto" else None,
            }
            
            # Add quantization if enabled
            if self.use_quantization:
                model_kwargs["load_in_4bit"] = True
                model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
                logger.info("Using 4-bit quantization for memory efficiency")
            
            logger.info(f"Model kwargs: {model_kwargs}")
            
            # Load model
            logger.info("Loading model from HuggingFace...")
            model_load_start = time.time()
            
            try:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                model_load_time = (time.time() - model_load_start) * 1000
                logger.info("Model loaded successfully", extra={
                    'load_time_ms': round(model_load_time, 2),
                    'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024
                })
            except Exception as e:
                logger.error(f"Model loading failed: {e}", exc_info=True)
                raise
            
            # Load processor
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            logger.info("Processor loaded successfully")
            
            # Move to device if not using device_map
            if self.device != "auto" and not self.use_quantization:
                logger.info(f"Moving model to {self.device}")
                self.model = self.model.to(self.device)
            
            # Cache the model
            self.cache_manager.cache_model(self.model_name, (self.model, self.processor))
            
            # Log memory after loading
            log_memory_usage(logger, "After model loading")
            
            load_duration = (time.time() - load_start) * 1000
            logger.info(f"Model loading complete", extra={
                'total_duration_ms': round(load_duration, 2),
                'model_name': self.model_name,
                'device': str(self.device),
                'quantized': self.use_quantization
            })
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise
    
    def process_with_reasoning(self, 
                             image: Union[str, Path, Image.Image], 
                             ocr_text: str, 
                             provider: str = None,
                             resource_type: str = None) -> Dict[str, Any]:
        """
        Process image with chain-of-thought spatial reasoning.
        
        Args:
            image: Input image (path or PIL Image)
            ocr_text: High-accuracy OCR text from Surya
            provider: Provider type ("DEWA" or "SEWA"), auto-detected if None
            
        Returns:
            Structured data matching the appropriate schema
        """
        process_start = time.time()
        
        logger.info("Starting process_with_reasoning", extra={
            'image_type': type(image).__name__,
            'ocr_text_length': len(ocr_text),
            'provider': provider
        })
        
        try:
            # Load image if path
            if isinstance(image, (str, Path)):
                image = Image.open(image)
                logger.debug(f"Loaded image from path: {image.size}")
            
            # Auto-detect provider if not specified
            if not provider and (resource_type == "utility" or resource_type is None):
                provider = self._detect_provider(ocr_text)
                logger.info(f"Auto-detected provider: {provider}")
            
            # Get appropriate schema based on resource type
            schema_class = self._get_schema_for_resource_type(resource_type, provider)
            logger.info(f"Using schema: {schema_class.__name__}", extra={
                'resource_type': resource_type,
                'provider': provider
            })
            
            # Step 1: Extract spatial reasoning
            logger.info("Extracting spatial understanding...")
            spatial_understanding = self._extract_spatial_reasoning(image, ocr_text)
            
            # Step 2: Extract structured data with spatial context
            logger.info("Extracting structured data...")
            structured_data = self._extract_structured_data(
                image, ocr_text, schema_class, spatial_understanding
            )
            
            # Step 3: Validate and fill missing fields with fallback
            logger.info("Validating extraction...")
            validated_data = self._validate_and_complete(
                structured_data, ocr_text, schema_class
            )
            
            return validated_data
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            # Return error instead of falling back
            return {"error": str(e), "provider_name": provider}
    
    def process_direct(self, 
                      image: Union[str, Path, Image.Image], 
                      ocr_text: str, 
                      provider: str = None,
                      resource_type: str = None) -> Dict[str, Any]:
        """
        Direct extraction without spatial reasoning (faster but less accurate).
        
        Args:
            image: Input image
            ocr_text: OCR text
            provider: Provider type
            
        Returns:
            Structured data
        """
        try:
            # Load image if path
            if isinstance(image, (str, Path)):
                image = Image.open(image)
            
            # Auto-detect provider if not specified and using utility resource type
            if not provider and (resource_type == "utility" or resource_type is None):
                provider = self._detect_provider(ocr_text)
            
            # Get appropriate schema based on resource type
            schema_class = self._get_schema_for_resource_type(resource_type, provider)
            logger.info(f"Using schema: {schema_class.__name__}", extra={
                'resource_type': resource_type,
                'provider': provider
            })
            
            # Build prompt for direct extraction
            prompt = self.prompt_builder.build_direct_extraction_prompt(
                ocr_text, schema_class
            )
            
            # Create message format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Process vision info
            images, videos = process_vision_info(messages)
            
            # Prepare inputs
            inputs = self.processor(
                text=[text],
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False
                )
            
            # Decode response - extract only generated part
            generated_ids_trimmed = outputs[0][inputs.input_ids.shape[1]:]
            response = self.processor.decode(generated_ids_trimmed, skip_special_tokens=True)
            
            # Parse JSON
            structured_data = self._parse_json_response(response)
            
            # Validate
            return self._validate_and_complete(structured_data, ocr_text, schema_class)
            
        except Exception as e:
            logger.error(f"Direct processing failed: {e}")
            # Return error instead of falling back
            return {"error": str(e), "provider_name": provider}
    
    def _extract_spatial_reasoning(self, image: Image.Image, ocr_text: str) -> Dict[str, Any]:
        """Extract spatial understanding from image and OCR text"""
        # Build reasoning prompt
        prompt = self.prompt_builder.build_reasoning_prompt(ocr_text)
        
        # Create message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process vision info
        images, videos = process_vision_info(messages)
        
        # Prepare inputs
        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate reasoning
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True
            )
        
        # Decode response - extract only generated part
        generated_ids_trimmed = outputs[0][inputs.input_ids.shape[1]:]
        reasoning_text = self.processor.decode(generated_ids_trimmed, skip_special_tokens=True)
        
        # Parse spatial understanding
        return self.spatial_reasoner.parse_reasoning(reasoning_text)
    
    def _extract_structured_data(self, 
                                image: Image.Image, 
                                ocr_text: str, 
                                schema_class: BaseModel,
                                spatial_understanding: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured data using spatial context"""
        # Build extraction prompt with spatial context
        prompt = self.prompt_builder.build_extraction_prompt(
            ocr_text, schema_class, spatial_understanding
        )
        
        # Create message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process vision info
        images, videos = process_vision_info(messages)
        
        # Prepare inputs
        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate structured output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False
            )
        
        # Decode response - extract only generated part
        generated_ids_trimmed = outputs[0][inputs.input_ids.shape[1]:]
        response = self.processor.decode(generated_ids_trimmed, skip_special_tokens=True)
        
        # Parse JSON
        return self._parse_json_response(response)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from model response"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            return json.loads(json_str)
            
        except Exception as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Response: {response}")
            return {}
    
    def _validate_and_complete(self, 
                             data: Dict[str, Any], 
                             ocr_text: str,
                             schema_class: BaseModel) -> Dict[str, Any]:
        """Validate extracted data and fill missing fields"""
        try:
            # Create instance to validate
            instance = schema_class(**data)
            validated_data = instance.model_dump()
            
            # Check for missing critical fields
            missing_fields = self._get_missing_critical_fields(validated_data, schema_class)
            
            if missing_fields:
                logger.warning(f"Missing critical fields: {missing_fields}")
                # Add provider name if missing
                if "provider_name" not in validated_data:
                    validated_data["provider_name"] = "DEWA" if schema_class == DEWABill else "SEWA"
            
            return validated_data
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            # Return original data
            return data
    
    def _detect_provider(self, text: str) -> str:
        """Detect provider from OCR text"""
        text_lower = text.lower()
        
        if "dubai electricity" in text_lower or "dewa" in text_lower:
            return "DEWA"
        elif "sharjah electricity" in text_lower or "sewa" in text_lower:
            return "SEWA"
        else:
            # Default to DEWA
            return "DEWA"
    
    def _get_missing_critical_fields(self, data: Dict[str, Any], schema_class: BaseModel) -> list:
        """Get list of missing critical fields"""
        critical_fields = {
            "account_number", "bill_date", "total_amount",
            "electricity_kwh", "water_m3"
        }
        
        missing = []
        for field in critical_fields:
            if field in schema_class.model_fields and not data.get(field):
                missing.append(field)
        
        return missing
    
    def _get_schema_for_resource_type(self, resource_type: str, provider: str = None) -> BaseModel:
        """Get the appropriate schema based on resource type and provider"""
        if resource_type == "energy":
            return EnergyBill
        elif resource_type == "water":
            return WaterBill
        elif resource_type == "waste":
            return WasteBill
        elif resource_type == "utility" or resource_type is None:
            # Fall back to provider-specific schemas
            return DEWABill if provider == "DEWA" else SEWABill
        else:
            # Default to energy if unknown
            return EnergyBill