"""Qwen3-0.6B processor for structured data extraction from OCR text"""

import json
import torch
from typing import Dict, Any, Optional, Type, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
import logging
from datetime import datetime
import re

from ..models import DEWABill, SEWABill, BaseUtilityBill

logger = logging.getLogger(__name__)


class QwenProcessor:
    """Processor for extracting structured data using Qwen3-0.6B model"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", device: Optional[str] = None):
        """Initialize Qwen processor
        
        Args:
            model_name: Hugging Face model name
            device: Device to run model on (cuda/cpu/auto)
        """
        self.model_name = model_name
        
        # Determine device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
    def load_model(self):
        """Load the Qwen model and tokenizer"""
        if self.is_loaded:
            return
            
        logger.info(f"Loading Qwen model: {self.model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model with appropriate dtype for device
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            self.model.eval()
            self.is_loaded = True
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def create_extraction_prompt(self, ocr_text: str, provider: str) -> str:
        """Create a prompt for structured data extraction
        
        Args:
            ocr_text: Raw OCR text from the bill
            provider: Provider name (DEWA/SEWA)
            
        Returns:
            Formatted prompt for the model
        """
        if provider.upper() == "DEWA":
            fields = """
- provider_name: "Dubai Electricity and Water Authority (DEWA)"
- account_number: 10-digit customer account number (NOT the invoice number)
- bill_date: in DD/MM/YYYY format
- electricity_consumption: value in kWh
- current_reading: meter reading
- previous_reading: meter reading
- total_co2_emissions: in kgCO2e (if available)
"""
        else:  # SEWA
            fields = """
- provider_name: "Sharjah Electricity and Water Authority (SEWA)"
- account_number: 10-digit customer account number (NOT the invoice number)
- bill_date: in DD/MM/YYYY format
- electricity_consumption: value in kWh
- water_consumption: value in m³
- current_electricity_reading: meter reading
- previous_electricity_reading: meter reading
"""

        prompt = f"""Extract the following information from this {provider} utility bill text and return it as valid JSON:

{fields}

IMPORTANT: The account_number is the customer's account number, NOT the invoice/bill number. Look for "Account Number" in the text.

Bill text:
{ocr_text}

Important:
1. Return ONLY valid JSON, no explanations
2. Use exact field names as shown
3. Convert dates to DD/MM/YYYY format
4. Extract numeric values only (no units in values)
5. If a field is not found, use empty string "" or 0 for numbers

JSON:"""
        
        return prompt
    
    def extract_with_retry(
        self, 
        ocr_text: str, 
        provider: str, 
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Extract structured data with retry logic
        
        Args:
            ocr_text: Raw OCR text
            provider: Provider name (DEWA/SEWA)
            max_retries: Maximum number of retry attempts
            
        Returns:
            Extracted data dictionary
        """
        if not self.is_loaded:
            self.load_model()
            
        for attempt in range(max_retries):
            try:
                # Create prompt
                prompt = self.create_extraction_prompt(ocr_text, provider)
                
                # Tokenize
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt",
                    max_length=2048,
                    truncation=True
                ).to(self.device)
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.1 if attempt == 0 else 0.3 + (attempt * 0.1),
                        do_sample=attempt > 0,  # Use sampling on retries
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                )
                
                # Extract JSON from response
                json_str = self._extract_json(response)
                if json_str:
                    data = json.loads(json_str)
                    
                    # Validate and clean data
                    cleaned_data = self._clean_extracted_data(data, provider)
                    return cleaned_data
                    
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    # On last attempt, try to extract what we can
                    return self._fallback_extraction(ocr_text, provider)
            except Exception as e:
                logger.error(f"Extraction error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return self._fallback_extraction(ocr_text, provider)
                    
        return {}
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from model response
        
        Args:
            text: Model response text
            
        Returns:
            JSON string or None
        """
        # Try to find JSON-like content
        # Look for content between { and }
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            return match.group(0)
            
        # If no match, try to clean and return the whole text
        text = text.strip()
        if text.startswith('{') and text.endswith('}'):
            return text
            
        return None
    
    def _clean_extracted_data(self, data: Dict[str, Any], provider: str) -> Dict[str, Any]:
        """Clean and validate extracted data
        
        Args:
            data: Raw extracted data
            provider: Provider name
            
        Returns:
            Cleaned data dictionary
        """
        cleaned = {}
        
        # Common fields
        cleaned['provider_name'] = data.get('provider_name', 
            "Dubai Electricity and Water Authority (DEWA)" if provider.upper() == "DEWA" 
            else "Sharjah Electricity and Water Authority (SEWA)")
        
        # Account number
        account = str(data.get('account_number', ''))
        cleaned['account_number'] = ''.join(filter(str.isdigit, account))
        
        # Bill date
        bill_date = data.get('bill_date', '')
        if bill_date and '/' in bill_date:
            cleaned['bill_date'] = bill_date
        else:
            # Try to parse other formats
            cleaned['bill_date'] = self._parse_date(bill_date)
        
        # Electricity consumption
        cleaned['electricity_consumption'] = self._parse_number(
            data.get('electricity_consumption', 0)
        )
        
        # Meter readings
        if provider.upper() == "DEWA":
            cleaned['current_reading'] = str(data.get('current_reading', ''))
            cleaned['previous_reading'] = str(data.get('previous_reading', ''))
            cleaned['total_co2_emissions'] = self._parse_number(
                data.get('total_co2_emissions', 0)
            )
        else:  # SEWA
            cleaned['water_consumption'] = self._parse_number(
                data.get('water_consumption', 0)
            )
            cleaned['current_electricity_reading'] = str(
                data.get('current_electricity_reading', '')
            )
            cleaned['previous_electricity_reading'] = str(
                data.get('previous_electricity_reading', '')
            )
            
        return cleaned
    
    def _parse_number(self, value: Any) -> float:
        """Parse a number from various formats
        
        Args:
            value: Value to parse
            
        Returns:
            Float value
        """
        if isinstance(value, (int, float)):
            return float(value)
            
        if isinstance(value, str):
            # Remove non-numeric characters except decimal point
            cleaned = re.sub(r'[^\d.]', '', value)
            try:
                return float(cleaned) if cleaned else 0.0
            except:
                return 0.0
                
        return 0.0
    
    def _parse_date(self, date_str: str) -> str:
        """Parse date to DD/MM/YYYY format
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            Date in DD/MM/YYYY format or empty string
        """
        if not date_str:
            return ""
            
        # Try various date formats
        formats = [
            '%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y',
            '%d.%m.%Y', '%Y.%m.%d', '%d %b %Y',
            '%B %d, %Y', '%d %B %Y'
        ]
        
        for fmt in formats:
            try:
                date_obj = datetime.strptime(date_str.strip(), fmt)
                return date_obj.strftime('%d/%m/%Y')
            except:
                continue
                
        return ""
    
    def _fallback_extraction(self, ocr_text: str, provider: str) -> Dict[str, Any]:
        """Fallback extraction using regex patterns
        
        Args:
            ocr_text: Raw OCR text
            provider: Provider name
            
        Returns:
            Extracted data dictionary
        """
        logger.info("Using fallback extraction method")
        
        data = {
            'provider_name': (
                "Dubai Electricity and Water Authority (DEWA)" if provider.upper() == "DEWA"
                else "Sharjah Electricity and Water Authority (SEWA)"
            )
        }
        
        # Account number - look for "Account Number" followed by 10 digits
        # Try multiple patterns to find the actual account number
        account_patterns = [
            r'Account\s*Number\s*\n?\s*(\d{10})',
            r'Account\s*No\.?\s*:?\s*(\d{10})',
            r'Customer\s*(?:Number|No\.?)\s*:?\s*(\d{10})',
            r'Account\s*#\s*:?\s*(\d{10})',
        ]
        
        account_found = False
        for pattern in account_patterns:
            account_match = re.search(pattern, ocr_text, re.IGNORECASE | re.MULTILINE)
            if account_match:
                data['account_number'] = account_match.group(1)
                account_found = True
                break
        
        # If no specific pattern found, look for 10-digit number NOT preceded by "Invoice"
        if not account_found:
            # Find all 10-digit numbers
            all_numbers = list(re.finditer(r'\b\d{10}\b', ocr_text))
            for match in all_numbers:
                # Check if this number is NOT part of an invoice number
                start = max(0, match.start() - 50)
                context = ocr_text[start:match.start()].lower()
                if 'invoice' not in context and 'bill' not in context:
                    data['account_number'] = match.group(0)
                    break
        
        # Date patterns
        date_patterns = [
            r'\d{2}/\d{2}/\d{4}',
            r'\d{2}-\d{2}-\d{4}',
            r'\d{2}\.\d{2}\.\d{4}'
        ]
        for pattern in date_patterns:
            date_match = re.search(pattern, ocr_text)
            if date_match:
                data['bill_date'] = date_match.group(0).replace('-', '/').replace('.', '/')
                break
        
        # Electricity consumption (look for kWh)
        kwh_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:kWh|KWH)', ocr_text, re.IGNORECASE)
        if kwh_match:
            data['electricity_consumption'] = float(kwh_match.group(1))
        
        # Water consumption for SEWA (look for m³)
        if provider.upper() == "SEWA":
            water_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:m³|m3|cubic)', ocr_text, re.IGNORECASE)
            if water_match:
                data['water_consumption'] = float(water_match.group(1))
        
        return data
    
    def process_to_pydantic(
        self, 
        ocr_text: str, 
        provider: str,
        source_document: str
    ) -> Union[DEWABill, SEWABill]:
        """Process OCR text and return Pydantic model
        
        Args:
            ocr_text: Raw OCR text
            provider: Provider name (DEWA/SEWA)
            source_document: Source document filename
            
        Returns:
            Pydantic model instance
        """
        start_time = datetime.now()
        
        # Extract data
        extracted_data = self.extract_with_retry(ocr_text, provider)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Build Pydantic model data
        if provider.upper() == "DEWA":
            model_data = self._build_dewa_model_data(
                extracted_data, source_document, processing_time
            )
            return DEWABill(**model_data)
        else:
            model_data = self._build_sewa_model_data(
                extracted_data, source_document, processing_time
            )
            return SEWABill(**model_data)
    
    def _build_dewa_model_data(
        self, 
        extracted: Dict[str, Any], 
        source_doc: str,
        processing_time: float
    ) -> Dict[str, Any]:
        """Build DEWA model data from extracted information"""
        return {
            "document_type": "utility_bill",
            "extracted_data": {
                "bill_info": {
                    "provider_name": extracted.get('provider_name', ''),
                    "account_number": extracted.get('account_number', ''),
                    "bill_date": extracted.get('bill_date', ''),
                    "billing_period": {
                        "start_date": "",
                        "end_date": "",
                        "periodicity": "Monthly"
                    }
                },
                "consumption_data": {
                    "electricity": {
                        "value": extracted.get('electricity_consumption', 0.0),
                        "unit": "kWh",
                        "data_quality": "measured",
                        "meter_reading": {
                            "current": extracted.get('current_reading', ''),
                            "previous": extracted.get('previous_reading', ''),
                            "reading_type": "actual"
                        }
                    },
                    "renewable_percentage": 0.0,
                    "peak_demand": {
                        "value": 0.0,
                        "unit": "kW"
                    }
                },
                "emission_factor_reference": {
                    "region": "United Arab Emirates",
                    "grid_mix": "UAE_GRID_2024",
                    "year": "2024"
                },
                "emissions_data": {
                    "scope2": {
                        "totalCO2e": {
                            "value": extracted.get('total_co2_emissions', 0.0),
                            "unit": "kgCO2e"
                        },
                        "breakdown": {
                            "electricity": extracted.get('total_co2_emissions', 0.0)
                        }
                    }
                }
            },
            "validation": {
                "confidence": 0.0,  # Will be calculated by model
                "extraction_method": "qwen3_0.6b_surya",
                "manual_verification_required": False
            },
            "metadata": {
                "source_document": source_doc,
                "page_numbers": [1],
                "extraction_timestamp": datetime.utcnow().isoformat() + "Z",
                "processing_time_seconds": processing_time
            }
        }
    
    def _build_sewa_model_data(
        self, 
        extracted: Dict[str, Any], 
        source_doc: str,
        processing_time: float
    ) -> Dict[str, Any]:
        """Build SEWA model data from extracted information"""
        return {
            "document_type": "utility_bill",
            "extracted_data": {
                "bill_info": {
                    "provider_name": extracted.get('provider_name', ''),
                    "account_number": extracted.get('account_number', ''),
                    "bill_date": extracted.get('bill_date', ''),
                    "billing_period": {
                        "start_date": "",
                        "end_date": "",
                        "periodicity": "Monthly"
                    }
                },
                "consumption_data": {
                    "electricity": {
                        "value": extracted.get('electricity_consumption', 0.0),
                        "unit": "kWh",
                        "data_quality": "measured",
                        "meter_reading": {
                            "current": extracted.get('current_electricity_reading', ''),
                            "previous": extracted.get('previous_electricity_reading', ''),
                            "reading_type": "actual"
                        }
                    },
                    "water": {
                        "value": extracted.get('water_consumption', 0.0),
                        "unit": "m³",
                        "data_quality": "measured"
                    },
                    "renewable_percentage": 0.0,
                    "peak_demand": {
                        "value": 0.0,
                        "unit": "kW"
                    }
                },
                "emission_factor_reference": {
                    "region": "United Arab Emirates",
                    "grid_mix": "UAE_GRID_2024",
                    "year": "2024"
                }
            },
            "validation": {
                "confidence": 0.0,  # Will be calculated by model
                "extraction_method": "qwen3_0.6b_surya",
                "manual_verification_required": False
            },
            "metadata": {
                "source_document": source_doc,
                "page_numbers": [1],
                "extraction_timestamp": datetime.utcnow().isoformat() + "Z",
                "processing_time_seconds": processing_time
            }
        }
    
    def unload_model(self):
        """Unload model from memory"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        self.is_loaded = False
        
        # Clear GPU cache if using CUDA
        if self.device == "cuda":
            torch.cuda.empty_cache()