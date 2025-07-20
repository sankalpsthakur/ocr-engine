#!/usr/bin/env python3
"""
Simplified Qwen Integration Module for Structured Data Extraction
Uses text-based processing to extract structured information from OCR results
"""

import time
import json
import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

@dataclass
class ExtractedField:
    """Represents an extracted field from OCR text"""
    name: str
    value: str
    confidence: float
    pattern_used: str

class QwenExtractor:
    """Simplified structured data extractor for OCR text"""
    
    def __init__(self, model_name: str = "qwen-text-fallback"):
        """Initialize the text-based extractor"""
        self.model_name = model_name
        self.device = "cpu"
        self.extraction_patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> Dict[str, Dict[str, str]]:
        """Initialize extraction patterns for different document types"""
        return {
            "bill": {
                "customer_name": r"(?:Customer|Name|Account Holder)[:\s]+([A-Za-z\s]{3,30})",
                "account_number": r"(?:Account|Acc|A/C|Customer)(?:\s+No|Number|ID)?[:\s]*(\d{6,15})",
                "bill_date": r"(?:Bill Date|Date|Invoice Date)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                "due_date": r"(?:Due Date|Payment Due)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                "total_amount": r"(?:Total|Amount Due|Total Amount)[:\s]*(\d+[.,]\d{2})\s*(AED|USD|SAR|EUR|\$)",
                "previous_reading": r"(?:Previous|Last)\s+(?:Reading|Meter)[:\s]*(\d+(?:[.,]\d+)?)",
                "current_reading": r"(?:Current|Present)\s+(?:Reading|Meter)[:\s]*(\d+(?:[.,]\d+)?)",
                "units_consumed": r"(?:Units|Consumption|Used|Consumed)[:\s]*(\d+(?:[.,]\d+)?)",
                "service_address": r"(?:Service Address|Address)[:\s]*([A-Za-z0-9\s,.-]{10,100})"
            },
            "invoice": {
                "invoice_number": r"(?:Invoice|Invoice No|Invoice Number)[:\s#]*(\w{3,20})",
                "invoice_date": r"(?:Invoice Date|Date)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                "vendor_name": r"(?:From|Vendor|Company)[:\s]*([A-Za-z\s&]{3,50})",
                "customer_name": r"(?:To|Bill To|Customer)[:\s]*([A-Za-z\s]{3,30})",
                "subtotal": r"(?:Subtotal|Sub Total)[:\s]*(\d+[.,]\d{2})",
                "tax_amount": r"(?:Tax|VAT|GST)[:\s]*(\d+[.,]\d{2})",
                "total_amount": r"(?:Total|Grand Total|Amount Due)[:\s]*(\d+[.,]\d{2})",
                "payment_terms": r"(?:Payment Terms|Terms)[:\s]*([A-Za-z0-9\s]{5,50})"
            },
            "receipt": {
                "store_name": r"^([A-Za-z\s&]{3,50})",
                "date": r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                "time": r"(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)",
                "total": r"(?:Total|Amount)[:\s]*(\d+[.,]\d{2})",
                "payment_method": r"(?:Payment|Paid by|Method)[:\s]*([A-Za-z\s]{3,20})",
                "receipt_number": r"(?:Receipt|Ref|Reference)[:\s#]*(\w{3,20})"
            },
            "general": {
                "phone": r"(\+?\d{1,3}[-.\s]?\d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{2,4})",
                "email": r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
                "amount": r"(\d+[.,]\d{2})\s*(AED|USD|SAR|EUR|GBP|\$|€|£)",
                "date": r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                "website": r"(https?://[^\s]+|www\.[^\s]+)",
                "postal_code": r"(\d{5}(?:-\d{4})?)"
            }
        }
    
    def load_model(self) -> float:
        """Simulate model loading for consistency with pipeline"""
        print(f"🔧 Loading Qwen text extractor: {self.model_name}")
        start_time = time.time()
        
        # Simulate loading time
        time.sleep(0.1)
        
        load_time = time.time() - start_time
        print(f"✅ Text extractor ready in {load_time:.3f} seconds")
        return load_time
    
    def extract_structured_data(self, ocr_text: str, image_path: str = None, 
                              extraction_type: str = "bill") -> Tuple[Dict[str, Any], float]:
        """
        Extract structured data from OCR text using pattern matching
        
        Args:
            ocr_text: Raw text from OCR
            image_path: Optional path to source image (ignored in text-only mode)
            extraction_type: Type of extraction (bill, invoice, receipt, general)
            
        Returns:
            Tuple of (extracted_data_dict, processing_time)
        """
        start_time = time.time()
        
        print(f"  🧠 Running text-based structured extraction ({extraction_type})...")
        
        # Get patterns for the extraction type
        patterns = self.extraction_patterns.get(extraction_type, self.extraction_patterns["general"])
        
        extracted_data = {
            "extraction_method": "text_pattern_matching",
            "extraction_type": extraction_type,
            "source_text_length": len(ocr_text),
            "fields": {}
        }
        
        # Extract fields using patterns
        total_confidence = 0.0
        field_count = 0
        
        for field_name, pattern in patterns.items():
            matches = re.findall(pattern, ocr_text, re.IGNORECASE | re.MULTILINE)
            
            if matches:
                # Take the first match and calculate confidence based on pattern specificity
                value = matches[0] if isinstance(matches[0], str) else matches[0][0]
                confidence = min(0.95, 0.6 + (len(pattern) / 200))  # Simple confidence heuristic
                
                extracted_data["fields"][field_name] = {
                    "value": value.strip(),
                    "confidence": round(confidence, 2),
                    "pattern_used": field_name,
                    "all_matches": matches[:3]  # Keep up to 3 matches
                }
                
                total_confidence += confidence
                field_count += 1
        
        # Add summary statistics
        extracted_data["summary"] = {
            "fields_extracted": field_count,
            "average_confidence": round(total_confidence / field_count if field_count > 0 else 0, 2),
            "text_coverage": round((sum(len(str(f["value"])) for f in extracted_data["fields"].values()) / len(ocr_text)) * 100, 1)
        }
        
        # Extract additional context
        extracted_data["context"] = self._extract_context(ocr_text)
        
        processing_time = time.time() - start_time
        
        print(f"    ✅ Text extraction completed in {processing_time:.3f}s, extracted {field_count} fields")
        
        return extracted_data, processing_time
    
    def _extract_context(self, text: str) -> Dict[str, Any]:
        """Extract additional context from the text"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Find potential headers (first few lines with caps or title case)
        headers = []
        for line in lines[:5]:
            if len(line) > 5 and (line.isupper() or line.istitle()):
                headers.append(line)
        
        # Extract all numeric values
        numbers = re.findall(r'\d+(?:[.,]\d+)*', text)
        
        # Extract all dates
        dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
        
        # Word count and statistics
        words = text.split()
        
        return {
            "potential_headers": headers[:3],
            "total_lines": len(lines),
            "total_words": len(words),
            "numeric_values": numbers[:10],  # Top 10 numbers
            "dates_found": list(set(dates)),  # Unique dates
            "average_line_length": round(sum(len(line) for line in lines) / len(lines) if lines else 0, 1)
        }
    
    def extract_bill_fields(self, ocr_text: str, image_path: str = None) -> Tuple[Dict[str, Any], float]:
        """Specialized extraction for utility bills"""
        return self.extract_structured_data(ocr_text, image_path, "bill")
    
    def extract_invoice_fields(self, ocr_text: str, image_path: str = None) -> Tuple[Dict[str, Any], float]:
        """Specialized extraction for invoices"""
        return self.extract_structured_data(ocr_text, image_path, "invoice")
    
    def batch_extract(self, ocr_results: List[Dict[str, Any]], 
                     extraction_type: str = "bill") -> List[Dict[str, Any]]:
        """
        Batch process multiple OCR results
        
        Args:
            ocr_results: List of dicts with 'text' and optional 'image_path' keys
            extraction_type: Type of extraction to perform
            
        Returns:
            List of extraction results with timing info
        """
        results = []
        total_start = time.time()
        
        for i, ocr_result in enumerate(ocr_results):
            print(f"🔄 Processing item {i+1}/{len(ocr_results)}")
            
            text = ocr_result.get('text', '')
            image_path = ocr_result.get('image_path')
            
            extracted_data, processing_time = self.extract_structured_data(
                text, image_path, extraction_type
            )
            
            results.append({
                'index': i,
                'extracted_data': extracted_data,
                'processing_time': processing_time,
                'source_text_length': len(text)
            })
        
        total_time = time.time() - total_start
        print(f"✅ Batch processing completed in {total_time:.2f} seconds")
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the extractor"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "extraction_method": "regex_pattern_matching",
            "supported_types": list(self.extraction_patterns.keys()),
            "total_patterns": sum(len(patterns) for patterns in self.extraction_patterns.values()),
            "note": "Text-based extraction using optimized regex patterns"
        }

def test_qwen_integration():
    """Test function for simplified Qwen integration"""
    # Sample OCR text from a utility bill
    sample_text = """
    DEWA - Dubai Electricity & Water Authority
    Customer Name: John Smith
    Account Number: 123456789
    Bill Date: 15/01/2024
    Due Date: 30/01/2024
    
    Previous Reading: 1250 kWh
    Current Reading: 1380 kWh
    Units Consumed: 130 kWh
    
    Electricity Charges: 89.50 AED
    Water Charges: 25.30 AED
    Municipality Fee: 15.00 AED
    Total Amount: 129.80 AED
    
    Service Address: 123 Main Street, Dubai, UAE
    Phone: +971-4-601-9999
    Website: www.dewa.gov.ae
    """
    
    print("🧪 Testing Simplified Qwen Integration")
    
    extractor = QwenExtractor()
    load_time = extractor.load_model()
    
    extracted_data, processing_time = extractor.extract_bill_fields(sample_text)
    
    print(f"\n📊 Extraction Results:")
    print(f"Load Time: {load_time:.3f}s")
    print(f"Processing Time: {processing_time:.3f}s")
    print(f"Extracted Data: {json.dumps(extracted_data, indent=2)}")
    
    model_info = extractor.get_model_info()
    print(f"\n🔧 Model Info: {json.dumps(model_info, indent=2)}")

if __name__ == "__main__":
    test_qwen_integration()