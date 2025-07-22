"""Prompt building utilities for Qwen VL"""

from typing import Dict, Any, Type
from pydantic import BaseModel
import json


class PromptBuilder:
    """Build prompts for different stages of VLM processing"""
    
    def build_reasoning_prompt(self, ocr_text: str) -> str:
        """Build prompt for spatial reasoning extraction"""
        prompt = f"""Analyze this utility bill image alongside the OCR text to understand the document structure.

OCR Text (99.9% accurate):
{ocr_text[:2000]}...

Please identify:
1. Visual layout and sections (headers, tables, totals)
2. Spatial relationships (which values belong to which labels)
3. Table structures (rows, columns, cells)
4. Visual hierarchy (bold, size, positioning)
5. Any OCR errors you can correct from visual context

Focus on understanding how the visual layout maps to our extraction needs. Be specific about:
- Where account information is located
- How consumption data is organized (tables, lists, etc.)
- Where totals and summary information appear
- Any visual cues that help identify field types"""
        
        return prompt
    
    def build_extraction_prompt(self, 
                              ocr_text: str, 
                              schema_class: Type[BaseModel],
                              spatial_understanding: Dict[str, Any]) -> str:
        """Build prompt for structured data extraction"""
        
        # Get schema fields
        schema_fields = self._get_schema_fields(schema_class)
        
        # Build spatial context summary
        spatial_summary = self._summarize_spatial_understanding(spatial_understanding)
        
        prompt = f"""Based on your spatial understanding and the provided OCR text, extract data according to this schema.

Schema Fields Required:
{json.dumps(schema_fields, indent=2)}

OCR Text:
{ocr_text[:2000]}...

Spatial Understanding:
{spatial_summary}

Instructions:
1. Extract values for all required fields
2. Use visual cues to resolve ambiguities
3. For table data, maintain row/column relationships
4. Apply any OCR corrections you identified
5. Return data as valid JSON matching the schema

Output JSON:"""
        
        return prompt
    
    def build_direct_extraction_prompt(self, 
                                     ocr_text: str, 
                                     schema_class: Type[BaseModel]) -> str:
        """Build prompt for direct extraction without reasoning"""
        
        schema_fields = self._get_schema_fields(schema_class)
        
        prompt = f"""Extract structured data from this utility bill.

Required Fields:
{json.dumps(schema_fields, indent=2)}

OCR Text:
{ocr_text[:2000]}...

Instructions:
1. Extract values for all fields listed above
2. Use the image to verify OCR text accuracy
3. Return data as valid JSON
4. Include null for missing fields

Output JSON:"""
        
        return prompt
    
    def _get_schema_fields(self, schema_class: Type[BaseModel]) -> Dict[str, Any]:
        """Extract important fields from schema"""
        fields = {}
        
        # Key fields to extract
        key_fields = [
            "account_number", "bill_date", "total_amount",
            "electricity_kwh", "water_m3", "carbon_kg_co2e",
            "premise_id", "flat_number", "vat_number",
            "invoice_number", "due_date", "customer_name"
        ]
        
        schema = schema_class.model_json_schema()
        properties = schema.get("properties", {})
        
        for field in key_fields:
            if field in properties:
                field_info = properties[field]
                fields[field] = {
                    "type": field_info.get("type", "string"),
                    "description": field_info.get("description", ""),
                    "required": field in schema.get("required", [])
                }
        
        return fields
    
    def _summarize_spatial_understanding(self, understanding: Dict[str, Any]) -> str:
        """Create summary of spatial understanding"""
        summary_parts = []
        
        # Sections
        if understanding.get("sections"):
            section_summary = "Document sections: "
            section_types = [s["type"] for s in understanding["sections"]]
            summary_parts.append(section_summary + ", ".join(section_types))
        
        # Tables
        if understanding.get("tables"):
            for table in understanding["tables"]:
                if table.get("headers"):
                    summary_parts.append(f"Table with columns: {', '.join(table['headers'])}")
        
        # Hierarchies
        if understanding.get("hierarchies"):
            important_elements = [h["element"] for h in understanding["hierarchies"][:3]]
            if important_elements:
                summary_parts.append(f"Key elements: {', '.join(important_elements)}")
        
        # Corrections
        if understanding.get("corrections"):
            summary_parts.append(f"OCR corrections needed: {len(understanding['corrections'])}")
        
        return "\n".join(summary_parts) if summary_parts else "No specific spatial features identified"