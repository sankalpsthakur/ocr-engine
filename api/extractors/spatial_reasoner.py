"""Spatial reasoning extraction and parsing"""

import re
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class SpatialReasoner:
    """Extract and parse spatial relationships from VLM reasoning"""
    
    def parse_reasoning(self, reasoning_text: str) -> Dict[str, Any]:
        """
        Parse spatial reasoning output into structured format.
        
        Args:
            reasoning_text: Raw reasoning text from VLM
            
        Returns:
            Structured spatial understanding
        """
        understanding = {
            "sections": self._extract_sections(reasoning_text),
            "tables": self._extract_tables(reasoning_text),
            "hierarchies": self._extract_hierarchies(reasoning_text),
            "alignments": self._extract_alignments(reasoning_text),
            "corrections": self._extract_corrections(reasoning_text)
        }
        
        return understanding
    
    def _extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """Extract document sections and their locations"""
        sections = []
        
        # Pattern for section identification
        section_patterns = [
            r"(?i)(header|top).*?contain.*?(account|bill|customer)",
            r"(?i)(middle|center).*?contain.*?(consumption|usage|meter)",
            r"(?i)(bottom|footer).*?contain.*?(total|amount|payment)",
            r"(?i)section.*?titled.*?['\"]([^'\"]+)['\"]",
        ]
        
        for pattern in section_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                sections.append({
                    "type": self._classify_section(match.group()),
                    "content": match.group(),
                    "position": self._extract_position(match.group())
                })
        
        return sections
    
    def _extract_tables(self, text: str) -> List[Dict[str, Any]]:
        """Extract table structures and relationships"""
        tables = []
        
        # Pattern for table identification
        table_patterns = [
            r"(?i)table.*?(?:with|containing|showing).*?(\d+).*?rows?.*?(\d+).*?columns?",
            r"(?i)(\d+)\s*x\s*(\d+).*?table",
            r"(?i)tabular.*?data.*?(?:with|containing).*?([^.]+)",
        ]
        
        for pattern in table_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                table_info = {
                    "description": match.group(),
                    "structure": self._parse_table_structure(match.group())
                }
                
                # Extract column headers if mentioned
                header_match = re.search(
                    r"(?i)columns?.*?(?:are|include|:)\s*([^.]+)", 
                    text[match.start():match.start()+200]
                )
                if header_match:
                    table_info["headers"] = self._parse_headers(header_match.group(1))
                
                tables.append(table_info)
        
        return tables
    
    def _extract_hierarchies(self, text: str) -> List[Dict[str, str]]:
        """Extract visual hierarchies (bold, size, positioning)"""
        hierarchies = []
        
        hierarchy_patterns = [
            r"(?i)(bold|large|prominent).*?(?:text|font).*?(?:for|showing|indicating)\s*([^.,]+)",
            r"(?i)([^.,]+).*?(?:is|appears).*?(bold|large|highlighted)",
            r"(?i)(total|amount|sum).*?(?:is|appears).*?(emphasized|prominent)",
        ]
        
        for pattern in hierarchy_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                hierarchies.append({
                    "element": match.group(2) if match.lastindex >= 2 else match.group(1),
                    "style": match.group(1) if match.lastindex >= 2 else "prominent",
                    "purpose": self._infer_purpose(match.group())
                })
        
        return hierarchies
    
    def _extract_alignments(self, text: str) -> List[Dict[str, Any]]:
        """Extract label-value alignments and relationships"""
        alignments = []
        
        alignment_patterns = [
            r"(?i)([^:]+):\s*([^,.\n]+)",
            r"(?i)(?:label|field).*?['\"]([^'\"]+)['\"].*?(?:value|shows|is)\s*([^.,\n]+)",
            r"(?i)([^.,]+).*?(?:aligned|next to|beside|follows).*?([^.,]+)",
        ]
        
        for pattern in alignment_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if match.lastindex >= 2:
                    alignments.append({
                        "label": match.group(1).strip(),
                        "value": match.group(2).strip(),
                        "relationship": "adjacent"
                    })
        
        return alignments
    
    def _extract_corrections(self, text: str) -> List[Dict[str, str]]:
        """Extract OCR corrections identified by visual context"""
        corrections = []
        
        correction_patterns = [
            r"(?i)(?:should be|actually|correctly)\s*['\"]([^'\"]+)['\"].*?(?:not|instead of)\s*['\"]([^'\"]+)['\"]",
            r"(?i)(?:misread|incorrect).*?['\"]([^'\"]+)['\"].*?(?:should|is)\s*['\"]([^'\"]+)['\"]",
            r"(?i)visual.*?shows?\s*['\"]([^'\"]+)['\"].*?OCR.*?['\"]([^'\"]+)['\"]",
        ]
        
        for pattern in correction_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                corrections.append({
                    "correct": match.group(1),
                    "incorrect": match.group(2),
                    "confidence": "high"
                })
        
        return corrections
    
    def _classify_section(self, text: str) -> str:
        """Classify section type based on content"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["header", "top", "account", "customer"]):
            return "header"
        elif any(word in text_lower for word in ["consumption", "usage", "meter", "reading"]):
            return "consumption"
        elif any(word in text_lower for word in ["total", "amount", "payment", "bottom"]):
            return "summary"
        else:
            return "other"
    
    def _extract_position(self, text: str) -> str:
        """Extract position information"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["top", "header", "upper"]):
            return "top"
        elif any(word in text_lower for word in ["middle", "center"]):
            return "middle"
        elif any(word in text_lower for word in ["bottom", "footer", "lower"]):
            return "bottom"
        else:
            return "unknown"
    
    def _parse_table_structure(self, text: str) -> Dict[str, int]:
        """Parse table dimensions"""
        # Try to extract rows and columns
        numbers = re.findall(r'\d+', text)
        
        if len(numbers) >= 2:
            return {"rows": int(numbers[0]), "columns": int(numbers[1])}
        else:
            return {"rows": 0, "columns": 0}
    
    def _parse_headers(self, text: str) -> List[str]:
        """Parse column headers from text"""
        # Split by common delimiters
        headers = re.split(r'[,;]|\sand\s', text)
        return [h.strip() for h in headers if h.strip()]
    
    def _infer_purpose(self, text: str) -> str:
        """Infer the purpose of a visual element"""
        text_lower = text.lower()
        
        if "total" in text_lower:
            return "summary"
        elif "account" in text_lower or "customer" in text_lower:
            return "identification"
        elif "consumption" in text_lower or "usage" in text_lower:
            return "measurement"
        else:
            return "information"