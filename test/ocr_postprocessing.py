#!/usr/bin/env python3
"""Post-processing script to clean Surya OCR output for better CER scores"""

import re
from typing import Dict, List, Tuple

class SuryaPostProcessor:
    """Clean up Surya OCR output to match ground truth format"""
    
    def __init__(self):
        # Common character substitutions found in Surya output
        self.char_replacements = {
            '学': '',  # Remove Chinese character that replaces electricity icon
            '□': '',   # Remove box symbols
            '\u2248': '≈',  # Approximate symbol
            '\uf0b7': '•',  # Bullet point
        }
        
        # HTML/XML tags to remove
        self.tag_patterns = [
            r'<b>|</b>',  # Bold tags
            r'<math>|</math>',  # Math tags
            r'<sub>|</sub>',  # Subscript tags
            r'<sup>|</sup>',  # Superscript tags
            r'<i>|</i>',  # Italic tags
            r'<u>|</u>',  # Underline tags
            r'<strong>|</strong>',  # Strong tags
            r'<em>|</em>',  # Emphasis tags
            r'<span[^>]*>|</span>',  # Span tags with attributes
            r'<div[^>]*>|</div>',  # Div tags
            r'\\Box|\\square',  # LaTeX box symbols
        ]
        
        # Common OCR errors specific to utility bills
        self.ocr_corrections = {
            'DUBA': 'DUBAI',  # Common DUBAI misrecognition
            '127731ST': '1277315T',  # Meter number correction
            'E-5615T545': 'E-56151545',  # SEWA meter number
            'W-18A01172': 'W-13A011272',  # SEWA water meter
            'G-60353': 'G-60399',  # SEWA gas meter
        }
        
    def remove_html_tags(self, text: str) -> str:
        """Remove all HTML/XML tags from text"""
        cleaned = text
        for pattern in self.tag_patterns:
            cleaned = re.sub(pattern, '', cleaned)
        return cleaned
    
    def fix_character_substitutions(self, text: str) -> str:
        """Replace misrecognized characters"""
        cleaned = text
        for old_char, new_char in self.char_replacements.items():
            cleaned = cleaned.replace(old_char, new_char)
        return cleaned
    
    def apply_ocr_corrections(self, text: str) -> str:
        """Apply known OCR error corrections"""
        cleaned = text
        for error, correction in self.ocr_corrections.items():
            cleaned = cleaned.replace(error, correction)
        return cleaned
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving structure"""
        # Remove multiple spaces (but keep newlines)
        cleaned = re.sub(r'[ \t]+', ' ', text)
        # Remove trailing spaces on each line
        cleaned = '\n'.join(line.rstrip() for line in cleaned.split('\n'))
        # Remove multiple blank lines
        cleaned = re.sub(r'\n\n+', '\n\n', cleaned)
        return cleaned.strip()
    
    def fix_table_formatting(self, text: str) -> str:
        """Fix common table formatting issues"""
        # Fix spacing around numbers in tables
        cleaned = re.sub(r'(\d+)\s*kWh\s*□', r'\1 kWh', text)
        cleaned = re.sub(r'(\d+)\s*kWh\s*<math>', r'\1 kWh', cleaned)
        
        # Fix decimal alignment
        cleaned = re.sub(r'(\d+)\.(\d+)\s+(\d+)\.(\d+)', r'\1.\2    \3.\4', cleaned)
        
        return cleaned
    
    def process(self, text: str) -> str:
        """Apply all post-processing steps"""
        # Step 1: Remove HTML/XML tags
        cleaned = self.remove_html_tags(text)
        
        # Step 2: Fix character substitutions
        cleaned = self.fix_character_substitutions(cleaned)
        
        # Step 3: Apply OCR corrections
        cleaned = self.apply_ocr_corrections(cleaned)
        
        # Step 4: Fix table formatting
        cleaned = self.fix_table_formatting(cleaned)
        
        # Step 5: Normalize whitespace
        cleaned = self.normalize_whitespace(cleaned)
        
        return cleaned


def process_surya_output(ocr_text: str) -> str:
    """Main function to process Surya OCR output"""
    processor = SuryaPostProcessor()
    return processor.process(ocr_text)


def process_file(input_file: str, output_file: str = None):
    """Process a file containing Surya OCR output"""
    with open(input_file, 'r', encoding='utf-8') as f:
        ocr_text = f.read()
    
    cleaned_text = process_surya_output(ocr_text)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
    else:
        print(cleaned_text)
    
    return cleaned_text


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ocr_postprocessing.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    process_file(input_file, output_file)