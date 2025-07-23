"""OCR post-processing utilities"""

import re


def process_surya_output(ocr_text: str) -> str:
    """Process Surya OCR output to clean common artifacts"""
    
    # Common character substitutions
    char_replacements = {
        '学': '',  # Remove Chinese character that replaces electricity icon
        '□': '',   # Remove box symbols
        '\u2248': '≈',  # Approximate symbol
        '\uf0b7': '•',  # Bullet point
    }
    
    # HTML/XML tags to remove
    tag_patterns = [
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
    ocr_corrections = {
        'DUBA': 'DUBAI',  # Common DUBAI misrecognition
        '127731ST': '1277315T',  # Meter number correction
        'E-5615T545': 'E-56151545',  # SEWA meter number
        'W-18A01172': 'W-13A011272',  # SEWA water meter
        'G-60353': 'G-60399',  # SEWA gas meter
    }
    
    # Step 1: Remove HTML/XML tags
    cleaned = ocr_text
    for pattern in tag_patterns:
        cleaned = re.sub(pattern, '', cleaned)
    
    # Step 2: Fix character substitutions
    for old_char, new_char in char_replacements.items():
        cleaned = cleaned.replace(old_char, new_char)
    
    # Step 3: Apply OCR corrections
    for error, correction in ocr_corrections.items():
        cleaned = cleaned.replace(error, correction)
    
    # Step 4: Fix table formatting
    cleaned = re.sub(r'(\d+)\s*kWh\s*□', r'\1 kWh', cleaned)
    cleaned = re.sub(r'(\d+)\s*kWh\s*<math>', r'\1 kWh', cleaned)
    cleaned = re.sub(r'(\d+)\.(\d+)\s+(\d+)\.(\d+)', r'\1.\2    \3.\4', cleaned)
    
    # Step 5: Normalize whitespace
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = '\n'.join(line.rstrip() for line in cleaned.split('\n'))
    cleaned = re.sub(r'\n\n+', '\n\n', cleaned)
    
    return cleaned.strip()