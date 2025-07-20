"""Surya OCR wrapper for text extraction from utility bills"""

import os
import json
import subprocess
import tempfile
import shutil
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class SuryaExtractor:
    """Wrapper for Surya OCR functionality"""
    
    def __init__(self, temp_dir: Optional[Path] = None):
        """Initialize Surya extractor
        
        Args:
            temp_dir: Directory for temporary files
        """
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "surya_ocr_temp"
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        self.max_dimension = 2000  # Max width/height for preprocessing
        
    def extract_text(self, file_path: Path, timeout: int = 60) -> Tuple[str, Dict]:
        """Extract text from image/PDF using Surya OCR
        
        Args:
            file_path: Path to the input file
            timeout: Timeout in seconds for OCR process
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        start_time = datetime.now()
        output_dir = self.temp_dir / f"ocr_{start_time.timestamp()}"
        output_dir.mkdir(exist_ok=True)
        
        try:
            # Preprocess SEWA files or large images
            input_file = file_path
            if 'SEWA' in file_path.name.upper():
                logger.info(f"Preprocessing SEWA file: {file_path.name}")
                input_file = self._preprocess_image(file_path)
            
            # Run surya_ocr command
            cmd = ["surya_ocr", str(input_file), "--output_dir", str(output_dir)]
            
            logger.info(f"Running Surya OCR on {file_path.name}")
            
            try:
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=timeout,
                    check=False
                )
            except subprocess.TimeoutExpired:
                logger.error(f"OCR timed out after {timeout} seconds")
                return "", {"error": f"OCR timed out after {timeout} seconds"}
            
            if result.returncode != 0:
                logger.error(f"OCR failed: {result.stderr}")
                return "", {"error": f"OCR process failed: {result.stderr}"}
            
            # Find and parse results
            # Use the input file stem (could be preprocessed)
            input_stem = Path(input_file).stem.replace("preprocessed_", "")
            results_file = output_dir / input_stem / "results.json"
            
            if not results_file.exists():
                # Try alternative path
                alt_results = list(output_dir.rglob("results.json"))
                if alt_results:
                    results_file = alt_results[0]
                else:
                    logger.error("OCR results file not found")
                    return "", {"error": "OCR results file not found"}
            
            # Parse OCR results
            with open(results_file, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
            
            # Extract text and metadata
            text_parts = []
            total_confidence = 0.0
            confidence_count = 0
            bbox_data = []
            
            for filename, pages in ocr_data.items():
                if isinstance(pages, list):
                    for page_idx, page in enumerate(pages):
                        if 'text_lines' in page:
                            for line in page['text_lines']:
                                text = line.get('text', '')
                                if text:
                                    text_parts.append(text)
                                    
                                # Collect confidence scores
                                if 'confidence' in line:
                                    total_confidence += line['confidence']
                                    confidence_count += 1
                                    
                                # Collect bounding box data
                                if 'bbox' in line:
                                    bbox_data.append({
                                        'text': text,
                                        'bbox': line['bbox'],
                                        'page': page_idx,
                                        'confidence': line.get('confidence', 0)
                                    })
                                    
                elif isinstance(pages, dict) and 'text_lines' in pages:
                    # Single page format
                    for line in pages['text_lines']:
                        text = line.get('text', '')
                        if text:
                            text_parts.append(text)
                            
                        if 'confidence' in line:
                            total_confidence += line['confidence']
                            confidence_count += 1
                            
                        if 'bbox' in line:
                            bbox_data.append({
                                'text': text,
                                'bbox': line['bbox'],
                                'page': 0,
                                'confidence': line.get('confidence', 0)
                            })
            
            # Calculate average confidence
            avg_confidence = (total_confidence / confidence_count) if confidence_count > 0 else 0.0
            
            # Join text parts
            extracted_text = '\n'.join(text_parts)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Build metadata
            metadata = {
                'processing_time_seconds': processing_time,
                'average_confidence': round(avg_confidence, 4),
                'total_lines': len(text_parts),
                'total_characters': len(extracted_text),
                'bbox_data': bbox_data[:10],  # Include first 10 bboxes for reference
                'extraction_method': 'surya_ocr'
            }
            
            logger.info(
                f"OCR completed: {len(text_parts)} lines, "
                f"{len(extracted_text)} chars, "
                f"confidence: {avg_confidence:.2%}"
            )
            
            return extracted_text, metadata
            
        except Exception as e:
            logger.error(f"Error during OCR extraction: {e}")
            return "", {"error": str(e)}
            
        finally:
            # Cleanup
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)
    
    def extract_with_postprocessing(self, file_path: Path) -> Tuple[str, Dict]:
        """Extract text with post-processing applied
        
        Args:
            file_path: Path to the input file
            
        Returns:
            Tuple of (processed_text, metadata)
        """
        raw_text, metadata = self.extract_text(file_path)
        
        if not raw_text:
            return raw_text, metadata
            
        # Apply post-processing
        processed_text = self._postprocess_text(raw_text)
        
        metadata['postprocessing_applied'] = True
        metadata['raw_character_count'] = len(raw_text)
        metadata['processed_character_count'] = len(processed_text)
        
        return processed_text, metadata
    
    def _postprocess_text(self, text: str) -> str:
        """Apply post-processing to OCR text
        
        Args:
            text: Raw OCR text
            
        Returns:
            Processed text
        """
        # Common OCR corrections
        corrections = {
            # Common OCR errors
            'O': '0',  # Letter O to zero in numeric contexts
            'l': '1',  # Lowercase L to one in numeric contexts
            'S': '5',  # S to 5 in numeric contexts
            
            # Specific to utility bills
            'kWn': 'kWh',
            'KWH': 'kWh',
            'kwh': 'kWh',
            'm3': 'm³',
            'CO2e': 'CO2e',
            
            # Provider names
            'DEWA': 'DEWA',
            'SEWA': 'SEWA',
            'Dubai Electricity & Water Authority': 'Dubai Electricity and Water Authority',
            'Sharjah Electricity & Water Authority': 'Sharjah Electricity and Water Authority',
        }
        
        # Apply corrections
        processed = text
        for old, new in corrections.items():
            processed = processed.replace(old, new)
        
        # Fix spacing issues
        import re
        
        # Remove multiple spaces
        processed = re.sub(r'\s+', ' ', processed)
        
        # Fix spacing around numbers and units
        processed = re.sub(r'(\d)\s+(kWh|m³|AED|Dhs)', r'\1 \2', processed)
        
        # Fix date formats
        processed = re.sub(r'(\d{2})\s*/\s*(\d{2})\s*/\s*(\d{4})', r'\1/\2/\3', processed)
        
        # Remove trailing/leading whitespace from lines
        lines = processed.split('\n')
        processed_lines = [line.strip() for line in lines if line.strip()]
        processed = '\n'.join(processed_lines)
        
        return processed
    
    def validate_extraction(self, text: str, file_type: str) -> Dict[str, bool]:
        """Validate extracted text contains expected fields
        
        Args:
            text: Extracted text
            file_type: Type of bill (DEWA/SEWA)
            
        Returns:
            Dictionary of validation results
        """
        validations = {}
        
        # Common validations
        validations['has_account_number'] = bool(re.search(r'\b\d{10}\b', text))
        validations['has_date'] = bool(re.search(r'\d{2}/\d{2}/\d{4}', text))
        validations['has_electricity'] = bool(re.search(r'\d+\s*kWh', text, re.IGNORECASE))
        
        # Provider-specific validations
        if file_type.upper() == "DEWA":
            validations['has_provider_name'] = 'DEWA' in text or 'Dubai Electricity' in text
            validations['has_emissions'] = bool(re.search(r'\d+\s*kg\s*CO2', text, re.IGNORECASE))
        else:  # SEWA
            validations['has_provider_name'] = 'SEWA' in text or 'Sharjah Electricity' in text
            validations['has_water'] = bool(re.search(r'\d+\s*m³', text, re.IGNORECASE))
        
        validations['overall_valid'] = (
            validations['has_account_number'] and 
            validations['has_date'] and 
            validations['has_electricity'] and
            validations['has_provider_name']
        )
        
        return validations
    
    def _preprocess_image(self, image_path: Path) -> Path:
        """Preprocess image to reduce complexity for OCR
        
        Args:
            image_path: Path to original image
            
        Returns:
            Path to preprocessed image
        """
        try:
            # Open image
            img = Image.open(image_path)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Check if resizing is needed
            width, height = img.size
            if width > self.max_dimension or height > self.max_dimension:
                # Calculate new dimensions maintaining aspect ratio
                if width > height:
                    new_width = self.max_dimension
                    new_height = int(height * (self.max_dimension / width))
                else:
                    new_height = self.max_dimension
                    new_width = int(width * (self.max_dimension / height))
                
                logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Enhance contrast
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)
            
            # Save preprocessed image
            preprocessed_path = self.temp_dir / f"preprocessed_{image_path.name}"
            img.save(preprocessed_path, 'PNG', optimize=True, quality=95)
            
            return preprocessed_path
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image_path  # Return original if preprocessing fails
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)