"""
Orientation Handler Utility
Handles automatic orientation detection and correction for images and PDFs
Supports multiple detection methods with fallback strategies
"""

import logging
from typing import Union, Optional, Tuple
from PIL import Image, ImageOps
import io
import numpy as np
import httpx
import asyncio

logger = logging.getLogger(__name__)

class OrientationHandler:
    """Handles orientation detection and correction for images"""
    
    # Surya service URL (will be set by gateway)
    SURYA_SERVICE_URL = "http://localhost:8001"
    
    @staticmethod
    def fix_image_orientation(image: Union[Image.Image, bytes, str]) -> Image.Image:
        """
        Fix image orientation using EXIF data if available
        
        Args:
            image: PIL Image, bytes, or path to image
            
        Returns:
            Correctly oriented PIL Image
        """
        if isinstance(image, (bytes, str)):
            image = Image.open(io.BytesIO(image) if isinstance(image, bytes) else image)
        
        try:
            # Apply EXIF orientation if available
            image = ImageOps.exif_transpose(image)
            logger.info("Applied EXIF orientation correction")
        except Exception as e:
            logger.debug(f"No EXIF orientation data or error: {type(e).__name__}: {e}")
        
        return image
    
    @staticmethod
    async def detect_orientation_by_surya(
        image: Image.Image,
        surya_url: str = None
    ) -> Tuple[int, float]:
        """
        Detect orientation by comparing Surya OCR confidence at different rotations
        
        Args:
            image: PIL Image
            surya_url: URL of Surya service (uses class default if not provided)
            
        Returns:
            Tuple of (best_angle, best_confidence)
        """
        if surya_url is None:
            surya_url = OrientationHandler.SURYA_SERVICE_URL
            
        best_angle = 0
        best_confidence = 0
        best_text_length = 0
        
        angles_to_check = [0, 90, 180, 270]
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for angle in angles_to_check:
                try:
                    # Rotate image if needed
                    if angle == 0:
                        rotated = image
                    else:
                        rotated = image.rotate(360 - angle, expand=True)
                    
                    # Convert to bytes for API call
                    img_buffer = io.BytesIO()
                    rotated.save(img_buffer, format='PNG')
                    img_bytes = img_buffer.getvalue()
                    
                    # Call Surya OCR service
                    files = {'file': ('test.png', img_bytes, 'image/png')}
                    response = await client.post(f"{surya_url}/ocr", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Extract confidence and text length
                        confidence = result.get('confidence', 0) or 0
                        text = result.get('text', '') or ''
                        text_length = len(text.strip())
                        
                        logger.debug(f"Angle {angle}°: confidence={confidence:.2f}, text_length={text_length}")
                        
                        # Prefer higher confidence and longer text
                        if confidence > best_confidence or (confidence == best_confidence and text_length > best_text_length):
                            best_confidence = confidence
                            best_angle = angle
                            best_text_length = text_length
                    else:
                        logger.warning(f"Surya OCR failed for angle {angle}°: HTTP {response.status_code}")
                        
                except Exception as e:
                    logger.warning(f"Error checking angle {angle}°: {type(e).__name__}: {str(e)}", exc_info=True)
        
        logger.info(f"Best orientation via Surya: {best_angle}° (confidence: {best_confidence:.2f})")
        return best_angle, best_confidence
    
    
    @staticmethod
    async def detect_orientation_for_utility_bill(
        image: Image.Image, 
        surya_url: str = None
    ) -> int:
        """
        Detect orientation specifically for utility bills by looking for header patterns
        
        Args:
            image: PIL Image
            surya_url: URL of Surya service
            
        Returns:
            Best rotation angle (0, 90, 180, or 270)
        """
        header_patterns = [
            'DEWA', 'SEWA', 'Dubai Electricity', 'Sharjah Electricity',
            'Account Number', 'Account No', 'Bill Date', 'Invoice Date',
            'Customer Name', 'Service Address', 'Utility Bill'
        ]
        
        if surya_url is None:
            surya_url = OrientationHandler.SURYA_SERVICE_URL
            
        best_angle = 0
        max_header_matches = 0
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for angle in [0, 90, 180, 270]:
                try:
                    if angle == 0:
                        rotated = image
                    else:
                        rotated = image.rotate(360 - angle, expand=True)
                    
                    # Crop only the top 25% of the image
                    width, height = rotated.size
                    top_region = rotated.crop((0, 0, width, height // 4))
                    
                    # Convert to bytes for API call
                    img_buffer = io.BytesIO()
                    top_region.save(img_buffer, format='PNG')
                    img_bytes = img_buffer.getvalue()
                    
                    # Call Surya OCR service
                    files = {'file': ('test.png', img_bytes, 'image/png')}
                    response = await client.post(f"{surya_url}/ocr", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        text = result.get('text', '').upper()
                        
                        # Count header pattern matches
                        matches = sum(1 for pattern in header_patterns if pattern.upper() in text)
                        
                        if matches > max_header_matches:
                            max_header_matches = matches
                            best_angle = angle
                            
                except Exception as e:
                    logger.debug(f"OCR failed for angle {angle}°: {type(e).__name__}: {e}")
        
        logger.info(f"Utility bill orientation: {best_angle}° (header matches: {max_header_matches})")
        return best_angle
    
    @staticmethod
    async def auto_orient(
        image: Union[Image.Image, bytes, str],
        detection_method: str = 'auto',
        is_utility_bill: bool = False,
        surya_url: str = None
    ) -> Image.Image:
        """
        Automatically orient image using specified or best available method
        
        Args:
            image: PIL Image, bytes, or path to image
            detection_method: 'exif', 'surya', 'utility', or 'auto'
            is_utility_bill: Whether this is a utility bill (uses specialized detection)
            surya_url: URL of Surya service
            
        Returns:
            Correctly oriented PIL Image
        """
        # Convert to PIL Image if needed
        if isinstance(image, (bytes, str)):
            image = Image.open(io.BytesIO(image) if isinstance(image, bytes) else image)
        
        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Step 1: Always try EXIF first (it's free and instant)
        original_image = image.copy()
        image = OrientationHandler.fix_image_orientation(image)
        
        # If EXIF worked, we might be done
        if detection_method == 'exif':
            return image
        
        # Check if EXIF actually changed the image
        exif_worked = not np.array_equal(np.array(image), np.array(original_image))
        if exif_worked and detection_method == 'auto':
            return image
        
        # Step 2: Use specified method or auto-detect
        angle = 0
        
        if is_utility_bill or detection_method == 'utility':
            angle = await OrientationHandler.detect_orientation_for_utility_bill(image, surya_url)
        elif detection_method == 'surya':
            angle, _ = await OrientationHandler.detect_orientation_by_surya(image, surya_url)
        elif detection_method == 'auto':
            # Try methods in order of speed/reliability
            try:
                # For utility bills, use header pattern detection
                if is_utility_bill:
                    angle = await OrientationHandler.detect_orientation_for_utility_bill(image, surya_url)
                else:
                    # Use Surya OCR confidence comparison
                    angle, confidence = await OrientationHandler.detect_orientation_by_surya(image, surya_url)
                    # Only rotate if we have reasonable confidence
                    if confidence < 0.5:  # Threshold for minimum confidence
                        angle = 0
                        
            except Exception as e:
                logger.warning(f"Auto-orientation failed: {type(e).__name__}: {e}", exc_info=True)
                angle = 0
        
        # Apply rotation if needed
        if angle != 0:
            image = image.rotate(360 - angle, expand=True)
            logger.info(f"Rotated image by {angle}°")
        
        return image
    
    @staticmethod
    def apply_rotation(image: Image.Image, angle: int) -> Image.Image:
        """
        Apply rotation to image
        
        Args:
            image: PIL Image
            angle: Rotation angle (0, 90, 180, or 270)
            
        Returns:
            Rotated image
        """
        if angle == 0:
            return image
        
        # PIL rotates counter-clockwise, so adjust
        return image.rotate(360 - angle, expand=True)