"""Image preprocessing utilities for better VLM performance"""

from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from typing import Union, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocess images for optimal VLM performance"""
    
    def __init__(self, target_size: Tuple[int, int] = (1344, 1344)):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target image size for VLM (Qwen2-VL default)
        """
        self.target_size = target_size
    
    def preprocess(self, 
                  image: Union[str, Path, Image.Image],
                  enhance: bool = True) -> Image.Image:
        """
        Preprocess image for VLM processing.
        
        Args:
            image: Input image
            enhance: Whether to apply enhancement
            
        Returns:
            Preprocessed PIL Image
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        # Convert to RGB if needed
        if image.mode not in ('RGB', 'L'):
            if image.mode == 'RGBA':
                # Handle transparency
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            else:
                image = image.convert('RGB')
        
        # Apply enhancements if requested
        if enhance:
            image = self._enhance_image(image)
        
        # Resize if too large
        image = self._smart_resize(image)
        
        # Fix orientation
        image = self._fix_orientation(image)
        
        return image
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply enhancements for better OCR"""
        try:
            # Increase contrast slightly
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # Sharpen slightly
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # Ensure good brightness
            enhancer = ImageEnhance.Brightness(image)
            
            # Calculate average brightness
            grayscale = image.convert('L')
            brightness = np.array(grayscale).mean()
            
            # Adjust if too dark or too bright
            if brightness < 100:
                image = enhancer.enhance(1.3)
            elif brightness > 200:
                image = enhancer.enhance(0.9)
            
            return image
            
        except Exception as e:
            logger.warning(f"Enhancement failed: {e}")
            return image
    
    def _smart_resize(self, image: Image.Image) -> Image.Image:
        """Resize image intelligently to fit target size"""
        width, height = image.size
        target_w, target_h = self.target_size
        
        # Don't upscale small images too much
        if width < target_w / 2 and height < target_h / 2:
            # Just double the size
            new_size = (width * 2, height * 2)
            return image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Calculate aspect ratio
        aspect = width / height
        target_aspect = target_w / target_h
        
        if aspect > target_aspect:
            # Image is wider
            new_width = target_w
            new_height = int(target_w / aspect)
        else:
            # Image is taller
            new_height = target_h
            new_width = int(target_h * aspect)
        
        # Only resize if significantly larger
        if width > target_w * 1.5 or height > target_h * 1.5:
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
    def _fix_orientation(self, image: Image.Image) -> Image.Image:
        """Fix image orientation based on EXIF data"""
        try:
            # Get EXIF data
            exif = image._getexif()
            
            if exif:
                orientation_key = 274  # Orientation tag
                if orientation_key in exif:
                    orientation = exif[orientation_key]
                    
                    # Rotate based on orientation
                    rotations = {
                        3: 180,
                        6: 270,
                        8: 90
                    }
                    
                    if orientation in rotations:
                        image = image.rotate(rotations[orientation], expand=True)
            
            return image
            
        except:
            # If EXIF processing fails, return original
            return image
    
    def prepare_for_ocr(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """
        Special preprocessing for OCR accuracy.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image optimized for OCR
        """
        # Load and do basic preprocessing
        image = self.preprocess(image, enhance=False)
        
        # Convert to grayscale for better OCR
        if image.mode != 'L':
            image = image.convert('L')
        
        # Apply adaptive thresholding-like effect
        image_array = np.array(image)
        
        # Increase contrast
        mean = image_array.mean()
        std = image_array.std()
        
        # Normalize
        normalized = (image_array - mean) / (std + 1e-6)
        
        # Scale back to 0-255
        normalized = normalized * 50 + 128
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        
        # Convert back to PIL
        image = Image.fromarray(normalized)
        
        # Final sharpening
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)
        
        return image