"""
Image compression utility for OCR Engine
Compresses images larger than 512KB while preserving aspect ratio
"""

import io
import logging
from typing import Tuple, Optional
from PIL import Image

logger = logging.getLogger(__name__)

def is_image_format(content_type: str) -> bool:
    """Check if the content type indicates an image format"""
    return content_type and content_type.startswith('image/')

def get_image_size_kb(content: bytes) -> float:
    """Get image size in KB"""
    return len(content) / 1024

def compress_image(
    content: bytes, 
    filename: str,
    target_size_kb: int = 512,
    min_quality: int = 30,
    max_dimension: int = 2048
) -> Tuple[bytes, bool]:
    """
    Compress image to target size while preserving aspect ratio
    
    Args:
        content: Image content in bytes
        filename: Original filename for logging
        target_size_kb: Target size in KB (default 512KB)
        min_quality: Minimum JPEG quality to try (default 30)
        max_dimension: Maximum width or height in pixels (default 2048)
    
    Returns:
        Tuple of (compressed_content, was_compressed)
    """
    original_size_kb = get_image_size_kb(content)
    
    # If already under target size, return as-is
    if original_size_kb <= target_size_kb:
        logger.info(f"Image {filename} is already under {target_size_kb}KB ({original_size_kb:.1f}KB)")
        return content, False
    
    logger.info(f"Compressing image {filename} from {original_size_kb:.1f}KB to under {target_size_kb}KB")
    
    try:
        # Load image
        img = Image.open(io.BytesIO(content))
        
        # Convert RGBA to RGB if needed (JPEG doesn't support transparency)
        if img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            rgb_img.paste(img, mask=img.split()[-1] if 'A' in img.mode else None)
            img = rgb_img
        
        # Get original dimensions
        original_width, original_height = img.size
        logger.info(f"Original dimensions: {original_width}x{original_height}")
        
        # Check if image dimensions exceed max_dimension
        needs_resize = original_width > max_dimension or original_height > max_dimension
        
        if needs_resize:
            # Calculate scale to fit within max_dimension while preserving aspect ratio
            scale = min(max_dimension / original_width, max_dimension / original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            logger.info(f"Resizing image to {new_width}x{new_height} to fit within {max_dimension}x{max_dimension}")
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Try quality reduction first (95, 85, 75, 65, 55, 45, 35, 30)
        quality_levels = [95, 85, 75, 65, 55, 45, 35, 30]
        
        for quality in quality_levels:
            if quality < min_quality:
                break
                
            # Save to bytes buffer with current quality
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            compressed_content = buffer.getvalue()
            compressed_size_kb = get_image_size_kb(compressed_content)
            
            logger.debug(f"Quality {quality}%: {compressed_size_kb:.1f}KB")
            
            if compressed_size_kb <= target_size_kb:
                logger.info(f"Compressed to {compressed_size_kb:.1f}KB using quality {quality}%")
                return compressed_content, True
        
        # If quality reduction alone isn't enough, resize proportionally
        logger.info("Quality reduction insufficient, resizing image proportionally")
        
        # Calculate scale factor to achieve target size
        # Estimate that file size scales roughly with area
        current_size_kb = compressed_size_kb  # Last attempt size
        scale_factor = (target_size_kb / current_size_kb) ** 0.5
        
        # Apply safety margin
        scale_factor *= 0.9
        
        # Resize progressively
        while scale_factor > 0.1:
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            # Ensure minimum dimensions
            if new_width < 800 or new_height < 600:
                logger.warning(f"Image would be too small ({new_width}x{new_height}), using minimum size")
                new_width = max(800, new_width)
                new_height = max(600, new_height)
            
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Try with quality 85 after resize
            buffer = io.BytesIO()
            resized_img.save(buffer, format='JPEG', quality=85, optimize=True)
            compressed_content = buffer.getvalue()
            compressed_size_kb = get_image_size_kb(compressed_content)
            
            logger.debug(f"Resized to {new_width}x{new_height}: {compressed_size_kb:.1f}KB")
            
            if compressed_size_kb <= target_size_kb:
                logger.info(f"Compressed to {compressed_size_kb:.1f}KB with dimensions {new_width}x{new_height}")
                return compressed_content, True
            
            # Reduce scale factor for next iteration
            scale_factor *= 0.8
        
        # If we still can't achieve target, return best effort
        logger.warning(f"Could not achieve target size. Final size: {compressed_size_kb:.1f}KB")
        return compressed_content, True
        
    except Exception as e:
        logger.error(f"Error compressing image {filename}: {e}")
        # Return original on error
        return content, False

def compress_image_for_qwen(
    content: bytes,
    filename: str,
    content_type: str
) -> Tuple[bytes, str]:
    """
    Compress image for Qwen VL processing if needed
    
    Args:
        content: Image content
        filename: Original filename
        content_type: MIME type
        
    Returns:
        Tuple of (content, content_type) - possibly compressed
    """
    # Only process images
    if not is_image_format(content_type):
        return content, content_type
    
    # Only compress if > 512KB
    if get_image_size_kb(content) <= 512:
        return content, content_type
    
    # Compress the image
    compressed_content, was_compressed = compress_image(content, filename)
    
    if was_compressed:
        # Update content type to JPEG if compressed
        return compressed_content, 'image/jpeg'
    
    return content, content_type

def compress_image_for_ocr(
    content: bytes,
    filename: str,
    content_type: str
) -> Tuple[bytes, str]:
    """
    Compress image for OCR processing if needed (target: 1MB)
    Uses higher quality settings to maintain OCR accuracy
    
    Args:
        content: Image content
        filename: Original filename
        content_type: MIME type
        
    Returns:
        Tuple of (content, content_type) - possibly compressed
    """
    # Only process images
    if not is_image_format(content_type):
        return content, content_type
    
    # Only compress if > 1MB
    if get_image_size_kb(content) <= 1024:
        return content, content_type
    
    # Compress the image with OCR-specific settings
    # Higher min quality (70) and larger max dimension (3000) for better OCR accuracy
    compressed_content, was_compressed = compress_image(
        content, 
        filename,
        target_size_kb=1024,  # 1MB target
        min_quality=70,       # Higher min quality for OCR
        max_dimension=3000    # Larger max dimension for OCR
    )
    
    if was_compressed:
        # Update content type to JPEG if compressed
        return compressed_content, 'image/jpeg'
    
    return content, content_type