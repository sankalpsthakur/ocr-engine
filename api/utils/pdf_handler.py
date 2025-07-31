"""
PDF Handler Utility
Converts PDFs to images for OCR processing with fallback mechanisms
"""

import os
import tempfile
import logging
from typing import List, Tuple, Optional, Union
from pathlib import Path
from PIL import Image
import io
from .orientation_handler import OrientationHandler

logger = logging.getLogger(__name__)

class PDFHandler:
    """Handles PDF to image conversion with multiple fallback methods"""
    
    @staticmethod
    def is_pdf(content_type: str, filename: str) -> bool:
        """Check if file is a PDF based on content type and extension"""
        return (
            content_type == 'application/pdf' or 
            filename.lower().endswith('.pdf')
        )
    
    @staticmethod
    async def convert_to_images(
        file_content: bytes, 
        filename: str,
        dpi: int = 200,
        auto_orient: bool = True
    ) -> List[Tuple[bytes, str]]:
        """
        Convert PDF to images with fallback mechanisms and orientation correction
        
        Args:
            file_content: PDF file content as bytes
            filename: Original filename
            dpi: DPI for image conversion
            auto_orient: Whether to automatically fix orientation
            
        Returns:
            List of tuples (image_bytes, page_filename)
        """
        try:
            # Try pdf2image first (if poppler is available, it's often faster)
            return await PDFHandler._convert_with_pdf2image(file_content, filename, dpi, auto_orient)
        except Exception as e:
            # Check if it's a poppler not found error
            error_msg = str(e).lower()
            if "poppler" in error_msg or "page count" in error_msg:
                # This is expected if poppler isn't installed - log at debug level
                logger.debug(f"pdf2image conversion failed (poppler not installed): {e}. Trying PyMuPDF...")
            else:
                # Other errors should be logged as warnings
                logger.warning(f"pdf2image conversion failed: {e}. Trying PyMuPDF...")
            
            try:
                # Fallback method: PyMuPDF (no system dependencies required)
                return await PDFHandler._convert_with_pymupdf(file_content, filename, dpi, auto_orient)
            except Exception as e2:
                logger.error(f"All PDF conversion methods failed: {e2}")
                raise Exception(f"Failed to convert PDF: {str(e2)}")
    
    @staticmethod
    async def _convert_with_pdf2image(
        file_content: bytes, 
        filename: str,
        dpi: int,
        auto_orient: bool = True
    ) -> List[Tuple[bytes, str]]:
        """Convert PDF using pdf2image library"""
        try:
            from pdf2image import convert_from_bytes
        except ImportError:
            raise ImportError("pdf2image not installed")
        
        # Create temporary file for pdf2image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
            tmp_pdf.write(file_content)
            tmp_pdf_path = tmp_pdf.name
        
        try:
            # Convert PDF to images
            images = convert_from_bytes(
                file_content,
                dpi=dpi,
                fmt='PNG',
                thread_count=4,
                use_pdftocairo=True  # More reliable than pdftoppm
            )
            
            result_images = []
            base_name = Path(filename).stem
            
            for i, image in enumerate(images):
                # Apply orientation correction if enabled
                if auto_orient:
                    # Check if this appears to be a utility bill based on filename
                    is_utility = any(provider in filename.upper() for provider in ['DEWA', 'SEWA'])
                    # We're already in an async context, so we can await directly
                    image = await OrientationHandler.auto_orient(image, is_utility_bill=is_utility)
                
                # Convert PIL Image to bytes
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                img_bytes = img_buffer.getvalue()
                
                # Create filename for this page
                page_filename = f"{base_name}_page_{i+1}.png"
                result_images.append((img_bytes, page_filename))
            
            logger.info(f"Successfully converted {len(result_images)} pages using pdf2image")
            return result_images
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_pdf_path):
                os.unlink(tmp_pdf_path)
    
    @staticmethod
    async def _convert_with_pymupdf(
        file_content: bytes, 
        filename: str,
        dpi: int,
        auto_orient: bool = True
    ) -> List[Tuple[bytes, str]]:
        """Convert PDF using PyMuPDF as fallback"""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF not installed - install with: pip install pymupdf")
        
        # Open PDF from bytes
        pdf_document = fitz.open(stream=file_content, filetype="pdf")
        
        try:
            result_images = []
            base_name = Path(filename).stem
            
            # Calculate zoom factor from DPI (default PDF DPI is 72)
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                
                # Check for page rotation in PDF metadata
                rotation = page.rotation
                
                # Apply rotation correction if needed
                if rotation != 0:
                    # Create rotation matrix that includes both zoom and rotation correction
                    rotation_mat = fitz.Matrix(zoom, zoom).prerotate(-rotation)
                    pix = page.get_pixmap(matrix=rotation_mat, alpha=False)
                    logger.info(f"Page {page_num+1} had {rotation}° rotation - corrected")
                else:
                    # No rotation needed, just zoom
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # Convert to PIL Image for additional orientation correction
                img_data = pix.tobytes("png")
                
                if auto_orient:
                    # Convert to PIL Image for orientation processing
                    pil_image = Image.open(io.BytesIO(img_data))
                    
                    # Check if this appears to be a utility bill
                    is_utility = any(provider in filename.upper() for provider in ['DEWA', 'SEWA'])
                    
                    # Apply auto-orientation
                    # We're already in an async context, so we can await directly
                    pil_image = await OrientationHandler.auto_orient(pil_image, is_utility_bill=is_utility)
                    
                    # Convert back to bytes
                    img_buffer = io.BytesIO()
                    pil_image.save(img_buffer, format='PNG')
                    img_data = img_buffer.getvalue()
                
                # Create filename for this page
                page_filename = f"{base_name}_page_{page_num+1}.png"
                result_images.append((img_data, page_filename))
            
            logger.info(f"Successfully converted {len(result_images)} pages using PyMuPDF")
            return result_images
            
        finally:
            pdf_document.close()
    
    @staticmethod
    def detect_pdf_type(file_content: bytes) -> str:
        """
        Detect if PDF is digital or scanned
        
        Returns:
            'digital', 'scanned', or 'mixed'
        """
        try:
            import fitz
            pdf = fitz.open(stream=file_content, filetype="pdf")
            
            text_pages = 0
            total_pages = pdf.page_count
            
            for page_num in range(total_pages):
                page = pdf[page_num]
                text = page.get_text().strip()
                if len(text) > 50:  # Meaningful text threshold
                    text_pages += 1
            
            pdf.close()
            
            if text_pages == 0:
                return 'scanned'
            elif text_pages == total_pages:
                return 'digital'
            else:
                return 'mixed'
                
        except Exception as e:
            logger.warning(f"Could not detect PDF type: {e}")
            return 'unknown'