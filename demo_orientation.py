#!/usr/bin/env python3
"""
Demo script to test orientation correction
Tests both image and PDF orientation handling
"""

import asyncio
import sys
import os
from pathlib import Path
from PIL import Image
import io

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from api.utils.orientation_handler import OrientationHandler

async def test_image_orientation():
    """Test orientation correction on images"""
    print("=== Testing Image Orientation Correction ===\n")
    
    # Test with a sample image (you'll need to provide test images)
    test_images = [
        "test_bills/DEWA.png",
        "test_bills/SEWA.png"
    ]
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"Skipping {image_path} - file not found")
            continue
            
        print(f"\nTesting: {image_path}")
        
        # Load image
        image = Image.open(image_path)
        print(f"Original size: {image.size}")
        
        # Test EXIF orientation
        print("\n1. Testing EXIF orientation fix...")
        exif_fixed = OrientationHandler.fix_image_orientation(image)
        print(f"   After EXIF fix: {exif_fixed.size}")
        
        # Test auto-orientation with Surya
        print("\n2. Testing auto-orientation with Surya OCR...")
        try:
            # Set Surya service URL (adjust if needed)
            OrientationHandler.SURYA_SERVICE_URL = "http://localhost:8001"
            
            # Check if it's a utility bill
            is_utility = any(provider in image_path.upper() for provider in ['DEWA', 'SEWA'])
            
            auto_oriented = await OrientationHandler.auto_orient(
                image, 
                is_utility_bill=is_utility
            )
            print(f"   After auto-orientation: {auto_oriented.size}")
            
            # Save corrected image
            output_path = f"oriented_{Path(image_path).name}"
            auto_oriented.save(output_path)
            print(f"   Saved to: {output_path}")
            
        except Exception as e:
            print(f"   Error: {e}")
            print("   Make sure Surya service is running on port 8001")

async def test_rotated_images():
    """Create and test rotated versions of images"""
    print("\n\n=== Testing Rotated Image Detection ===\n")
    
    test_image = "test_bills/DEWA.png"
    if not os.path.exists(test_image):
        print(f"Test image {test_image} not found")
        return
    
    # Create rotated versions
    original = Image.open(test_image)
    rotations = [90, 180, 270]
    
    for rotation in rotations:
        print(f"\nTesting {rotation}° rotated image...")
        
        # Create rotated image
        rotated = original.rotate(rotation, expand=True)
        
        # Save rotated version
        rotated_path = f"test_rotated_{rotation}.png"
        rotated.save(rotated_path)
        print(f"Created: {rotated_path}")
        
        # Test orientation detection
        try:
            angle, confidence = await OrientationHandler.detect_orientation_by_surya(rotated)
            print(f"Detected rotation: {angle}° (confidence: {confidence:.2f})")
            
            # Apply correction
            corrected = await OrientationHandler.auto_orient(rotated, is_utility_bill=True)
            corrected_path = f"test_rotated_{rotation}_corrected.png"
            corrected.save(corrected_path)
            print(f"Corrected and saved to: {corrected_path}")
            
        except Exception as e:
            print(f"Error: {e}")

async def test_pdf_orientation():
    """Test PDF orientation handling"""
    print("\n\n=== Testing PDF Orientation ===\n")
    
    from api.utils.pdf_handler import PDFHandler
    
    test_pdfs = ["test_bills/DEWA.pdf", "test_bills/SEWA.pdf"]
    
    for pdf_path in test_pdfs:
        if not os.path.exists(pdf_path):
            print(f"Skipping {pdf_path} - file not found")
            continue
            
        print(f"\nTesting PDF: {pdf_path}")
        
        try:
            # Read PDF content
            with open(pdf_path, 'rb') as f:
                pdf_content = f.read()
            
            # Convert with orientation correction
            print("Converting PDF to images with orientation correction...")
            images = await PDFHandler.convert_to_images(
                pdf_content, 
                pdf_path,
                auto_orient=True
            )
            
            print(f"Converted {len(images)} pages")
            
            # Save converted images
            for i, (img_bytes, img_name) in enumerate(images):
                output_path = f"pdf_page_{i+1}_{img_name}"
                with open(output_path, 'wb') as f:
                    f.write(img_bytes)
                print(f"Saved: {output_path}")
                
        except Exception as e:
            print(f"Error: {e}")

async def main():
    """Run all orientation tests"""
    print("OCR Engine - Orientation Correction Demo")
    print("========================================")
    print("\nNote: Make sure the Surya service is running on port 8001")
    print("Run: python services/surya/surya_service.py\n")
    
    # Run tests
    await test_image_orientation()
    await test_rotated_images()
    await test_pdf_orientation()
    
    print("\n\nDemo complete!")
    print("\nCheck the generated files to verify orientation correction:")
    print("- oriented_*.png - Auto-oriented images")
    print("- test_rotated_*_corrected.png - Corrected rotated images")
    print("- pdf_page_*.png - PDF pages with orientation correction")

if __name__ == "__main__":
    asyncio.run(main())