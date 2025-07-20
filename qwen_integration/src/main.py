"""Main orchestration for Qwen-Surya OCR pipeline"""

import logging
import sys
from pathlib import Path
from typing import Union, Optional
import json
import argparse

from .extractors import SuryaExtractor, QwenProcessor
from .models import DEWABill, SEWABill

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OCRPipeline:
    """Main pipeline for OCR and structured data extraction"""
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen3-0.6B",
        device: Optional[str] = None,
        temp_dir: Optional[Path] = None
    ):
        """Initialize the OCR pipeline
        
        Args:
            model_name: Qwen model to use
            device: Device for model (cuda/cpu/auto)
            temp_dir: Directory for temporary files
        """
        self.surya_extractor = SuryaExtractor(temp_dir=temp_dir)
        self.qwen_processor = QwenProcessor(model_name=model_name, device=device)
        self.is_initialized = False
        
    def initialize(self):
        """Initialize the pipeline (load models)"""
        if not self.is_initialized:
            logger.info("Initializing OCR pipeline...")
            self.qwen_processor.load_model()
            self.is_initialized = True
            logger.info("Pipeline initialized successfully")
    
    def detect_provider(self, file_path: Path, ocr_text: str = "") -> str:
        """Detect the utility provider from filename or OCR text
        
        Args:
            file_path: Path to the bill file
            ocr_text: Optional OCR text to check
            
        Returns:
            Provider name (DEWA/SEWA)
        """
        filename = file_path.stem.upper()
        
        # Check filename first
        if 'DEWA' in filename:
            return 'DEWA'
        elif 'SEWA' in filename:
            return 'SEWA'
            
        # Check OCR text
        text_upper = ocr_text.upper()
        if 'DUBAI ELECTRICITY' in text_upper or 'DEWA' in text_upper:
            return 'DEWA'
        elif 'SHARJAH ELECTRICITY' in text_upper or 'SEWA' in text_upper:
            return 'SEWA'
            
        # Default based on content
        if 'water' in text_upper and 'm³' in ocr_text:
            return 'SEWA'  # SEWA bills typically have water consumption
        else:
            return 'DEWA'  # Default to DEWA
    
    def process_bill(
        self, 
        file_path: Union[str, Path],
        provider: Optional[str] = None,
        apply_postprocessing: bool = True
    ) -> Union[DEWABill, SEWABill]:
        """Process a utility bill file
        
        Args:
            file_path: Path to the bill file
            provider: Optional provider override (DEWA/SEWA)
            apply_postprocessing: Whether to apply OCR post-processing
            
        Returns:
            Pydantic model with extracted data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Initialize if needed
        if not self.is_initialized:
            self.initialize()
            
        logger.info(f"Processing bill: {file_path.name}")
        
        # Step 1: Extract text using Surya OCR
        if apply_postprocessing:
            ocr_text, ocr_metadata = self.surya_extractor.extract_with_postprocessing(file_path)
        else:
            ocr_text, ocr_metadata = self.surya_extractor.extract_text(file_path)
            
        if not ocr_text:
            error_msg = ocr_metadata.get('error', 'Unknown OCR error')
            raise RuntimeError(f"OCR extraction failed: {error_msg}")
            
        logger.info(
            f"OCR extraction complete: {ocr_metadata.get('total_lines', 0)} lines, "
            f"confidence: {ocr_metadata.get('average_confidence', 0):.2%}"
        )
        
        # Step 2: Detect provider if not specified
        if not provider:
            provider = self.detect_provider(file_path, ocr_text)
            logger.info(f"Detected provider: {provider}")
        
        # Step 3: Validate OCR output
        validations = self.surya_extractor.validate_extraction(ocr_text, provider)
        if not validations['overall_valid']:
            logger.warning(f"OCR validation warnings: {validations}")
        
        # Step 4: Process with Qwen to extract structured data
        try:
            bill_model = self.qwen_processor.process_to_pydantic(
                ocr_text=ocr_text,
                provider=provider,
                source_document=file_path.name
            )
            
            # Add OCR metadata to model
            if hasattr(bill_model.metadata, 'processing_time_seconds'):
                total_time = (
                    bill_model.metadata.processing_time_seconds + 
                    ocr_metadata.get('processing_time_seconds', 0)
                )
                bill_model.metadata.processing_time_seconds = total_time
                
            logger.info(
                f"Structured extraction complete. "
                f"Confidence: {bill_model.validation.confidence:.2%}, "
                f"Manual verification: {bill_model.validation.manual_verification_required}"
            )
            
            return bill_model
            
        except Exception as e:
            logger.error(f"Failed to extract structured data: {e}")
            raise
    
    def process_batch(
        self, 
        file_paths: list[Union[str, Path]],
        output_dir: Optional[Path] = None
    ) -> list[Union[DEWABill, SEWABill]]:
        """Process multiple bills
        
        Args:
            file_paths: List of file paths
            output_dir: Optional directory to save results
            
        Returns:
            List of processed bill models
        """
        results = []
        
        for file_path in file_paths:
            try:
                bill = self.process_bill(file_path)
                results.append(bill)
                
                # Save result if output directory specified
                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(exist_ok=True, parents=True)
                    
                    output_file = output_dir / f"{Path(file_path).stem}_extracted.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(bill.model_dump(), f, indent=2)
                        
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results.append(None)
                
        return results
    
    def cleanup(self):
        """Clean up resources"""
        self.surya_extractor.cleanup()
        self.qwen_processor.unload_model()
        self.is_initialized = False


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Extract structured data from utility bills using Surya OCR and Qwen"
    )
    parser.add_argument(
        "files", 
        nargs="+", 
        help="Path(s) to bill files (PNG, PDF)"
    )
    parser.add_argument(
        "--provider", 
        choices=["DEWA", "SEWA"],
        help="Override provider detection"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path,
        help="Directory to save extracted JSON files"
    )
    parser.add_argument(
        "--no-postprocessing",
        action="store_true",
        help="Disable OCR post-processing"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device for Qwen model"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Qwen model name"
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = OCRPipeline(
        model_name=args.model,
        device=args.device if args.device != "auto" else None
    )
    
    try:
        # Process files
        for file_path in args.files:
            file_path = Path(file_path)
            print(f"\nProcessing: {file_path.name}")
            print("-" * 50)
            
            try:
                bill = pipeline.process_bill(
                    file_path,
                    provider=args.provider,
                    apply_postprocessing=not args.no_postprocessing
                )
                
                # Print summary
                print(f"Provider: {bill.extracted_data.bill_info.provider_name}")
                print(f"Account: {bill.extracted_data.bill_info.account_number}")
                print(f"Date: {bill.extracted_data.bill_info.bill_date}")
                print(f"Electricity: {bill.extracted_data.consumption_data.electricity.value} kWh")
                
                if hasattr(bill.extracted_data.consumption_data, 'water') and bill.extracted_data.consumption_data.water:
                    print(f"Water: {bill.extracted_data.consumption_data.water.value} m³")
                    
                print(f"Confidence: {bill.validation.confidence:.2%}")
                print(f"Manual verification required: {bill.validation.manual_verification_required}")
                
                # Save if requested
                if args.output_dir:
                    args.output_dir.mkdir(exist_ok=True, parents=True)
                    output_file = args.output_dir / f"{file_path.stem}_extracted.json"
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(bill.model_dump(), f, indent=2)
                        
                    print(f"Saved to: {output_file}")
                    
            except Exception as e:
                print(f"ERROR: {e}")
                if args.output_dir:
                    # Save error report
                    error_file = args.output_dir / f"{file_path.stem}_error.json"
                    with open(error_file, 'w') as f:
                        json.dump({"error": str(e), "file": str(file_path)}, f, indent=2)
                        
    finally:
        # Cleanup
        pipeline.cleanup()
        

if __name__ == "__main__":
    main()