#!/usr/bin/env python3
"""Example script for processing utility bills"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qwen_integration.src import OCRPipeline
import json


def main():
    # Example usage
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    test_bills = [
        project_root / "test_bills" / "DEWA.png",
        project_root / "test_bills" / "SEWA.png"
    ]
    
    # Create pipeline
    pipeline = OCRPipeline()
    
    # Process each bill
    for bill_path in test_bills:
        if bill_path.exists():
            print(f"\nProcessing: {bill_path.name}")
            print("=" * 60)
            
            try:
                # Process the bill
                result = pipeline.process_bill(bill_path)
                
                # Print extracted information
                print("\nExtracted Information:")
                print(f"Provider: {result.extracted_data.bill_info.provider_name}")
                print(f"Account Number: {result.extracted_data.bill_info.account_number}")
                print(f"Bill Date: {result.extracted_data.bill_info.bill_date}")
                print(f"\nConsumption:")
                print(f"  Electricity: {result.extracted_data.consumption_data.electricity.value} kWh")
                
                if hasattr(result.extracted_data.consumption_data, 'water') and result.extracted_data.consumption_data.water:
                    print(f"  Water: {result.extracted_data.consumption_data.water.value} m³")
                
                if result.extracted_data.consumption_data.electricity.meter_reading:
                    print(f"\nMeter Readings:")
                    print(f"  Current: {result.extracted_data.consumption_data.electricity.meter_reading.current}")
                    print(f"  Previous: {result.extracted_data.consumption_data.electricity.meter_reading.previous}")
                
                print(f"\nValidation:")
                print(f"  Confidence: {result.validation.confidence:.2%}")
                print(f"  Manual Verification Required: {result.validation.manual_verification_required}")
                
                # Save to JSON
                output_file = f"{bill_path.stem}_qwen_extracted.json"
                with open(output_file, 'w') as f:
                    json.dump(result.model_dump(), f, indent=2)
                print(f"\nSaved to: {output_file}")
                
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"File not found: {bill_path}")
    
    # Cleanup
    pipeline.cleanup()


if __name__ == "__main__":
    main()