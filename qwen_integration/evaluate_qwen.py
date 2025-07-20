#!/usr/bin/env python3
"""Comprehensive evaluation script for Qwen-enhanced OCR pipeline"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import Levenshtein

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qwen_integration.src import OCRPipeline
from qwen_integration.src.models import DEWABill, SEWABill

# Configuration
TEST_BILLS_DIR = Path(__file__).parent.parent / "test_bills"
SYNTHETIC_BILLS_DIR = TEST_BILLS_DIR / "synthetic_test_bills"
GROUND_TRUTH_FILE = Path(__file__).parent.parent / "benchmark_output_ground_truth" / "raw_text_ground_truth.json"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_ground_truth() -> Dict:
    """Load ground truth data from JSON file"""
    with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    import re
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate"""
    if not reference:
        return 100.0 if hypothesis else 0.0
    
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    
    distance = Levenshtein.distance(ref_norm, hyp_norm)
    cer = (distance / len(ref_norm)) * 100
    return cer


def extract_text_from_model(bill_model) -> str:
    """Extract text representation from Pydantic model for CER calculation"""
    parts = []
    
    # Extract key information
    data = bill_model.extracted_data
    
    # Provider and account
    parts.append(data.bill_info.provider_name)
    if data.bill_info.account_number:
        parts.append(f"Account Number: {data.bill_info.account_number}")
    
    # Bill date
    if data.bill_info.bill_date:
        parts.append(f"Bill Date: {data.bill_info.bill_date}")
    
    # Electricity consumption
    if data.consumption_data.electricity:
        parts.append(f"Electricity: {data.consumption_data.electricity.value} {data.consumption_data.electricity.unit}")
        
        # Meter readings
        if data.consumption_data.electricity.meter_reading:
            if data.consumption_data.electricity.meter_reading.current:
                parts.append(f"Current Reading: {data.consumption_data.electricity.meter_reading.current}")
            if data.consumption_data.electricity.meter_reading.previous:
                parts.append(f"Previous Reading: {data.consumption_data.electricity.meter_reading.previous}")
    
    # Water consumption (SEWA)
    if hasattr(data.consumption_data, 'water') and data.consumption_data.water:
        parts.append(f"Water: {data.consumption_data.water.value} {data.consumption_data.water.unit}")
    
    # Emissions (DEWA)
    if hasattr(data, 'emissions_data') and data.emissions_data:
        if 'totalCO2e' in data.emissions_data.scope2:
            co2_value = data.emissions_data.scope2['totalCO2e']['value']
            if co2_value > 0:
                parts.append(f"CO2 Emissions: {co2_value} kgCO2e")
    
    return '\n'.join(parts)


def evaluate_single_file(pipeline: OCRPipeline, file_path: Path, ground_truth: str) -> Dict:
    """Evaluate a single file"""
    print(f"Processing: {file_path.name}")
    
    try:
        # Process with Qwen pipeline
        start_time = datetime.now()
        bill_model = pipeline.process_bill(file_path)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Extract text for CER calculation
        extracted_text = extract_text_from_model(bill_model)
        
        # Also get raw OCR text for comparison
        raw_ocr_text, _ = pipeline.surya_extractor.extract_text(file_path)
        
        # Calculate CER for both
        cer_structured = calculate_cer(ground_truth, extracted_text)
        cer_raw = calculate_cer(ground_truth, raw_ocr_text)
        
        # Field accuracy check
        field_accuracy = {
            'has_provider': bool(bill_model.extracted_data.bill_info.provider_name),
            'has_account': bool(bill_model.extracted_data.bill_info.account_number),
            'has_date': bool(bill_model.extracted_data.bill_info.bill_date),
            'has_electricity': bill_model.extracted_data.consumption_data.electricity.value > 0,
            'has_meter_reading': bool(
                bill_model.extracted_data.consumption_data.electricity.meter_reading and
                bill_model.extracted_data.consumption_data.electricity.meter_reading.current
            )
        }
        
        # Provider-specific checks
        if "SEWA" in bill_model.extracted_data.bill_info.provider_name:
            field_accuracy['has_water'] = (
                hasattr(bill_model.extracted_data.consumption_data, 'water') and
                bill_model.extracted_data.consumption_data.water and
                bill_model.extracted_data.consumption_data.water.value > 0
            )
        
        result = {
            'file': file_path.name,
            'status': 'success',
            'cer_structured': round(cer_structured, 2),
            'cer_raw': round(cer_raw, 2),
            'cer_improvement': round(cer_raw - cer_structured, 2),
            'meets_target': cer_structured < 2.0,
            'confidence': bill_model.validation.confidence,
            'manual_verification_required': bill_model.validation.manual_verification_required,
            'processing_time': round(processing_time, 2),
            'field_accuracy': field_accuracy,
            'field_completeness': sum(field_accuracy.values()) / len(field_accuracy)
        }
        
        print(f"  CER (Structured): {result['cer_structured']}%")
        print(f"  CER (Raw OCR): {result['cer_raw']}%")
        print(f"  Improvement: {result['cer_improvement']}%")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Field Completeness: {result['field_completeness']:.2%}")
        print(f"  Status: {'✓ MEETS TARGET!' if result['meets_target'] else '✗ Needs improvement'}")
        
        # Save extracted data
        output_file = RESULTS_DIR / f"{file_path.stem}_qwen_extracted.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(bill_model.model_dump(), f, indent=2)
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        result = {
            'file': file_path.name,
            'status': 'failed',
            'error': str(e),
            'cer_structured': 100.0,
            'cer_raw': 100.0,
            'meets_target': False
        }
    
    return result


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate Qwen-enhanced OCR pipeline")
    parser.add_argument("--quick", action="store_true", help="Run quick evaluation on subset")
    parser.add_argument("--files", nargs="+", help="Specific files to test")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu", help="Device for Qwen model")
    args = parser.parse_args()
    
    print("=" * 80)
    print("QWEN-ENHANCED OCR EVALUATION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {args.device}")
    
    # Load ground truth
    ground_truth_data = load_ground_truth()
    files_gt = ground_truth_data.get('files', {})
    
    # Create pipeline
    pipeline = OCRPipeline(device=args.device)
    pipeline.initialize()
    
    # Get test files
    test_files = []
    
    if args.files:
        for file_path in args.files:
            path = Path(file_path)
            if path.exists():
                test_files.append(path)
    elif args.quick:
        # Quick mode - test subset
        test_files = [
            TEST_BILLS_DIR / "DEWA.png",
            TEST_BILLS_DIR / "SEWA.png"
        ]
        test_files = [f for f in test_files if f.exists()]
    else:
        # Full evaluation
        for pattern in ["*.png", "*.pdf"]:
            test_files.extend(list(TEST_BILLS_DIR.glob(pattern)))
    
    print(f"\nFound {len(test_files)} test files")
    print("-" * 80)
    
    # Evaluate files
    results = []
    for file_path in test_files:
        if file_path.name in files_gt:
            gt_text = files_gt[file_path.name]['raw_text']
            result = evaluate_single_file(pipeline, file_path, gt_text)
            results.append(result)
        else:
            print(f"\nSkipping {file_path.name} - no ground truth available")
    
    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    if results:
        successful_results = [r for r in results if r['status'] == 'success']
        
        if successful_results:
            # Overall metrics
            avg_cer_structured = sum(r['cer_structured'] for r in successful_results) / len(successful_results)
            avg_cer_raw = sum(r['cer_raw'] for r in successful_results) / len(successful_results)
            avg_improvement = sum(r['cer_improvement'] for r in successful_results) / len(successful_results)
            avg_confidence = sum(r['confidence'] for r in successful_results) / len(successful_results)
            avg_processing_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
            files_meeting_target = sum(1 for r in successful_results if r['meets_target'])
            
            print(f"\nTotal files tested: {len(results)}")
            print(f"Successful runs: {len(successful_results)}/{len(results)}")
            print(f"\nAverage CER (Qwen-structured): {avg_cer_structured:.2f}%")
            print(f"Average CER (Raw OCR): {avg_cer_raw:.2f}%")
            print(f"Average Improvement: {avg_improvement:.2f}%")
            print(f"Average Confidence: {avg_confidence:.2%}")
            print(f"Average Processing Time: {avg_processing_time:.1f}s")
            print(f"\nFiles meeting <2% target: {files_meeting_target}/{len(successful_results)} ({files_meeting_target/len(successful_results)*100:.0f}%)")
            
            # Field completeness
            avg_field_completeness = sum(r.get('field_completeness', 0) for r in successful_results) / len(successful_results)
            print(f"Average Field Completeness: {avg_field_completeness:.2%}")
            
            # Best and worst performers
            sorted_by_cer = sorted(successful_results, key=lambda x: x['cer_structured'])
            print(f"\nBest performer: {sorted_by_cer[0]['file']} (CER: {sorted_by_cer[0]['cer_structured']}%)")
            print(f"Worst performer: {sorted_by_cer[-1]['file']} (CER: {sorted_by_cer[-1]['cer_structured']}%)")
            
            # Save summary
            summary = {
                'timestamp': datetime.now().isoformat(),
                'configuration': {
                    'device': args.device,
                    'total_files': len(results),
                    'successful_runs': len(successful_results)
                },
                'overall_metrics': {
                    'avg_cer_structured': round(avg_cer_structured, 2),
                    'avg_cer_raw': round(avg_cer_raw, 2),
                    'avg_improvement': round(avg_improvement, 2),
                    'avg_confidence': round(avg_confidence, 4),
                    'avg_processing_time': round(avg_processing_time, 2),
                    'files_meeting_target': files_meeting_target,
                    'success_rate': round((files_meeting_target / len(successful_results)) * 100, 2),
                    'avg_field_completeness': round(avg_field_completeness, 4)
                },
                'detailed_results': results
            }
            
            summary_file = RESULTS_DIR / "qwen_evaluation_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n✓ Results saved to: {summary_file}")
            
            # Final verdict
            if avg_cer_structured < 2.0:
                print("\n🎉 SUCCESS! Average CER is below 2% target!")
            else:
                print(f"\n⚠ Average CER ({avg_cer_structured:.2f}%) is above 2% target.")
    
    # Cleanup
    pipeline.cleanup()


if __name__ == "__main__":
    main()