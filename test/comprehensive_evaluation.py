#!/usr/bin/env python3
"""Comprehensive evaluation script that tests all files including synthetic test bills"""

import os
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess
from datetime import datetime
import Levenshtein
import re
from ocr_postprocessing import process_surya_output

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

def load_degradation_metadata() -> Dict:
    """Load degradation metadata for synthetic files"""
    metadata_file = SYNTHETIC_BILLS_DIR / "degradation_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def get_all_test_files() -> Tuple[List[Path], List[Path]]:
    """Get all test files categorized by type"""
    # Original test files
    original_files = []
    for file_path in TEST_BILLS_DIR.iterdir():
        if file_path.suffix.lower() in ['.png', '.pdf'] and file_path.is_file():
            original_files.append(file_path)
    
    # Synthetic test files
    synthetic_files = []
    if SYNTHETIC_BILLS_DIR.exists():
        for file_path in SYNTHETIC_BILLS_DIR.iterdir():
            if file_path.suffix.lower() == '.png' and file_path.is_file():
                synthetic_files.append(file_path)
    
    return sorted(original_files), sorted(synthetic_files)

def run_surya_ocr(file_path: Path, apply_postprocessing: bool = True) -> str:
    """Run Surya OCR with optional post-processing"""
    try:
        # Create simple output directory structure
        output_dir = RESULTS_DIR / "temp_ocr"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run surya_ocr with timeout to handle hanging issues
        cmd = ["surya_ocr", str(file_path), "--output_dir", str(output_dir)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            print(f"  ✗ OCR timed out after 180 seconds")
            return ""
        
        if result.returncode != 0:
            print(f"  ✗ OCR failed: {result.stderr}")
            return ""
        
        # Find output file
        results_file = output_dir / file_path.stem / "results.json"
        
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
                
            # Extract text
            text_parts = []
            for filename, pages in ocr_data.items():
                if isinstance(pages, list):
                    for page in pages:
                        if 'text_lines' in page:
                            for line in page['text_lines']:
                                text_parts.append(line.get('text', ''))
                elif isinstance(pages, dict) and 'text_lines' in pages:
                    for line in pages['text_lines']:
                        text_parts.append(line.get('text', ''))
            
            raw_text = '\n'.join(text_parts)
            
            # Clean up temp files
            import shutil
            shutil.rmtree(output_dir / file_path.stem, ignore_errors=True)
            
            # Apply post-processing if requested
            if apply_postprocessing:
                return process_surya_output(raw_text)
            else:
                return raw_text
        else:
            return ""
            
    except Exception as e:
        print(f"  ✗ Error processing {file_path.name}: {e}")
        return ""

def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
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

def get_ground_truth_for_synthetic(file_name: str, ground_truth_data: Dict) -> str:
    """Get ground truth for synthetic files based on their clean version"""
    # Extract base name (e.g., DEWA from DEWA_clean.png or DEWA_degraded_000.png)
    base_name = file_name.split('_')[0] + '.png'
    
    files_gt = ground_truth_data.get('files', {})
    if base_name in files_gt:
        return files_gt[base_name]['raw_text']
    
    return None

def evaluate_files(files: List[Path], ground_truth_data: Dict, file_type: str, degradation_metadata: Dict = None, apply_postprocessing: bool = True) -> List[Dict]:
    """Evaluate a list of files"""
    results = []
    files_gt = ground_truth_data.get('files', {})
    
    print(f"\nEvaluating {len(files)} {file_type} files...")
    print("-" * 80)
    
    for i, file_path in enumerate(files, 1):
        file_name = file_path.name
        print(f"\n[{i}/{len(files)}] Processing: {file_name}")
        
        # Get ground truth
        if file_type == "synthetic":
            gt_text = get_ground_truth_for_synthetic(file_name, ground_truth_data)
            if not gt_text:
                print(f"  ⚠ No ground truth found (based on {file_name.split('_')[0]}.png)")
                continue
        else:
            if file_name not in files_gt:
                print(f"  ⚠ No ground truth found")
                continue
            gt_text = files_gt[file_name]['raw_text']
        
        # Run OCR
        ocr_text = run_surya_ocr(file_path, apply_postprocessing=apply_postprocessing)
        
        if not ocr_text:
            print(f"  ✗ OCR failed or returned empty text")
            result = {
                'file': file_name,
                'type': file_type,
                'cer': 100.0,
                'meets_target': False,
                'status': 'failed'
            }
        else:
            # Calculate CER
            cer = calculate_cer(gt_text, ocr_text)
            
            result = {
                'file': file_name,
                'type': file_type,
                'cer': round(cer, 2),
                'meets_target': cer < 2.0,
                'status': 'success'
            }
            
            print(f"  CER: {result['cer']}%")
            print(f"  Status: {'✓ MEETS TARGET!' if result['meets_target'] else '✗ Needs improvement'}")
        
        # Add degradation info if available
        if degradation_metadata and file_name in degradation_metadata:
            result['degradation'] = degradation_metadata[file_name]
        
        results.append(result)
        
        # Save OCR output with simple naming
        if ocr_text:
            output_file = RESULTS_DIR / f"{file_path.stem}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(ocr_text)
    
    return results

def analyze_results_by_degradation(results: List[Dict]) -> Dict:
    """Analyze results grouped by degradation severity"""
    degradation_analysis = {}
    
    for result in results:
        if 'degradation' not in result:
            continue
        
        severity = result['degradation']['severity']
        if severity not in degradation_analysis:
            degradation_analysis[severity] = {
                'files': [],
                'avg_cer': 0,
                'success_rate': 0
            }
        
        degradation_analysis[severity]['files'].append(result)
    
    # Calculate averages
    for severity, data in degradation_analysis.items():
        files = data['files']
        if files:
            successful_files = [f for f in files if f['status'] == 'success']
            data['avg_cer'] = sum(f['cer'] for f in successful_files) / len(successful_files) if successful_files else 100.0
            data['success_rate'] = (len(successful_files) / len(files)) * 100
    
    return degradation_analysis

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Comprehensive Surya OCR Evaluation")
    parser.add_argument("--quick", action="store_true", help="Run quick evaluation on subset of files")
    parser.add_argument("--no-postprocessing", action="store_true", help="Disable post-processing")
    parser.add_argument("--files", nargs="+", help="Specific files to test")
    args = parser.parse_args()
    
    print("=" * 80)
    print("COMPREHENSIVE SURYA OCR EVALUATION" + (" (QUICK MODE)" if args.quick else ""))
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load ground truth and metadata
    ground_truth_data = load_ground_truth()
    degradation_metadata = load_degradation_metadata()
    
    # Get test files
    if args.files:
        # Test specific files
        original_files = []
        synthetic_files = []
        for file_path in args.files:
            path = Path(file_path)
            if path.exists():
                if "synthetic_test_bills" in str(path):
                    synthetic_files.append(path)
                else:
                    original_files.append(path)
    elif args.quick:
        # Quick mode - test subset
        original_files = [
            TEST_BILLS_DIR / "DEWA.png",
            TEST_BILLS_DIR / "SEWA.png"
        ]
        synthetic_files = [
            SYNTHETIC_BILLS_DIR / "DEWA_clean.png",
            SYNTHETIC_BILLS_DIR / "DEWA_degraded_000.png"
        ]
        # Filter existing files
        original_files = [f for f in original_files if f.exists()]
        synthetic_files = [f for f in synthetic_files if f.exists()]
    else:
        # Full evaluation
        original_files, synthetic_files = get_all_test_files()
    
    print(f"\nFound {len(original_files)} original test files")
    print(f"Found {len(synthetic_files)} synthetic test files")
    
    # Evaluate original files
    original_results = evaluate_files(original_files, ground_truth_data, "original", 
                                    apply_postprocessing=not args.no_postprocessing)
    
    # Evaluate synthetic files
    synthetic_results = evaluate_files(synthetic_files, ground_truth_data, "synthetic", 
                                     degradation_metadata, apply_postprocessing=not args.no_postprocessing)
    
    # Combine results
    all_results = original_results + synthetic_results
    
    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    if all_results:
        successful_results = [r for r in all_results if r['status'] == 'success']
        
        # Overall metrics
        if successful_results:
            avg_cer = sum(r['cer'] for r in successful_results) / len(successful_results)
            files_meeting_target = sum(1 for r in successful_results if r['meets_target'])
        else:
            avg_cer = 100.0
            files_meeting_target = 0
        
        print(f"\nTotal files tested: {len(all_results)}")
        print(f"Successful OCR runs: {len(successful_results)}/{len(all_results)}")
        print(f"Average CER (successful runs): {avg_cer:.2f}%")
        print(f"Files meeting <2% target: {files_meeting_target}/{len(successful_results)} ({(files_meeting_target/len(successful_results)*100 if successful_results else 0):.0f}%)")
        
        # Original vs Synthetic
        orig_success = [r for r in original_results if r['status'] == 'success']
        synth_success = [r for r in synthetic_results if r['status'] == 'success']
        
        if orig_success:
            orig_avg_cer = sum(r['cer'] for r in orig_success) / len(orig_success)
            print(f"\nOriginal files - Average CER: {orig_avg_cer:.2f}%")
        
        if synth_success:
            synth_avg_cer = sum(r['cer'] for r in synth_success) / len(synth_success)
            print(f"Synthetic files - Average CER: {synth_avg_cer:.2f}%")
        
        # Degradation analysis
        degradation_analysis = analyze_results_by_degradation(synthetic_results)
        if degradation_analysis:
            print("\nPerformance by degradation severity:")
            for severity in ['low', 'medium', 'high']:
                if severity in degradation_analysis:
                    data = degradation_analysis[severity]
                    print(f"  {severity.capitalize()}: CER={data['avg_cer']:.2f}%, Success rate={data['success_rate']:.0f}%")
        
        # Detailed results
        print("\nDetailed results:")
        print("-" * 80)
        
        # Failed files
        failed_files = [r for r in all_results if r['status'] == 'failed']
        if failed_files:
            print("\nFailed files:")
            for r in failed_files:
                print(f"  ✗ {r['file']}")
        
        # Successful files by CER
        if successful_results:
            print("\nSuccessful files (sorted by CER):")
            sorted_results = sorted(successful_results, key=lambda x: x['cer'])
            for r in sorted_results[:10]:  # Show top 10
                status = "✓" if r['meets_target'] else "✗"
                print(f"  {status} {r['file']}: CER = {r['cer']}%")
            
            if len(sorted_results) > 10:
                print(f"  ... and {len(sorted_results) - 10} more files")
        else:
            sorted_results = []
        
        # Save comprehensive results
        summary = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'original_files_count': len(original_files),
                'synthetic_files_count': len(synthetic_files),
                'total_files': len(all_results),
                'ground_truth_file': str(GROUND_TRUTH_FILE)
            },
            'overall_metrics': {
                'successful_runs': len(successful_results),
                'failed_runs': len(failed_files),
                'average_cer': round(avg_cer, 2),
                'files_meeting_target': files_meeting_target,
                'success_rate': round((len(successful_results) / len(all_results)) * 100, 2) if all_results else 0
            },
            'by_type': {
                'original': {
                    'count': len(original_results),
                    'successful': len(orig_success),
                    'avg_cer': round(sum(r['cer'] for r in orig_success) / len(orig_success), 2) if orig_success else None
                },
                'synthetic': {
                    'count': len(synthetic_results),
                    'successful': len(synth_success),
                    'avg_cer': round(sum(r['cer'] for r in synth_success) / len(synth_success), 2) if synth_success else None
                }
            },
            'degradation_analysis': degradation_analysis,
            'detailed_results': all_results
        }
        
        # Save simplified results
        results_file = RESULTS_DIR / "evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        # Also save summary text
        summary_file = RESULTS_DIR / "evaluation_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Surya OCR Evaluation Summary\n")
            f.write(f"{'='*40}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Average CER: {avg_cer:.2f}%\n")
            f.write(f"Files meeting <2% target: {files_meeting_target}/{len(successful_results)}\n")
            if sorted_results:
                f.write(f"\nTop performing files:\n")
                for r in sorted_results[:5]:
                    f.write(f"  - {r['file']}: CER = {r['cer']}%\n")
        
        print(f"\n✓ Results saved to: {results_file}")
        
        # Final verdict
        if avg_cer < 2.0:
            print("\n🎉 SUCCESS! Average CER is below 2% target!")
        else:
            print(f"\n⚠ Average CER ({avg_cer:.2f}%) is above 2% target. Further optimization needed.")

if __name__ == "__main__":
    main()