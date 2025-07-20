#!/usr/bin/env python3
"""
Comprehensive profiling script for Surya OCR pipeline
Measures time and memory usage for the complete pipeline including post-processing
"""

import time
import psutil
import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import tracemalloc

# Add test directory to path for imports
sys.path.append(str(Path(__file__).parent / "test"))

try:
    from ocr_postprocessing import process_surya_output
except ImportError:
    print("Warning: Could not import post-processing module")
    def process_surya_output(text: str) -> str:
        return text

# Surya imports
try:
    from PIL import Image
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    print("✅ Surya modules imported successfully")
except ImportError as e:
    print(f"❌ Error importing Surya modules: {e}")
    sys.exit(1)

@dataclass
class ProfileResult:
    file_name: str
    total_time: float
    ocr_time: float
    postprocessing_time: float
    peak_memory_mb: float
    memory_usage_over_time: List[Tuple[float, float]]
    raw_text_length: int
    processed_text_length: int
    text_lines_count: int
    success: bool
    error_message: str = ""

class MemoryProfiler:
    """Track memory usage over time"""
    
    def __init__(self):
        self.start_time = None
        self.memory_samples = []
        self.peak_memory = 0
    
    def start(self):
        """Start memory profiling"""
        self.start_time = time.time()
        self.memory_samples = []
        self.peak_memory = 0
        tracemalloc.start()
    
    def sample(self):
        """Take a memory sample"""
        if self.start_time is None:
            return
        
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        elapsed_time = time.time() - self.start_time
        
        self.memory_samples.append((elapsed_time, memory_mb))
        self.peak_memory = max(self.peak_memory, memory_mb)
    
    def stop(self):
        """Stop memory profiling and return peak memory"""
        tracemalloc.stop()
        return self.peak_memory, self.memory_samples

class SuryaPipelineProfiler:
    """Profile the complete Surya OCR pipeline"""
    
    def __init__(self, test_dir: str = "test_bills/"):
        self.test_dir = Path(test_dir)
        self.detection_predictor = None
        self.recognition_predictor = None
    
    def load_models(self):
        """Load Surya models"""
        print("Loading Surya models...")
        start_time = time.time()
        
        try:
            self.detection_predictor = DetectionPredictor()
            self.recognition_predictor = RecognitionPredictor()
            
            load_time = time.time() - start_time
            print(f"✅ Models loaded successfully in {load_time:.2f}s")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def run_surya_ocr(self, file_path: Path, memory_profiler: MemoryProfiler) -> Tuple[str, float, int]:
        """Run Surya OCR and return extracted text, execution time, and text lines count"""
        print(f"  🔍 Running Surya OCR on {file_path.name}...")
        
        memory_profiler.sample()
        start_time = time.time()
        
        try:
            # Load image
            image = Image.open(file_path)
            memory_profiler.sample()
            
            # Run OCR using the programmatic API
            predictions = self.recognition_predictor([image], det_predictor=self.detection_predictor)
            
            memory_profiler.sample()
            
            # Extract text from predictions
            raw_text = ""
            text_lines_count = 0
            
            if predictions and len(predictions) > 0:
                # Get first page predictions (OCRResult object)
                ocr_result = predictions[0]
                
                # Access text_lines attribute directly from OCRResult object
                if hasattr(ocr_result, 'text_lines'):
                    text_lines = ocr_result.text_lines
                    text_lines_count = len(text_lines)
                    
                    # Combine all text lines
                    text_parts = []
                    for line in text_lines:
                        if hasattr(line, 'text'):
                            text_parts.append(line.text)
                    
                    raw_text = '\n'.join(text_parts)
            
            ocr_time = time.time() - start_time
            memory_profiler.sample()
            
            print(f"    ✅ OCR completed in {ocr_time:.2f}s")
            print(f"    📝 Extracted {len(raw_text)} characters from {text_lines_count} text lines")
            
            return raw_text, ocr_time, text_lines_count
            
        except Exception as e:
            print(f"    ❌ OCR failed: {e}")
            raise e
    
    def profile_file(self, file_path: Path) -> ProfileResult:
        """Profile a single file through the complete pipeline"""
        print(f"\n🔄 Profiling {file_path.name}")
        
        # Initialize memory profiler
        memory_profiler = MemoryProfiler()
        memory_profiler.start()
        
        total_start_time = time.time()
        
        try:
            # Run Surya OCR
            print("Running Surya OCR...")
            memory_profiler.sample()
            
            raw_text, ocr_time, text_lines_count = self.run_surya_ocr(file_path, memory_profiler)
            
            # Post-processing
            print("  🔧 Running post-processing...")
            post_start_time = time.time()
            memory_profiler.sample()
            
            processed_text = process_surya_output(raw_text)
            
            postprocessing_time = time.time() - post_start_time
            memory_profiler.sample()
            
            total_time = time.time() - total_start_time
            peak_memory, memory_samples = memory_profiler.stop()
            
            print(f"  ✅ Processing completed in {total_time:.2f}s")
            print(f"  📊 Peak memory usage: {peak_memory:.1f} MB")
            print(f"  📝 Text length: {len(raw_text)} → {len(processed_text)} characters")
            
            return ProfileResult(
                file_name=file_path.name,
                total_time=total_time,
                ocr_time=ocr_time,
                postprocessing_time=postprocessing_time,
                peak_memory_mb=peak_memory,
                memory_usage_over_time=memory_samples,
                raw_text_length=len(raw_text),
                processed_text_length=len(processed_text),
                text_lines_count=text_lines_count,
                success=True
            )
            
        except Exception as e:
            total_time = time.time() - total_start_time
            peak_memory, memory_samples = memory_profiler.stop()
            
            print(f"  ❌ Error processing {file_path.name}: {e}")
            
            return ProfileResult(
                file_name=file_path.name,
                total_time=total_time,
                ocr_time=0,
                postprocessing_time=0,
                peak_memory_mb=peak_memory,
                memory_usage_over_time=memory_samples,
                raw_text_length=0,
                processed_text_length=0,
                text_lines_count=0,
                success=False,
                error_message=str(e)
            )
    
    def run_comprehensive_profiling(self, files: List[Path]) -> List[ProfileResult]:
        """Run comprehensive profiling on multiple files"""
        print(f"🚀 Starting Surya OCR Pipeline Profiling")
        print(f"📁 Test directory: {self.test_dir}")
        print(f"📋 Files to process: {len(files)}")
        
        # Load models first
        if not self.load_models():
            print("❌ Failed to load models, aborting profiling")
            return []
        
        results = []
        
        for i, file_path in enumerate(files, 1):
            print(f"\n{'='*60}")
            print(f"Processing file {i}/{len(files)}: {file_path.name}")
            print(f"{'='*60}")
            
            result = self.profile_file(file_path)
            results.append(result)
        
        return results

def save_results(results: List[ProfileResult], output_file: str = "profiling_results.json"):
    """Save profiling results to JSON file"""
    
    # Convert results to serializable format
    serializable_results = []
    for result in results:
        serializable_results.append({
            "file_name": result.file_name,
            "total_time": result.total_time,
            "ocr_time": result.ocr_time,
            "postprocessing_time": result.postprocessing_time,
            "peak_memory_mb": result.peak_memory_mb,
            "memory_usage_over_time": result.memory_usage_over_time,
            "raw_text_length": result.raw_text_length,
            "processed_text_length": result.processed_text_length,
            "text_lines_count": result.text_lines_count,
            "success": result.success,
            "error_message": result.error_message
        })
    
    # Add summary statistics
    successful_results = [r for r in results if r.success]
    
    if successful_results:
        summary = {
            "total_files": len(results),
            "successful_files": len(successful_results),
            "failed_files": len(results) - len(successful_results),
            "average_total_time": sum(r.total_time for r in successful_results) / len(successful_results),
            "average_ocr_time": sum(r.ocr_time for r in successful_results) / len(successful_results),
            "average_postprocessing_time": sum(r.postprocessing_time for r in successful_results) / len(successful_results),
            "average_peak_memory_mb": sum(r.peak_memory_mb for r in successful_results) / len(successful_results),
            "max_peak_memory_mb": max(r.peak_memory_mb for r in successful_results),
            "min_peak_memory_mb": min(r.peak_memory_mb for r in successful_results),
            "total_text_lines": sum(r.text_lines_count for r in successful_results),
            "average_text_lines_per_file": sum(r.text_lines_count for r in successful_results) / len(successful_results)
        }
    else:
        summary = {
            "total_files": len(results),
            "successful_files": 0,
            "failed_files": len(results),
            "error": "No files processed successfully"
        }
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "detailed_results": serializable_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n📁 Results saved to {output_file}")
    return output_data

def print_summary(results: List[ProfileResult]):
    """Print a summary of profiling results"""
    
    successful_results = [r for r in results if r.success]
    
    print(f"\n{'='*80}")
    print(f"📊 PROFILING SUMMARY")
    print(f"{'='*80}")
    
    print(f"📁 Total files processed: {len(results)}")
    print(f"✅ Successful: {len(successful_results)}")
    print(f"❌ Failed: {len(results) - len(successful_results)}")
    
    if successful_results:
        print(f"\n⏱️  TIMING ANALYSIS:")
        print(f"   Average total time: {sum(r.total_time for r in successful_results) / len(successful_results):.2f}s")
        print(f"   Average OCR time: {sum(r.ocr_time for r in successful_results) / len(successful_results):.2f}s")
        print(f"   Average post-processing time: {sum(r.postprocessing_time for r in successful_results) / len(successful_results):.2f}s")
        
        print(f"\n💾 MEMORY ANALYSIS:")
        print(f"   Average peak memory: {sum(r.peak_memory_mb for r in successful_results) / len(successful_results):.1f} MB")
        print(f"   Maximum peak memory: {max(r.peak_memory_mb for r in successful_results):.1f} MB")
        print(f"   Minimum peak memory: {min(r.peak_memory_mb for r in successful_results):.1f} MB")
        
        print(f"\n📝 TEXT EXTRACTION ANALYSIS:")
        print(f"   Total text lines extracted: {sum(r.text_lines_count for r in successful_results)}")
        print(f"   Average text lines per file: {sum(r.text_lines_count for r in successful_results) / len(successful_results):.1f}")
        print(f"   Total characters extracted: {sum(r.raw_text_length for r in successful_results)}")
        print(f"   Average characters per file: {sum(r.raw_text_length for r in successful_results) / len(successful_results):.1f}")
    
    # Show failed files
    failed_results = [r for r in results if not r.success]
    if failed_results:
        print(f"\n❌ FAILED FILES:")
        for result in failed_results:
            print(f"   {result.file_name}: {result.error_message}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile Surya OCR Pipeline")
    parser.add_argument("--files", nargs="+", default=["DEWA.png"], 
                       help="Test files to profile (default: DEWA.png)")
    parser.add_argument("--test-dir", default="test_bills/", 
                       help="Directory containing test files")
    
    args = parser.parse_args()
    
    profiler = SuryaPipelineProfiler(args.test_dir)
    
    # Prepare file paths
    test_files = []
    for file_name in args.files:
        file_path = Path(args.test_dir) / file_name
        if file_path.exists():
            test_files.append(file_path)
        else:
            print(f"⚠️  Warning: File {file_path} not found")
    
    if not test_files:
        print("❌ No valid test files found")
        sys.exit(1)
    
    # Run profiling
    results = profiler.run_comprehensive_profiling(test_files)
    
    # Save and display results
    save_results(results)
    print_summary(results)