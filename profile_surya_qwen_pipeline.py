#!/usr/bin/env python3
"""
Comprehensive profiling script for Surya OCR + Qwen VL pipeline
Measures time and memory usage for the complete pipeline including structured data extraction
"""

import time
import psutil
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import tracemalloc
import gc

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from qwen_integration import QwenExtractor
except ImportError:
    print("Warning: Could not import Qwen integration module")
    QwenExtractor = None

try:
    from test.ocr_postprocessing import process_surya_output
except ImportError:
    print("Warning: Could not import post-processing module")
    def process_surya_output(text: str) -> str:
        return text

# Surya imports
try:
    from PIL import Image
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
except ImportError as e:
    print(f"Error importing Surya: {e}")
    sys.exit(1)

@dataclass
class PipelineResult:
    """Results from pipeline profiling"""
    file_name: str
    file_size_mb: float
    
    # Surya OCR metrics
    surya_load_time: float
    surya_ocr_time: float
    surya_peak_memory_mb: float
    raw_text_length: int
    text_lines_count: int
    
    # Post-processing metrics
    postprocessing_time: float
    processed_text_length: int
    
    # Qwen extraction metrics
    qwen_load_time: float
    qwen_extraction_time: float
    qwen_peak_memory_mb: float
    extracted_fields_count: int
    
    # Total pipeline metrics
    total_time: float
    peak_memory_mb: float
    memory_samples: List[float]

class MemoryProfiler:
    """Memory usage profiler"""
    
    def __init__(self):
        self.samples = []
        self.peak_memory = 0.0
        tracemalloc.start()
    
    def sample(self):
        """Take a memory sample"""
        # Get current memory usage
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.samples.append(memory_mb)
        self.peak_memory = max(self.peak_memory, memory_mb)
        
        # Also track tracemalloc
        current, peak = tracemalloc.get_traced_memory()
        return memory_mb
    
    def get_peak(self) -> float:
        """Get peak memory usage"""
        return self.peak_memory
    
    def get_samples(self) -> List[float]:
        """Get all memory samples"""
        return self.samples.copy()

class SuryaQwenProfiler:
    """Complete profiling system for Surya+Qwen pipeline"""
    
    def __init__(self):
        self.detection_predictor = None
        self.recognition_predictor = None
        self.qwen_extractor = None
        self.surya_load_time = 0.0
        self.qwen_load_time = 0.0
    
    def load_models(self, memory_profiler: MemoryProfiler):
        """Load both Surya and Qwen models"""
        print("🔧 Loading Surya OCR models...")
        memory_profiler.sample()
        
        start_time = time.time()
        
        # Load Surya models
        self.detection_predictor = DetectionPredictor()
        memory_profiler.sample()
        
        self.recognition_predictor = RecognitionPredictor()
        memory_profiler.sample()
        
        self.surya_load_time = time.time() - start_time
        print(f"✅ Surya models loaded in {self.surya_load_time:.2f} seconds")
        
        # Load Qwen model if available
        if QwenExtractor:
            print("🔧 Loading Qwen VL model...")
            memory_profiler.sample()
            
            self.qwen_extractor = QwenExtractor()
            self.qwen_load_time = self.qwen_extractor.load_model()
            memory_profiler.sample()
        else:
            print("⚠️ Qwen integration not available, skipping...")
            self.qwen_load_time = 0.0
    
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
            
            for page_prediction in predictions:
                for text_line in page_prediction.text_lines:
                    raw_text += text_line.text + "\n"
                    text_lines_count += 1
            
            processing_time = time.time() - start_time
            
            print(f"    ✅ OCR completed in {processing_time:.2f}s, extracted {len(raw_text)} characters, {text_lines_count} lines")
            return raw_text, processing_time, text_lines_count
            
        except Exception as e:
            print(f"    ❌ OCR failed: {e}")
            processing_time = time.time() - start_time
            return "", processing_time, 0
    
    def run_postprocessing(self, raw_text: str, memory_profiler: MemoryProfiler) -> Tuple[str, float]:
        """Run post-processing on OCR text"""
        print(f"  🧹 Running post-processing...")
        
        memory_profiler.sample()
        start_time = time.time()
        
        try:
            processed_text = process_surya_output(raw_text)
            processing_time = time.time() - start_time
            
            print(f"    ✅ Post-processing completed in {processing_time:.4f}s")
            return processed_text, processing_time
            
        except Exception as e:
            print(f"    ❌ Post-processing failed: {e}")
            processing_time = time.time() - start_time
            return raw_text, processing_time
    
    def run_qwen_extraction(self, processed_text: str, image_path: Path, 
                          memory_profiler: MemoryProfiler) -> Tuple[Dict, float]:
        """Run Qwen structured data extraction"""
        if not self.qwen_extractor:
            print("  ⚠️ Qwen extraction skipped (not available)")
            return {}, 0.0
        
        print(f"  🧠 Running Qwen structured extraction...")
        
        memory_profiler.sample()
        start_time = time.time()
        
        try:
            # Determine extraction type based on content
            text_lower = processed_text.lower()
            if any(word in text_lower for word in ['dewa', 'sewa', 'electricity', 'water', 'utility']):
                extraction_type = "bill"
            elif any(word in text_lower for word in ['invoice', 'bill to', 'invoice no']):
                extraction_type = "invoice"
            elif any(word in text_lower for word in ['receipt', 'thank you', 'total']):
                extraction_type = "receipt"
            else:
                extraction_type = "general"
            
            # Run extraction
            extracted_data, qwen_time = self.qwen_extractor.extract_structured_data(
                processed_text, str(image_path), extraction_type
            )
            
            # Count extracted fields
            fields_count = self._count_fields(extracted_data)
            
            total_time = time.time() - start_time
            memory_profiler.sample()
            
            print(f"    ✅ Qwen extraction completed in {total_time:.2f}s, extracted {fields_count} fields")
            return extracted_data, total_time
            
        except Exception as e:
            print(f"    ❌ Qwen extraction failed: {e}")
            total_time = time.time() - start_time
            return {"error": str(e)}, total_time
    
    def _count_fields(self, data: Dict) -> int:
        """Recursively count fields in extracted data"""
        if not isinstance(data, dict):
            return 0
        
        count = 0
        for key, value in data.items():
            if isinstance(value, dict):
                count += self._count_fields(value)
            elif isinstance(value, list):
                count += sum(self._count_fields(item) if isinstance(item, dict) else 1 for item in value)
            else:
                count += 1
        
        return count
    
    def profile_file(self, file_path: Path) -> PipelineResult:
        """Profile complete pipeline on a single file"""
        print(f"\n📄 Profiling {file_path.name}")
        
        # Initialize memory profiler
        memory_profiler = MemoryProfiler()
        memory_profiler.sample()
        
        # Get file size
        file_size_mb = file_path.stat().st_size / 1024 / 1024
        
        start_time = time.time()
        
        # Run Surya OCR
        raw_text, surya_time, text_lines = self.run_surya_ocr(file_path, memory_profiler)
        surya_peak_memory = memory_profiler.get_peak()
        
        # Run post-processing
        processed_text, postprocessing_time = self.run_postprocessing(raw_text, memory_profiler)
        
        # Run Qwen extraction
        extracted_data, qwen_time = self.run_qwen_extraction(processed_text, file_path, memory_profiler)
        
        # Calculate total metrics
        total_time = time.time() - start_time
        peak_memory = memory_profiler.get_peak()
        
        # Create result
        result = PipelineResult(
            file_name=file_path.name,
            file_size_mb=file_size_mb,
            surya_load_time=self.surya_load_time,
            surya_ocr_time=surya_time,
            surya_peak_memory_mb=surya_peak_memory,
            raw_text_length=len(raw_text),
            text_lines_count=text_lines,
            postprocessing_time=postprocessing_time,
            processed_text_length=len(processed_text),
            qwen_load_time=self.qwen_load_time,
            qwen_extraction_time=qwen_time,
            qwen_peak_memory_mb=peak_memory,
            extracted_fields_count=self._count_fields(extracted_data),
            total_time=total_time,
            peak_memory_mb=peak_memory,
            memory_samples=memory_profiler.get_samples()
        )
        
        print(f"✅ Total pipeline time: {total_time:.2f}s")
        print(f"📊 Peak memory usage: {peak_memory:.1f} MB")
        
        return result, extracted_data
    
    def profile_batch(self, file_paths: List[Path]) -> Tuple[List[PipelineResult], Dict]:
        """Profile pipeline on multiple files"""
        print(f"\n🚀 Starting batch profiling of {len(file_paths)} files")
        
        # Load models once
        global_memory_profiler = MemoryProfiler()
        self.load_models(global_memory_profiler)
        
        results = []
        extracted_data_all = {}
        
        for i, file_path in enumerate(file_paths, 1):
            print(f"\n📁 Processing file {i}/{len(file_paths)}: {file_path.name}")
            
            try:
                result, extracted_data = self.profile_file(file_path)
                results.append(result)
                extracted_data_all[file_path.name] = extracted_data
                
                # Force garbage collection between files
                gc.collect()
                
            except Exception as e:
                print(f"❌ Failed to process {file_path.name}: {e}")
                continue
        
        return results, extracted_data_all

def generate_analysis_report(results: List[PipelineResult], output_file: str = "surya_qwen_pipeline_analysis.json"):
    """Generate comprehensive analysis report"""
    if not results:
        print("❌ No results to analyze")
        return
    
    # Calculate statistics
    stats = {
        "summary": {
            "total_files": len(results),
            "successful_files": len([r for r in results if r.total_time > 0]),
            "average_file_size_mb": sum(r.file_size_mb for r in results) / len(results),
            "total_processing_time": sum(r.total_time for r in results),
            "average_total_time": sum(r.total_time for r in results) / len(results),
            "average_surya_time": sum(r.surya_ocr_time for r in results) / len(results),
            "average_qwen_time": sum(r.qwen_extraction_time for r in results) / len(results),
            "average_postprocessing_time": sum(r.postprocessing_time for r in results) / len(results),
            "average_peak_memory_mb": sum(r.peak_memory_mb for r in results) / len(results),
            "max_peak_memory_mb": max(r.peak_memory_mb for r in results),
            "total_text_characters": sum(r.raw_text_length for r in results),
            "total_extracted_fields": sum(r.extracted_fields_count for r in results),
        },
        "performance_breakdown": {
            "surya_percentage": (sum(r.surya_ocr_time for r in results) / sum(r.total_time for r in results)) * 100,
            "qwen_percentage": (sum(r.qwen_extraction_time for r in results) / sum(r.total_time for r in results)) * 100,
            "postprocessing_percentage": (sum(r.postprocessing_time for r in results) / sum(r.total_time for r in results)) * 100,
        },
        "detailed_results": [
            {
                "file_name": r.file_name,
                "file_size_mb": round(r.file_size_mb, 3),
                "total_time": round(r.total_time, 3),
                "surya_ocr_time": round(r.surya_ocr_time, 3),
                "qwen_extraction_time": round(r.qwen_extraction_time, 3),
                "postprocessing_time": round(r.postprocessing_time, 4),
                "peak_memory_mb": round(r.peak_memory_mb, 1),
                "raw_text_length": r.raw_text_length,
                "text_lines_count": r.text_lines_count,
                "extracted_fields_count": r.extracted_fields_count,
                "processing_rate_chars_per_sec": round(r.raw_text_length / r.total_time if r.total_time > 0 else 0, 1)
            }
            for r in results
        ],
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "surya_models": "Detection + Recognition",
            "qwen_model": "Qwen2.5-VL-3B-Instruct" if QwenExtractor else "Not Available"
        }
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n📊 Analysis Report:")
    print(f"Total files processed: {stats['summary']['total_files']}")
    print(f"Average processing time: {stats['summary']['average_total_time']:.2f}s")
    print(f"  - Surya OCR: {stats['summary']['average_surya_time']:.2f}s ({stats['performance_breakdown']['surya_percentage']:.1f}%)")
    print(f"  - Qwen extraction: {stats['summary']['average_qwen_time']:.2f}s ({stats['performance_breakdown']['qwen_percentage']:.1f}%)")
    print(f"  - Post-processing: {stats['summary']['average_postprocessing_time']:.4f}s ({stats['performance_breakdown']['postprocessing_percentage']:.1f}%)")
    print(f"Average peak memory: {stats['summary']['average_peak_memory_mb']:.1f} MB")
    print(f"Total characters extracted: {stats['summary']['total_text_characters']:,}")
    print(f"Total structured fields: {stats['summary']['total_extracted_fields']}")
    print(f"Report saved to: {output_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Profile Surya+Qwen OCR Pipeline')
    parser.add_argument('--files', nargs='+', help='Files to process')
    parser.add_argument('--test-dir', default='test_bills', help='Test directory')
    
    args = parser.parse_args()
    
    # Determine files to process
    if args.files:
        file_paths = []
        for file_arg in args.files:
            file_path = Path(args.test_dir) / file_arg
            if file_path.exists():
                file_paths.append(file_path)
            else:
                print(f"⚠️ File not found: {file_path}")
    else:
        # Process all supported files in test directory
        test_dir = Path(args.test_dir)
        if not test_dir.exists():
            print(f"❌ Test directory not found: {test_dir}")
            return
        
        file_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.pdf']:
            file_paths.extend(test_dir.glob(ext))
        
        if not file_paths:
            print(f"❌ No supported files found in {test_dir}")
            return
    
    # Run profiling
    profiler = SuryaQwenProfiler()
    results, extracted_data = profiler.profile_batch(file_paths)
    
    # Generate analysis
    generate_analysis_report(results)
    
    # Save extracted data
    with open("extracted_data_all.json", 'w') as f:
        json.dump(extracted_data, f, indent=2, default=str)
    
    print("\n✅ Profiling complete!")
    print("📄 Results saved to surya_qwen_pipeline_analysis.json")
    print("📄 Extracted data saved to extracted_data_all.json")

if __name__ == "__main__":
    main()