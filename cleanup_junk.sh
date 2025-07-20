#!/bin/bash
# Cleanup script to remove junk and temporary files

echo "Cleaning up junk files in OCR Engine..."

# Remove process and log files
echo "Removing process and log files..."
rm -f api.pid
rm -f api_server.log
rm -f install_log.txt

# Remove Python cache files
echo "Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null

# Remove duplicate test file in root
echo "Removing duplicate test files..."
rm -f test_api.py
rm -f test_ocr_endpoint.py

# Remove temporary test scripts in qwen_integration
echo "Removing temporary test scripts..."
rm -f qwen_integration/test_account_fix.py
rm -f qwen_integration/test_regex_only.py
rm -f qwen_integration/test_sewa2.py
rm -f qwen_integration/test_sewa_jpg.py
rm -f qwen_integration/test_sewa_ocr.py
rm -f qwen_integration/test_sewa_preprocess.py

# Remove debug scripts
echo "Removing debug scripts..."
rm -f qwen_integration/debug_ocr.py
rm -f qwen_integration/check_sewa_image.py

# Remove empty results directories
echo "Removing empty result directories..."
rm -rf results/surya/SEWA/

# Remove temporary output files
echo "Removing temporary output files..."
rm -f qwen_integration/sewa*.txt
rm -f qwen_integration/*.json
rm -f dewa_result.json
rm -f dewa_process_result.json

# Remove DS_Store files (macOS)
echo "Removing .DS_Store files..."
find . -name ".DS_Store" -delete 2>/dev/null

# Optional: Remove virtual environment (commented out by default)
# echo "Removing virtual environment..."
# rm -rf ocr_env/

echo "Cleanup complete!"
echo ""
echo "Files kept:"
echo "- Core application files"
echo "- Test suite in /test directory"
echo "- Test bills in /test_bills directory"
echo "- Ground truth data"
echo "- API and deployment configurations"
echo "- start_api.sh (kept for convenience)"
echo ""
echo "To remove virtual environment as well, uncomment the relevant lines in this script."