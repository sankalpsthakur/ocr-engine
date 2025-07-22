#!/bin/bash

echo "🧹 Cleaning up junk files from OCR Engine project..."

# Remove Python cache files
echo "Removing Python cache files..."
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Remove macOS system files
echo "Removing macOS system files..."
find . -type f -name ".DS_Store" -delete 2>/dev/null

# Remove temporary and backup files
echo "Removing temporary and backup files..."
find . -type f -name "*.tmp" -delete 2>/dev/null
find . -type f -name "*.log" -delete 2>/dev/null
find . -type f -name "*.swp" -delete 2>/dev/null
find . -type f -name "*.swo" -delete 2>/dev/null
find . -type f -name "*~" -delete 2>/dev/null
find . -type f -name "*.bak" -delete 2>/dev/null

# Remove pytest cache
echo "Removing pytest cache..."
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null

# Remove mypy cache
echo "Removing mypy cache..."
find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null

# Remove coverage files
echo "Removing coverage files..."
find . -type f -name ".coverage" -delete 2>/dev/null
find . -type f -name "coverage.xml" -delete 2>/dev/null
find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null

# Remove egg-info directories
echo "Removing egg-info directories..."
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null

# Remove build directories
echo "Removing build directories..."
find . -type d -name "build" -exec rm -rf {} + 2>/dev/null
find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null

# Remove .env files that are copies (keeping the main .env if it exists)
echo "Removing duplicate .env files..."
find . -type f -name ".env.*" -delete 2>/dev/null

# Remove IDE-specific directories (optional - uncomment if needed)
# echo "Removing IDE directories..."
# rm -rf .vscode 2>/dev/null
# rm -rf .idea 2>/dev/null

# Clean up empty directories
echo "Removing empty directories..."
find . -type d -empty -delete 2>/dev/null

echo "✅ Cleanup complete!"

# Show disk space saved
if command -v du &> /dev/null; then
    echo ""
    echo "📊 Remaining project size:"
    du -sh . 2>/dev/null || echo "Could not calculate size"
fi